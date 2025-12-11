import hashlib
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber
import streamlit as st
from dateutil import parser as dtparser


LOGS: List[str] = []
EVIDENCE_ROOT = Path("evidence")
AUDIT_LOG = Path("audit.log")


def log_event(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    LOGS.append(entry)
    logging.info(entry)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_audit(action: str, payload: Dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.utcnow().isoformat() + "Z", "action": action, **payload}
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def extract_tables_from_pdf(
    pdf_bytes: bytes, page_start: int = 1, page_end: Optional[int] = None, stop_after_first_table: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    tables: List[pd.DataFrame] = []
    debug: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        debug.append(f"PDF opened; pages={len(pdf.pages)}")
        last_page = page_end or len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num < page_start or page_num > last_page:
                continue
            page_tables = page.extract_tables()
            debug.append(f"Page {page_num}: found {len(page_tables)} tables")
            for tbl_idx, raw_table in enumerate(page_tables, start=1):
                if not raw_table:
                    continue
                df = pd.DataFrame(raw_table)
                tables.append(df)
                debug.append(f"  Table {tbl_idx}: shape={df.shape}")
                if stop_after_first_table:
                    break
    if not tables:
        return pd.DataFrame(), debug
    combined = pd.concat(tables, ignore_index=True)
    return combined, debug


def parse_transactions_from_text(pdf_bytes: bytes, page_start: int = 1, page_end: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fallback text parsing for statements where pdfplumber.extract_tables() yields empty/garbled tables.
    It stitches wrapped lines, grabs lines that start with a date, then parses trailing numeric tokens
    as Money In, Money Out, Fee, Balance (last number is always treated as balance).
    """
    debug: List[str] = []
    rows: List[Dict] = []

    def parse_line(line: str) -> Optional[Dict]:
        m = re.match(r"(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<body>.+)", line)
        if not m:
            return None
        body = m.group("body")
        # Find monetary-looking tokens, requiring a decimal or explicit cent marker to avoid IDs being misread as amounts.
        #  - optional sign
        #  - digits with optional thousands spaces
        #  - decimal with 1-2 digits and optional 'c', OR whole cents like '12c'
        #  - optional trailing '*'
        amount_pattern = r"([+-]?\d[\d\s]*\.\d{1,2}c?|[+-]?\d+c)(\*)?"
        matches = list(re.finditer(amount_pattern, body, flags=re.IGNORECASE))
        if not matches:
            return None
        first_amt_pos = matches[0].start()
        description = body[:first_amt_pos].strip()
        if not description:
            description = body.strip()
        # Collect amounts, last token is balance
        tokens = [(match.group(1), bool(match.group(2))) for match in matches]
        balance_str = tokens[-1][0]
        remainder = tokens[:-1]
        money_in = money_out = fee = None
        if remainder:
            # Assign in Money In, Money Out, Fee order when present
            if len(remainder) >= 1:
                money_in = parse_amount(remainder[0][0])
            if len(remainder) >= 2:
                money_out = parse_amount(remainder[1][0])
            if len(remainder) >= 3:
                fee = parse_amount(remainder[2][0])
            # If only one value before balance, decide credit vs debit by description heuristics
            if len(remainder) == 1:
                value = parse_amount(remainder[0][0])
                desc_lower = description.lower()
                credit_keywords = ["received", "deposit", "pay in", "salary", "refund", "credit"]
                is_credit = any(k in desc_lower for k in credit_keywords)
                if is_credit:
                    money_in, money_out = value, None
                else:
                    money_in, money_out = None, value
        balance = parse_amount(balance_str)
        return {
            "date": m.group("date"),
            "description": description,
            "money_in": money_in,
            "money_out": money_out,
            "fee": fee,
            "balance": balance,
            "raw_line": line.strip(),
        }

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        last_page = page_end or len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num < page_start or page_num > last_page:
                continue
            text = page.extract_text() or ""
            debug.append(f"[text-fallback] Page {page_num} chars={len(text)}")
            stitched_lines: List[str] = []
            current = ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if re.match(r"\d{2}/\d{2}/\d{4}", line):
                    if current:
                        stitched_lines.append(current)
                    current = line
                else:
                    current = f"{current} {line}".strip()
            if current:
                stitched_lines.append(current)

            for line in stitched_lines:
                parsed = parse_line(line)
                if parsed:
                    rows.append(parsed)
        debug.append(f"[text-fallback] Parsed rows={len(rows)}")
    return pd.DataFrame(rows), debug


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # Drop rows that are completely empty across expected fields
    if set(["date", "description", "amount", "balance"]).issubset(df.columns):
        df = df[~((df["date"].astype(str).str.strip() == "") & (df["description"].astype(str).str.strip() == "") & df["amount"].isna() & df["balance"].isna())]

    # Drop probable header rows
    header_mask = df.apply(
        lambda row: row.astype(str).str.lower().str.contains("date").any()
        or row.astype(str).str.lower().str.contains("description").any(),
        axis=1,
    )
    df = df[~header_mask]
    df = df.dropna(how="all")

    # Remove obvious noise rows containing boilerplate/footer text
    noise_patterns = [
        "unique document no",
        "client care",
        "capitecbank.co.za",
        "24hr client care centre",
        "capitec bank",
        "vat registration",
        "page",
        "spending summary",
        "money in summary",
        "money out summary",
        "scheduled payments",
        "fee summary",
        "card subscriptions",
        "transaction history",
        "category",
    ]
    mask_noise = df.apply(lambda row: any(pat in " ".join(row.astype(str)).lower() for pat in noise_patterns), axis=1)
    df = df[~mask_noise]

    # If the first non-noise row looks like a header, use it
    header_row_idx = None
    for idx, row in df.iterrows():
        row_lower = [str(x).strip().lower() for x in row]
        if any("date" in cell for cell in row_lower) and any("description" in cell for cell in row_lower):
            header_row_idx = idx
            break

    if header_row_idx is not None:
        df.columns = [str(c).strip().lower() for c in df.loc[header_row_idx].tolist()]
        df = df[df.index > header_row_idx]
    else:
        normalized_cols = [str(c).strip().lower() for c in df.columns]
        expected_cols = {"date", "description", "money_in", "money_out", "fee", "balance"}
        if expected_cols.issubset(set(normalized_cols)):
            df.columns = normalized_cols
        else:
            df.columns = [f"col_{i}" for i in range(len(df.columns))]

    # Map known columns
    col_map = {name: None for name in ["date", "description", "money_in", "money_out", "fee", "balance"]}
    for col in df.columns:
        cl = str(col).lower()
        if "date" in cl:
            col_map["date"] = col
        elif "desc" in cl:
            col_map["description"] = col
        elif "money in" in cl or "credit" in cl or "in" == cl:
            col_map["money_in"] = col
        elif "money out" in cl or "debit" in cl or "out" == cl:
            col_map["money_out"] = col
        elif "fee" in cl:
            col_map["fee"] = col
        elif "balance" in cl:
            col_map["balance"] = col

    records = []

    def pick_date(cells: List) -> Optional[str]:
        for val in cells:
            norm = normalize_date_string(val)
            if not norm:
                continue
            for dayfirst in (True, False):
                try:
                    dtparser.parse(norm, dayfirst=dayfirst)
                    return norm
                except Exception:
                    continue
        return None

    def pick_amount(cells: List) -> Optional[float]:
        chosen = None
        for val in cells:
            amt = parse_amount(val)
            if amt is not None:
                chosen = amt  # keep last numeric to favor rightmost amount column
        return chosen

    def pick_description(cells: List, skip_vals: List) -> str:
        best = ""
        for val in cells:
            if val in skip_vals:
                continue
            txt = str(val).strip()
            if txt and len(txt) > len(best):
                best = txt
        return best

    for _, row in df.iterrows():
        if header_row_idx is not None:
            date_val = normalize_date_string(row.get(col_map["date"], ""))
            money_in = parse_amount(row.get(col_map["money_in"], None)) if col_map["money_in"] else None
            money_out = parse_amount(row.get(col_map["money_out"], None)) if col_map["money_out"] else None
            fee_val = parse_amount(row.get(col_map["fee"], None)) if col_map["fee"] else None
            balance_val = parse_amount(row.get(col_map["balance"], None)) if col_map["balance"] else None
            description_val = str(row.get(col_map["description"], "")).strip()
            # Compute signed amount: inflow positive, outflow negative
            amount_val = None
            if money_in not in (None, ""):
                amount_val = money_in
            if money_out not in (None, ""):
                amount_val = (amount_val or 0) - abs(money_out)
            records.append(
                {
                    "date": date_val or "",
                    "description": description_val,
                    "amount": amount_val,
                    "balance": balance_val,
                }
            )
        else:
            cells = list(row)
            date_val = pick_date(cells)
            amount_val = pick_amount(cells)
            description_val = pick_description(cells, skip_vals=[date_val, amount_val])
            records.append(
                {
                    "date": date_val or "",
                    "description": description_val,
                    "amount": amount_val,
                    "balance": None,
                }
            )

    return pd.DataFrame.from_records(records, columns=["date", "description", "amount", "balance"])


def parse_amount(value: str) -> Optional[float]:
    """
    Parse an amount string with support for:
    - Thousand separators (spaces or commas)
    - Trailing '*' markers
    - Cent notation (e.g., '39.5c' meaning 0.395)
    """
    if value is None:
        return None
    raw_txt = str(value).strip()
    if raw_txt == "":
        return None

    cents = False
    # Detect trailing cent marker
    if re.search(r"(?:c|cent)s?$", raw_txt, flags=re.IGNORECASE):
        cents = True

    txt = raw_txt.replace(",", "").replace(" ", "")
    # Remove trailing non-numeric markers like '*'
    txt = re.sub(r"[*]+$", "", txt)

    # Strip trailing 'c' or 'cent(s)' for parsing
    txt = re.sub(r"(?:c|cent)s?$", "", txt, flags=re.IGNORECASE)

    # Keep only digits, optional leading sign, and decimal point
    cleaned = re.sub(r"[^0-9\.\-]", "", txt)
    if cleaned in ("", ".", "-", "-.", ".-"):
        return None
    try:
        amount = float(cleaned)
        if cents:
            amount = amount / 100.0
        return amount
    except Exception:
        return None


def normalize_date_string(raw: str) -> str:
    """
    Try to isolate a plausible date substring even if the cell has leading row numbers
    or extra text, and add separators when an 8-digit compact date is detected.
    """
    if raw is None:
        return ""
    txt = str(raw).strip()
    if txt == "":
        return ""
    # Common patterns with separators
    for pattern in (r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})"):
        m = re.search(pattern, txt)
        if m:
            return m.group(1)
    # Compact 8-digit date, e.g., 20240305 or 05032024
    m = re.search(r"(\d{8})", txt)
    if m:
        digits = m.group(1)
        # Prefer day-first when ambiguous (common for statements)
        return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
    return txt


def compliance_checks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    notes: List[str] = []
    parsed_dates: List = []
    for idx, row in df.iterrows():
        row_notes = []
        date_val = normalize_date_string(row["date"])
        parsed = None
        if not (pd.isna(row["date"]) or date_val == ""):
            for dayfirst in (True, False):
                try:
                    parsed = dtparser.parse(date_val, dayfirst=dayfirst).date()
                    break
                except Exception:
                    continue
        if parsed is None:
            row_notes.append("missing or unparseable date")
            parsed_dates.append(None)
        else:
            parsed_dates.append(parsed)
        # Amount numeric
        amt = parse_amount(row["amount"])
        if amt is None:
            row_notes.append("amount not numeric")
        else:
            df.at[idx, "amount"] = amt
        # Description non-empty
        if pd.isna(row["description"]) or str(row["description"]).strip() == "":
            row_notes.append("missing description")
        notes.append("; ".join(row_notes) if row_notes else "ok")
    df["notes"] = notes
    df["parsed_date"] = parsed_dates
    return df


def run_pipeline(
    user_id: str, uploaded_file, page_start: int, page_end: Optional[int], stop_after_first_table: bool, raw_passthrough: bool
) -> Tuple[pd.DataFrame, List[str], Dict]:
    pdf_bytes = uploaded_file.getvalue()
    log_event(f"User {user_id} uploaded file '{uploaded_file.name}' (size={len(pdf_bytes)} bytes)")
    log_event("Starting extraction")
    df_raw, debug = extract_tables_from_pdf(pdf_bytes, page_start=page_start, page_end=page_end, stop_after_first_table=stop_after_first_table)

    if raw_passthrough:
        if df_raw.empty:
            debug.append("Raw passthrough enabled but extracted tables are empty; falling back to parsed mode.")
        else:
            debug.append("Raw passthrough enabled: skipping text fallback and normalization.")
            df_checked = df_raw.copy()
            ok_count = len(df_checked)
            issue_count = 0
            meta = {
                "rows": len(df_checked),
                "cols": df_checked.shape[1] if not df_checked.empty else 0,
                "ok": ok_count,
                "issues": issue_count,
                "mode": "raw_passthrough",
            }
            return df_checked, debug, meta

    df_raw_text, debug_text = parse_transactions_from_text(pdf_bytes, page_start=page_start, page_end=page_end)
    debug.extend(debug_text)

    def usable_rows(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        return int(
            frame.apply(
                lambda row: any(str(v).strip() for v in row if not (v is None or (isinstance(v, float) and pd.isna(v)))),
                axis=1,
            ).sum()
        )

    if usable_rows(df_raw_text) > usable_rows(df_raw):
        debug.append("Text parsing produced more usable rows; using text-derived transactions.")
        df_raw = df_raw_text

    log_event(f"Extraction complete; rows={len(df_raw)} cols={df_raw.shape[1] if not df_raw.empty else 0}")
    df_norm = normalize_dataframe(df_raw)
    log_event("Normalization applied (date, description, amount, balance)")
    df_checked = compliance_checks(df_norm)

    # Final safety: if parsed result is empty, fall back to text-derived raw
    if df_checked.empty and not df_raw_text.empty:
        debug.append("Parsed output empty; falling back to text-derived tables.")
        df_norm = normalize_dataframe(df_raw_text)
        df_checked = compliance_checks(df_norm)

    ok_count = sum(df_checked["notes"] == "ok") if not df_checked.empty else 0
    issue_count = sum(df_checked["notes"] != "ok") if not df_checked.empty else 0
    log_event(f"Compliance checks completed; ok={ok_count} issues={issue_count}")
    meta = {
        "rows": len(df_checked),
        "cols": df_checked.shape[1] if not df_checked.empty else 0,
        "ok": ok_count,
        "issues": issue_count,
        "mode": "parsed",
    }
    return df_checked, debug, meta


def main() -> None:
    st.set_page_config(page_title="PDF to Excel Bank Statement", layout="wide")
    # Hide Streamlit's deploy button for internal use
    st.markdown(
        """
        <style>
        [data-testid="stDeployButton"] {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("PDF to Excel Bank Statement")
    st.caption("Upload a bank statement PDF, extract to DataFrame, run basic checks, and export to Excel.")

    user_id = st.text_input("User ID", value="user-001")
    case_number = st.text_input("Case number (optional)", value="")
    uploaded_file = st.file_uploader("Upload bank statement (PDF)", type=["pdf"])
    raw_passthrough = st.checkbox("Exact table layout (no parsing/normalization)", value=False)

    page_start = 1
    page_end: Optional[int] = None
    stop_after_first_table = False
    page_count = None
    if uploaded_file:
        with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
            page_count = len(pdf.pages)
        page_start = st.number_input("Start page", min_value=1, max_value=page_count, value=1, step=1)
        page_end = st.number_input("End page", min_value=page_start, max_value=page_count, value=page_count, step=1)

    run_pressed = st.button("Run")

    log_placeholder = st.sidebar.empty()
    st.sidebar.markdown("### Action Log")

    if run_pressed:
        if not uploaded_file:
            st.error("Please upload a PDF first.")
        else:
            log_event(f"Run button pressed by {user_id} (case={case_number or 'n/a'})")
            try:
                run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                safe_user = re.sub(r"[^A-Za-z0-9_-]", "_", user_id or "user")
                safe_case = re.sub(r"[^A-Za-z0-9_-]", "_", case_number or "case")
                run_dir = EVIDENCE_ROOT / f"{run_ts}_{safe_user}"
                run_dir.mkdir(parents=True, exist_ok=True)

                pdf_bytes = uploaded_file.getvalue()
                pdf_path = run_dir / uploaded_file.name
                pdf_path.write_bytes(pdf_bytes)
                pdf_hash = sha256_bytes(pdf_bytes)
                write_audit("upload_saved", {"user": user_id, "case": case_number, "run_ts": run_ts, "file": str(pdf_path), "sha256": pdf_hash, "size_bytes": len(pdf_bytes)})
                log_event(f"Saved PDF to {pdf_path} (sha256={pdf_hash}, case={case_number or 'n/a'})")

                df, debug, meta = run_pipeline(
                    user_id,
                    uploaded_file,
                    page_start=page_start,
                    page_end=page_end,
                    stop_after_first_table=stop_after_first_table,
                    raw_passthrough=raw_passthrough,
                )
                for msg in debug:
                    log_event(msg)
                    write_audit("debug", {"user": user_id, "case": case_number, "run_ts": run_ts, "message": msg})

                if df.empty:
                    st.warning("No tables found in PDF. DataFrame is empty.")
                    write_audit("extraction_empty", {"user": user_id, "case": case_number, "run_ts": run_ts})
                else:
                    st.success(f"Extraction complete. Rows: {len(df)}")
                    st.dataframe(df.head(20), use_container_width=True)
                    write_audit("extraction_complete", {"user": user_id, "case": case_number, "run_ts": run_ts, **meta})

                    # Provide archived original PDF retrieval (read-only exposure in app)
                    with st.expander("Original PDF (archived)"):
                        st.caption("Read-only access; deletion is not available in-app. Escalate to admin for removal per policy.")
                        st.code(f"{pdf_path} (sha256={pdf_hash})")
                        st.download_button(
                            label="Download original PDF",
                            data=pdf_path.read_bytes(),
                            file_name=uploaded_file.name,
                            mime="application/pdf",
                        )

                    # Excel export in-memory
                    buf = io.BytesIO()
                    df.to_excel(buf, index=False)
                    buf.seek(0)
                    excel_path = run_dir / f"{Path(uploaded_file.name).stem}_processed.xlsx"
                    excel_path.write_bytes(buf.getvalue())
                    excel_hash = sha256_bytes(buf.getvalue())
                    write_audit("excel_saved", {"user": user_id, "case": case_number, "run_ts": run_ts, "file": str(excel_path), "sha256": excel_hash, **meta})
                    log_event(f"Excel saved to {excel_path} (sha256={excel_hash}, case={case_number or 'n/a'})")
                    st.download_button(
                        label="Download Excel",
                        data=buf,
                        file_name=f"{Path(uploaded_file.name).stem}_processed.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                    log_event("Excel file prepared for download")
            except Exception as exc:  # broad catch to surface errors in UI
                st.error(f"Failed to process PDF: {exc}")
                log_event(f"Error: {exc}")
                write_audit("error", {"user": user_id, "error": str(exc)})

    # Render logs
    log_placeholder.text("\n".join(LOGS[-200:]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
