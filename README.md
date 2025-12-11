# PDF to Excel Bank Statement

Minimal Streamlit app that uploads a PDF bank statement, extracts tables/text into a pandas DataFrame, runs simple row-level checks, and exports to Excel with logs.

## Quick start
1. Create/activate venv (optional)
   - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
2. Install deps: `pip install -r requirements.txt`
3. Run app: `streamlit run app.py`

## Notes
- Uses `pdfplumber` for table extraction, falls back to plain text.
- Logs actions to sidebar and console.
- Exports Excel with a `notes` column for compliance flags.
