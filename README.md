# ChatRTL
# Hux Partners Analytics Copilot (Streamlit)

Ask general CPG/seasonings market questions **or** query your BigQuery table
(`mmprod.mm_prod.sales_inv_master`) via natural language. The app will:
1) Generate BigQuery SQL (with guardrails),
2) Execute it live,
3) Show results, and
4) Write observations.

## ⚠️ Security
**Do not commit secrets** (OpenAI keys, GCP private keys) to GitHub.
Use Streamlit Cloud Secrets or environment variables.

## Quick Start (Local)

```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
