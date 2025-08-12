# app.py
# ChatRTL Analytics Copilot (Streamlit)
import os, re, json
from typing import Dict, List, Any, Tuple

import streamlit as st
import pandas as pd

from openai import OpenAI
from google.cloud import bigquery
from google.oauth2 import service_account

# ────────────────────────────────────────────────────────────────────────────
# Streamlit Page Setup
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChatRTL Analytics Copilot", page_icon="📊", layout="wide")

# ────────────────────────────────────────────────────────────────────────────
# Config & Secrets
# Preferred: set in Streamlit secrets UI (Cloud) or .streamlit/secrets.toml (local)
#   [gcp_service_account]  # paste your service account JSON as TOML keys
#   [bigquery]
#   project = "mmprod"
#   dataset = "mm_prod"
#   table   = "sales_inv_master"
#   OPENAI_API_KEY = "sk-..."
#   OPENAI_MODEL_SQL = "gpt-4o-mini"
#   OPENAI_MODEL_ANALYSIS = "gpt-4o-mini"
# ────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets.")
    st.stop()

OPENAI_MODEL_SQL = st.secrets.get("OPENAI_MODEL_SQL", os.getenv("OPENAI_MODEL_SQL", "gpt-4o-mini"))
OPENAI_MODEL_ANALYSIS = st.secrets.get("OPENAI_MODEL_ANALYSIS", os.getenv("OPENAI_MODEL_ANALYSIS", "gpt-4o-mini"))

# BigQuery identifiers
BQ_PROJECT = st.secrets.get("bigquery", {}).get("project", os.getenv("BQ_PROJECT", "mmprod"))
BQ_DATASET = st.secrets.get("bigquery", {}).get("dataset", os.getenv("BQ_DATASET", "mm_prod"))
BQ_TABLE   = st.secrets.get("bigquery", {}).get("table",   os.getenv("BQ_TABLE",   "sales_inv_master"))

if "gcp_service_account" not in st.secrets:
    st.error("Missing [gcp_service_account] block in secrets. Paste your service account JSON there.")
    st.stop()

# Build credentials from secrets (NOT from a file path)
GCP_CREDENTIALS = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp_service_account"])
)

# ────────────────────────────────────────────────────────────────────────────
# Business Context / Prompting
# ────────────────────────────────────────────────────────────────────────────
HUX_CONTEXT = """
You are the Hux Partners Analytics Copilot. You specialize in Walmart supplier
analytics, producing BigQuery SQL and interpreting results for executives.

Data source: mmprod.mm_prod.sales_inv_master (POS + inventory + movement).
Always consider these metric families when applicable:

Sales Performance
- pos_sales_this_year / pos_sales_last_year
- pos_quantity_this_year / pos_quantity_last_year
- YOY % change, top/bottom SKUs or suppliers

Inventory Health
- store_on_hand_quantity_this_year / _last_year
- instock_percentage_this_year / _last_year
- repl_instock_percentage_this_year / _last_year

Supply Chain / Movement
- mtr_received_quantity_this_year / _last_year
- mtr_transferred_quantity_this_year / _last_year
- store_in_transit_* , store_in_warehouse_*

Markdowns & Returns
- mkdn_qty_yoy_percentage, mkdwn_amt_yoy_percentage
- returns (defective/overstock/recall/vendor/return center)

Write concise, client-ready insights:
- Compare to last year where possible
- Call out notable highs/lows
- Flag operational risks (e.g., low in-stock %, high returns)
- State assumptions briefly at the end if needed
""".strip()

# ────────────────────────────────────────────────────────────────────────────
# Constants ported from your script
# ────────────────────────────────────────────────────────────────────────────
DEBUG = False
MAX_ROWS_DEFAULT = 200
OBS_ROWS = 50
ALLOWED_PREFIXES = {f"{BQ_PROJECT}.{BQ_DATASET}."}

RAW_COLS = [
    "supplier_code","walmart_calendar_week","store_number","prime_item_number",
    "dollar_per_str_with_sales_per_week_or_per_day_ly","dollar_per_str_with_sales_per_week_or_per_day_ty",
    "dollar_per_store_per_week_or_per_day_last_year","dollar_per_store_per_week_or_per_day_this_year",
    "units_per_str_with_sales_per_week_or_per_day_ly","units_per_str_with_sales_per_week_or_per_day_ty",
    "units_per_store_per_week_or_per_day_last_year","units_per_store_per_week_or_per_day_this_year",
    "si_mumd_qty_last_year","si_mumd_qty_this_year","mkdn_qty_yoy_percentage","mkdn_qty_yoy",
    "si_total_mumd_amount_last_year","si_total_mumd_amount_this_year","mkdwn_amt_yoy_percentage","mkdwn_amt_yoy",
    "total_store_customer_returns_amount_defective_last_year","total_store_customer_returns_amount_defective_this_year",
    "total_store_customer_returns_quantity_defective_last_year","total_store_customer_returns_quantity_defective_this_year",
    "total_store_customer_returns_amount_last_year","total_store_customer_returns_amount_this_year",
    "total_store_customer_returns_quantity_last_year","total_store_customer_returns_quantity_this_year",
    "backroom_adjustment_quantity","current_clearance_on_hand_quantity_last_year","current_clearance_on_hand_quantity_last_year_eop",
    "current_clearance_on_hand_quantity_this_year","current_clearance_on_hand_quantity_this_year_eop",
    "current_clearance_on_hand_retail_last_year","current_clearance_on_hand_retail_last_year_eop",
    "current_clearance_on_hand_retail_this_year","current_clearance_on_hand_retail_this_year_eop",
    "current_rollback_on_hand_quantity_last_year","current_rollback_on_hand_quantity_last_year_eop",
    "current_rollback_on_hand_quantity_this_year","current_rollback_on_hand_quantity_this_year_eop",
    "current_rollback_on_ha_
