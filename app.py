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
    "current_rollback_on_hand_retail_last_year","current_rollback_on_hand_retail_last_year_eop",
    "current_rollback_on_hand_retail_this_year","current_rollback_on_hand_retail_this_year_eop",
    "instock_percentage_last_year","instock_percentage_this_year","inventory_adjustment_quantity_last_year",
    "inventory_adjustment_quantity_this_year","mtr_received_cost_amount_last_year","mtr_received_cost_amount_this_year",
    "mtr_received_quantity_last_year","mtr_received_quantity_this_year","mtr_received_retail_amount_last_year",
    "mtr_received_retail_amount_this_year","mtr_transferred_cost_amount_last_year","mtr_transferred_cost_amount_this_year",
    "mtr_transferred_quantity_last_year","mtr_transferred_quantity_this_year","mtr_transferred_retail_amount_last_year",
    "mtr_transferred_retail_amount_this_year","repl_instock_percentage_last_year","repl_instock_percentage_last_year_eop",
    "repl_instock_percentage_this_year","repl_instock_percentage_this_year_eop","store_in_transit_quantity_last_year",
    "store_in_transit_quantity_this_year","store_in_transit_retail_last_year","store_in_transit_retail_this_year",
    "store_in_warehouse_quantity_last_year","store_in_warehouse_quantity_this_year","store_in_warehouse_retail_last_year",
    "store_in_warehouse_retail_this_year","store_on_hand_quantity_last_year","store_on_hand_quantity_last_year_eop",
    "store_on_hand_quantity_this_year","store_on_hand_quantity_this_year_eop","store_on_hand_retail_last_year",
    "store_on_hand_retail_last_year_eop","store_on_hand_retail_this_year","store_on_hand_retail_this_year_eop",
    "store_on_order_quantity_last_year","store_on_order_quantity_this_year","store_on_order_retail_last_year",
    "store_on_order_retail_this_year","tab_receiving_quantity_last_year","tab_receiving_quantity_this_year",
    "target_shelf_quantity_last_year","target_shelf_quantity_this_year","total_store_on_hand_adjustment_quantity",
    "traited_item_count_last_year","traited_item_count_this_year","traited_store_count_last_year","traited_store_count_this_year",
    "valid_store_count_last_year","valid_store_count_this_year","store_returns_quantity_defective_and_overstock_last_year",
    "store_returns_quantity_defective_and_overstock_this_year","store_returns_quantity_recall_last_year",
    "store_returns_quantity_recall_this_year","store_returns_quantity_to_dc_last_year","store_returns_quantity_to_dc_this_year",
    "store_returns_quantity_to_return_center_last_year","store_returns_quantity_to_return_center_this_year",
    "store_returns_quantity_to_vendor_last_year","store_returns_quantity_to_vendor_this_year",
    "home_office_recommended_retail_price_last_year","home_office_recommended_retail_price_this_year",
    "pos_quantity_last_year","pos_quantity_this_year","pos_sales_last_year","pos_sales_this_year",
    "pos_store_count_last_year","pos_store_count_this_year","scan_count_last_year","scan_count_this_year",
    "store_spcf_cost_amnt_last_year","store_spcf_cost_amnt_last_year_eop","store_spcf_cost_amnt_this_year",
    "store_spcf_cost_amnt_this_year_eop","tonnage_last_year","tonnage_this_year","store_specific_retail_amount_last_year",
    "store_specific_retail_amount_last_year_eop","store_specific_retail_amount_this_year","store_specific_retail_amount_this_year_eop"
]
KEY_COLS = {"supplier_code","walmart_calendar_week","store_number","prime_item_number"}

DDL_DML_PATTERN = re.compile(r"\b(CREATE|ALTER|DROP|TRUNCATE|MERGE|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)

# ────────────────────────────────────────────────────────────────────────────
# Clients (cached)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clients():
    oai = OpenAI(api_key=OPENAI_API_KEY)
    bq = bigquery.Client(credentials=GCP_CREDENTIALS, project=BQ_PROJECT)
    return oai, bq

client, bq = get_clients()

# ────────────────────────────────────────────────────────────────────────────
# Helpers ported from your script
# ────────────────────────────────────────────────────────────────────────────
def _d(msg: str):
    if DEBUG: st.write(f"[debug] {msg}")

def extract_cte_names(sql: str) -> List[str]:
    m = re.search(r"\bWITH\b(.*?)(?=\bSELECT\b)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    with_body = m.group(1)
    return [name.lower() for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", with_body, re.IGNORECASE)]

def is_query_safe(sql: str) -> Tuple[bool, str]:
    if DDL_DML_PATTERN.search(sql):
        return False, "DDL/DML is not allowed."

    cte_names = set(extract_cte_names(sql))
    refs = re.findall(r"(?:FROM|JOIN)\s+([`A-Za-z0-9_$.:-]+)", sql, re.IGNORECASE)
    for r in refs:
        clean = r.strip("`")
        if clean.lower() in cte_names:       # allow CTE refs
            continue
        if clean.startswith("("):            # subquery
            continue
        if "." not in clean:
            return False, f"Unqualified table reference '{r}'. Use fully qualified names or a CTE."
        ok = any(clean.startswith(prefix) for prefix in ALLOWED_PREFIXES)
        if not ok:
            return False, f"Table '{clean}' is outside allowed datasets."
    return True, ""

def enforce_limit(sql: str, max_rows: int) -> str:
    if re.search(r"\bLIMIT\s+\d+\b", sql, re.IGNORECASE): return sql
    if re.search(r"\bGROUP\s+BY\b|\bcount\(|\bsum\(|\bavg\(|\bmin\(|\bmax\(", sql, re.IGNORECASE): return sql
    return sql.rstrip().rstrip(";") + f" LIMIT {max_rows}"

def dry_run(sql: str):
    return bq.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))

def run_sql(sql: str, max_rows: int) -> List[Dict[str, Any]]:
    job = bq.query(sql)
    it = job.result(max_results=max_rows)
    return [dict(row) for row in it]

def build_typed_cte() -> str:
    select_lines = []
    for c in RAW_COLS:
        if c in KEY_COLS:
            if c == "walmart_calendar_week":
                select_lines.append(f"SAFE_CAST({c} AS INT64) AS {c}")
            else:
                select_lines.append(c)
        else:
            select_lines.append(
                f"SAFE_CAST(REGEXP_REPLACE({c}, r'[%,$]', '') AS FLOAT64) AS {c}"
            )
    select_block = ",\n    ".join(select_lines)
    return f"""WITH typed_sales AS (
  SELECT
    {select_block}
  FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
)"""

def call_openai(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content.strip()

def model_json_for_sql(question: str) -> Dict[str, Any]:
    sys = {"role": "system", "content": "Return ONLY valid JSON. No prose.\n" + HUX_CONTEXT}
    cte = build_typed_cte()
    user = {
        "role": "user",
        "content": f"""
Write **BigQuery Standard SQL** to answer the business question using only this dataset:

- Allowed prefix: {BQ_PROJECT}.{BQ_DATASET}
- **Always begin** the query with this CTE verbatim, then query from `typed_sales`:
{cte}

Rules:
- Do NOT use DDL/DML (CREATE/ALTER/DROP/INSERT/UPDATE/DELETE/MERGE).
- Prefer simple grouping, window functions, and clear aliases.
- If a time window like "past 52 weeks" is requested, interpret weeks using walmart_calendar_week (INT).
- Return ONLY JSON in the form: {{"sql": "<query>", "rationale":"<brief>"}}

Question:
\"\"\"{question}\"\"\"
""".strip(),
    }
    txt = call_openai(OPENAI_MODEL_SQL, [sys, user], temperature=0.1)
    m = re.search(r"\{.*\}\s*$", txt, re.DOTALL)
    raw = m.group(0) if m else txt
    return json.loads(raw.strip("`").strip())

def analyze_rows(question: str, rows: List[Dict[str, Any]], sql: str) -> str:
    sys = {"role": "system", "content": HUX_CONTEXT}
    preview = {
        "question": question,
        "sql": sql,
        "rows_preview": rows[:OBS_ROWS],
        "row_count_used": min(len(rows), OBS_ROWS),
        "note": f"Total rows fetched (capped at {MAX_ROWS_DEFAULT})."
    }
    user = {"role": "user", "content": json.dumps(preview, ensure_ascii=False)}
    return call_openai(OPENAI_MODEL_ANALYSIS, [sys, user], temperature=0.2)

# ────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────
st.title("📊 ChatRTL Analytics Copilot")
st.caption(f"Dataset: `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`")

with st.sidebar:
    st.subheader("Settings")
    max_rows = st.number_input("Max rows to fetch", min_value=10, max_value=5000, value=MAX_ROWS_DEFAULT, step=10)
    model_sql = st.text_input("OpenAI Model (SQL)", OPENAI_MODEL_SQL)
    model_analysis = st.text_input("OpenAI Model (Analysis)", OPENAI_MODEL_ANALYSIS)
    if model_sql != OPENAI_MODEL_SQL:
        OPENAI_MODEL_SQL = model_sql
    if model_analysis != OPENAI_MODEL_ANALYSIS:
        OPENAI_MODEL_ANALYSIS = model_analysis

    st.markdown("---")
    if st.button("🔎 Test BigQuery connection"):
        try:
            rows = run_sql("SELECT 1 AS ok", 1)
            st.success(f"BigQuery OK: {rows}")
        except Exception as e:
            st.error(f"BigQuery error: {e}")

tabs = st.tabs(["Ask a question", "Run manual SQL", "Diagnostics"])

with tabs[0]:
    st.subheader("Ask about sales/inventory/movement")
    prompt = st.text_area("Question", value="Top 10 SKUs by YoY POS sales growth in the last 8 weeks.", height=100)
    run_q = st.button("Generate & Run", type="primary")
    if run_q and prompt.strip():
        with st.status("Generating SQL with LLM...", expanded=False) as status:
            try:
                sql_obj = model_json_for_sql(prompt)
                sql = (sql_obj.get("sql") or "").strip()
                rationale = (sql_obj.get("rationale") or "").strip()
                if not sql:
                    st.error("Model did not return SQL.")
                else:
                    ok, msg = is_query_safe(sql)
                    if not ok:
                        st.error(f"Blocked: {msg}")
                        with st.expander("Proposed SQL"):
                            st.code(sql, language="sql")
                    else:
                        # Dry-run for validation
                        try:
                            dry_run(sql)
                        except Exception as e:
                            st.error(f"Dry-run error: {e}")
                            with st.expander("SQL"):
                                st.code(sql, language="sql")
                            status.update(state="error")
                            st.stop()

                        sql_exec = enforce_limit(sql, max_rows)
                        status.update(label="Running SQL on BigQuery...")
                        rows = run_sql(sql_exec, max_rows)

                        status.update(label="Analyzing result preview...", state="complete")

                        st.markdown("#### Executed SQL")
                        st.code(sql_exec, language="sql")

                        if rationale:
                            with st.expander("Why this query"):
                                st.write(rationale)

                        st.markdown("#### Results")
                        if rows:
                            df = pd.DataFrame(rows)
                            st.dataframe(df, use_container_width=True)
                            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="results.csv")
                        else:
                            st.info("(No rows)")

                        st.markdown("#### Observations")
                        try:
                            obs = analyze_rows(prompt, rows, sql_exec)
                            st.write(obs)
                        except Exception as e:
                            st.warning(f"Observation generation error: {e}")
            except Exception as e:
                st.error(f"Model failed to propose SQL: {e}")

with tabs[1]:
    st.subheader("Manual SQL (SELECT-only)")
    sql_input = st.text_area("SQL", value="SELECT 1 AS ok", height=160)
    if st.button("Run SQL"):
        sql = sql_input.strip()
        if not sql:
            st.warning("Enter a SQL query.")
        else:
            ok, msg = is_query_safe(sql)
            if not ok:
                st.error(f"Blocked: {msg}")
            else:
                try:
                    dry_run(sql)
                except Exception as e:
                    st.error(f"Dry-run error: {e}")
                    st.code(sql, language="sql")
                    st.stop()
                sql_exec = enforce_limit(sql, max_rows)
                try:
                    rows = run_sql(sql_exec, max_rows)
                except Exception as e:
                    st.error(f"Execution error: {e}")
                else:
                    st.markdown("#### Executed SQL")
                    st.code(sql_exec, language="sql")
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="results.csv")
                    else:
                        st.info("(No rows)")
                    st.markdown("#### Observations")
                    try:
                        obs = analyze_rows("Manual SQL", rows, sql_exec)
                        st.write(obs)
                    except Exception as e:
                        st.warning(f"(Observation generation error: {e})")

with tabs[2]:
    st.subheader("Diagnostics")
    st.write("Allowed table prefixes:", ", ".join(ALLOWED_PREFIXES))
    with st.expander("Typed CTE preview"):
        st.code(build_typed_cte(), language="sql")
