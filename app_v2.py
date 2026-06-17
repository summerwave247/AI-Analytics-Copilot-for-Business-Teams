"""Chat-with-CSVs MVP, v2 (refactored).

Architecture:
    user_q
      -> rewrite_followup(prior_turn, user_q)       # standalone question
      -> resolved_q
      -> stateless SQL agent.invoke(resolved_q)     # no MemorySaver
      -> fresh SQLCaptureHandler + backstop scoped to this run's messages only
      -> pick_final_sql(tool_calls)                 # last executed sql_db_query
      -> validate_sql + execute_validated -> df
      -> generate_grounded_answer(resolved_q, sql, df)   # LLM grounded only in df
      -> turn dict (turn_id, original_q, resolved_q, grounded_answer,
                    sql, result_preview, tool_calls, raw_agent_answer, error)

The displayed answer is never `result_messages[-1].content` — that is stored
only as `raw_agent_answer` for debugging.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Tunable performance knobs
# ----------------------------------------------------------------------------
AGENT_RECURSION_LIMIT = 12
RESULT_PREVIEW_ROWS = 50

import pandas as pd
import streamlit as st
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# ============================================================================
# CSV ingestion helpers
# ============================================================================

SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _qident(name: str) -> str:
    if not SAFE_IDENT.match(name):
        return '"' + name.replace('"', '""') + '"'
    return f'"{name}"'


def to_table_name(filename: str) -> str:
    base = re.sub(r"\.[Cc][Ss][Vv]$", "", filename)
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", base).strip("_").lower()
    return base or "table1"


def ensure_unique_table_name(engine: Engine, base: str) -> str:
    insp = inspect(engine)
    name, i = base, 2
    while insp.has_table(name):
        name = f"{base}_{i}"
        i += 1
    return name


def files_fingerprint(files: List[Any]) -> str:
    h = hashlib.sha256()
    for f in files:
        name = getattr(f, "name", "unknown")
        pos = f.tell() if hasattr(f, "tell") else 0
        data = f.read()
        if hasattr(f, "seek"):
            f.seek(pos)
        h.update(name.encode("utf-8"))
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def make_engine(db_path: str) -> Engine:
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode = WAL;")
        conn.exec_driver_sql("PRAGMA synchronous = NORMAL;")
        conn.exec_driver_sql("PRAGMA temp_store = MEMORY;")
    return engine


def load_csvs(files: List[Any], db_path: str) -> List[Tuple[str, pd.DataFrame]]:
    Path(db_path).unlink(missing_ok=True)
    engine = make_engine(db_path)
    loaded: List[Tuple[str, pd.DataFrame]] = []
    for f in files:
        df = pd.read_csv(f)
        if hasattr(f, "seek"):
            f.seek(0)
        for col in df.columns:
            if df[col].dtype == "object" and df[col].notna().any():
                sample = df[col].dropna().astype(str).head(20)
                if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        base = to_table_name(getattr(f, "name", "table.csv"))
        table = ensure_unique_table_name(engine, base)
        df.to_sql(table, con=engine, if_exists="replace", index=False)
        loaded.append((table, df.head(50)))
    return loaded


def create_generic_indexes(db_path: str) -> None:
    engine = make_engine(db_path)
    insp = inspect(engine)
    with engine.begin() as conn:
        for t in insp.get_table_names():
            for c in (col["name"] for col in insp.get_columns(t)):
                if c == "id" or c.endswith("_id"):
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS "
                        f"{_qident(f'idx_{t}_{c}')} ON {_qident(t)} ({_qident(c)})"
                    ))


def schema_overview(db_path: str) -> pd.DataFrame:
    engine = make_engine(db_path)
    insp = inspect(engine)
    rows = []
    for t in insp.get_table_names():
        for c in insp.get_columns(t):
            rows.append({"table": t, "column": c["name"],
                         "type": str(c["type"]), "nullable": c.get("nullable")})
    return pd.DataFrame(rows)


def compute_reference_date(db_path: str) -> Optional[str]:
    """Use MAX(activity_date) as the 'today' for synthetic data."""
    engine = make_engine(db_path)
    insp = inspect(engine)
    tables = insp.get_table_names()
    candidates = [
        ("activities", "activity_date"),
        ("opportunities", "close_date"),
        ("opportunities", "created_date"),
        ("support_tickets", "created_date"),
    ]
    for tbl, col in candidates:
        if tbl in tables and any(c["name"] == col for c in insp.get_columns(tbl)):
            try:
                with engine.connect() as conn:
                    val = conn.exec_driver_sql(
                        f"SELECT MAX({_qident(col)}) FROM {_qident(tbl)}"
                    ).scalar()
                if val:
                    return str(val)[:10]
            except Exception:
                continue
    return None


# ============================================================================
# SQL safety
# ============================================================================

_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "TRUNCATE", "VACUUM", "PRAGMA", "ATTACH", "DETACH", "REINDEX",
)


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def validate_sql(sql: str) -> str:
    """Return the cleaned single-statement SQL; raise ValueError if unsafe."""
    if not sql or not sql.strip():
        raise ValueError("Empty SQL.")
    cleaned = _strip_sql_comments(sql).strip().rstrip(";").strip()
    if ";" in cleaned:
        raise ValueError("Only a single statement is allowed.")
    head = cleaned.lstrip("(").lstrip().upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", cleaned, flags=re.IGNORECASE):
            raise ValueError(f"Forbidden keyword: {kw}")
    return cleaned


def execute_validated(sql: str, db_path: str) -> pd.DataFrame:
    cleaned = validate_sql(sql)
    engine = make_engine(db_path)
    return pd.read_sql_query(cleaned, engine)


# ============================================================================
# SQL capture
# ============================================================================

_SQL_LEADING_KEYWORDS = (
    "SELECT", "WITH", "PRAGMA", "EXPLAIN",
    "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
)


def _looks_like_sql(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    head = s.strip().lstrip("(").lstrip().upper()
    return any(head.startswith(kw) for kw in _SQL_LEADING_KEYWORDS)


def _is_readonly_sql(sql: str) -> bool:
    head = sql.strip().lstrip("(").lstrip().upper()
    return head.startswith("SELECT") or head.startswith("WITH")


def _extract_sql_from_value(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if _looks_like_sql(s):
            return s
        if s.startswith("{") and s.endswith("}"):
            try:
                return _extract_sql_from_value(json.loads(s))
            except Exception:
                return None
        return None
    if isinstance(val, dict):
        for key in ("query", "sql", "input", "tool_input", "__arg1"):
            if key in val:
                found = _extract_sql_from_value(val[key])
                if found:
                    return found
        for v in val.values():
            found = _extract_sql_from_value(v)
            if found:
                return found
    return None


class SQLCaptureHandler(BaseCallbackHandler):
    """One instance per turn. Records every tool call this run produces."""

    def __init__(self) -> None:
        self.tool_calls: List[dict] = []
        self.queries: List[str] = []

    def on_tool_start(self, serialized, input_str, *, inputs=None, **kwargs):
        name = ""
        if isinstance(serialized, dict):
            name = serialized.get("name") or ""
            if not name and isinstance(serialized.get("id"), list) and serialized["id"]:
                name = serialized["id"][-1]
        if not name:
            name = kwargs.get("name") or "<unknown>"
        raw = inputs if inputs not in (None, {}) else input_str
        self.tool_calls.append({"name": str(name), "input": raw})
        sql = _extract_sql_from_value(raw)
        if sql and sql not in self.queries:
            self.queries.append(sql)


def backstop_capture_scoped(new_messages: List[Any], capture: SQLCaptureHandler) -> None:
    """Backstop ONLY over the messages produced this turn. Never scan thread history."""
    def _key(name: str, val: Any) -> str:
        try:
            return f"{name}|{json.dumps(val, default=str, sort_keys=True)}"
        except Exception:
            return f"{name}|{val!r}"

    seen = {_key(tc["name"], tc["input"]) for tc in capture.tool_calls}
    for msg in new_messages:
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                name, args = tc.get("name", ""), tc.get("args", {})
            else:
                name, args = getattr(tc, "name", ""), getattr(tc, "args", {})
            k = _key(str(name), args)
            if k in seen:
                continue
            seen.add(k)
            capture.tool_calls.append({"name": str(name), "input": args})
            sql = _extract_sql_from_value(args)
            if sql and sql not in capture.queries:
                capture.queries.append(sql)


def pick_final_sql(tool_calls: List[dict], queries: List[str]) -> Optional[str]:
    """Choose the SQL that was actually executed for the final answer.

    Heuristic: the last `sql_db_query`-style tool call (excluding checker/list/schema
    variants). Fall back to the last SELECT/WITH-looking captured query.
    """
    for tc in reversed(tool_calls):
        name = (tc.get("name") or "").lower()
        if "query" not in name:
            continue
        if any(token in name for token in ("checker", "list", "schema")):
            continue
        sql = _extract_sql_from_value(tc.get("input"))
        if sql and _is_readonly_sql(sql):
            return sql
    for q in reversed(queries):
        if _is_readonly_sql(q):
            return q
    return None


# ============================================================================
# Agent + LLM (cached)
# ============================================================================

GTM_SEMANTIC_REFERENCE = """GTM SEMANTIC REFERENCE — Always follow these definitions when interpreting business terms.

Core entities:
- account: a customer or prospect company. Primary key: accounts.account_id.
- opportunity: a potential sales deal. Primary key: opportunities.opportunity_id.
- campaign: a marketing campaign or channel-level acquisition activity.
- activity: a customer-facing sales or success interaction.
- support ticket: a customer issue or service request.

Join keys:
- accounts.account_id = opportunities.account_id
- accounts.account_id = activities.account_id
- accounts.account_id = support_tickets.account_id
- campaigns is standalone unless a campaign/source key exists.

Segment definitions:
- Enterprise account = accounts.segment = 'Enterprise'
- Mid-market account = accounts.segment = 'Mid-Market'
- SMB account = accounts.segment = 'SMB'

Opportunity definitions:
- Open opportunity = opportunities.stage NOT IN ('Closed Won', 'Closed Lost')
- Closed won opportunity = opportunities.stage = 'Closed Won'
- Closed lost opportunity = opportunities.stage = 'Closed Lost'

Pipeline definitions:
- Pipeline amount = SUM(opportunities.amount)
- Open pipeline = SUM(opportunities.amount) where stage NOT IN ('Closed Won', 'Closed Lost')
- Weighted pipeline = SUM(opportunities.amount * opportunities.probability)
- Average deal size = AVG(opportunities.amount)

Campaign metric definitions (ALWAYS group by channel and aggregate before dividing):
- Cost per qualified lead by channel = SUM(campaigns.spend) / SUM(campaigns.qualified_leads), grouped by campaigns.channel
- Cost per opportunity by channel = SUM(campaigns.spend) / SUM(campaigns.opportunities_created), grouped by campaigns.channel
- Lead-to-qualified-lead conversion rate by channel = SUM(campaigns.qualified_leads) / SUM(campaigns.leads), grouped by campaigns.channel
- Qualified-lead-to-opportunity conversion rate by channel = SUM(campaigns.opportunities_created) / SUM(campaigns.qualified_leads), grouped by campaigns.channel

Important campaign metric rule:
- NEVER compute channel-level CPL using row-level spend / qualified_leads unless the user explicitly asks for individual campaign rows.
- ALWAYS use NULLIF for the denominator and exclude zero-denominator groups with HAVING.

Activity definitions:
- No recent customer activity = no row exists in activities for the same account_id within the requested time window.
- Stalled high-value opportunity = an open opportunity above the requested amount threshold with no customer activity in the requested recent time window.

Support ticket definitions:
- Unresolved support ticket = support_tickets.status NOT IN ('Resolved', 'Closed'). 'Closed' is treated as resolved / no longer open and MUST be excluded from unresolved-ticket results.
- High-severity support ticket = support_tickets.severity IN ('High', 'Critical').
- If the user asks "which accounts", distinguish between ticket-level rows and distinct account counts: prefer account-level aggregation, or use COUNT(DISTINCT account_id) when reporting distinct accounts.
- Do NOT call a ticket-level row count "number of accounts". A result with N ticket rows may represent fewer than N distinct accounts.

Date logic:
- This is synthetic data.
- Do NOT use DATE('now').
- Use the dataset reference date provided by the app: {ref_date}.
- For "past N days", use DATE('{ref_date}', '-N days') through DATE('{ref_date}').
- Example: if ref_date = '2025-12-31' and the user asks for past 21 days, the SQL should use DATE('2025-12-31', '-21 days') through DATE('2025-12-31').

SQL rules:
- Use SELECT or WITH only.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, DETACH, VACUUM, or REINDEX.
- Prefer explicit columns over SELECT *.
- Include ORDER BY for ranked outputs.
- Use LIMIT only when the user asks for top/bottom/small sample, or when returning examples.
- Never invent columns, tables, metrics, or fake examples."""


SYSTEM_PROMPT_TEMPLATE = """You are a SQL data analyst working against a SQLite database that contains one table per uploaded CSV.

DATASET REFERENCE DATE: {ref_date}
For ANY relative time window the user mentions ("past 21 days", "last week", "this quarter"), interpret it relative to the DATASET REFERENCE DATE above — NOT the real current date. Do NOT use DATE('now'). Use literal dates derived from {ref_date}, e.g. DATE('{ref_date}', '-21 days').

When a question needs data from multiple tables, infer join keys from column names and run JOINs. ONLY issue SELECT queries; never INSERT, UPDATE, DELETE, or DDL.
Select only the columns you need. If unsure about the schema, inspect it first.
After running the SQL you need, return your final answer briefly — the application will summarize the result table for the user separately, so you do not need to repeat all the rows.

{semantic_reference}"""


@st.cache_resource(show_spinner=False)
def build_llm(model_name: str):
    return ChatOllama(model=model_name, temperature=0, keep_alive="5m")


@st.cache_resource(show_spinner=False)
def build_agent(db_path: str, db_hash: str, model_name: str, ref_date: str):
    """Stateless SQL agent. Cache busts on db_hash, model, or reference date change."""
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
    llm = build_llm(model_name)
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    rd = ref_date or "unknown"
    system = SYSTEM_PROMPT_TEMPLATE.format(
        ref_date=rd,
        semantic_reference=GTM_SEMANTIC_REFERENCE.format(ref_date=rd),
    )
    agent = create_react_agent(llm, tools=tools, prompt=SystemMessage(content=system))
    return agent


# ============================================================================
# Follow-up rewriting and grounded answer generation
# ============================================================================

_REWRITE_SYSTEM = (
    "You rewrite vague follow-up analytics questions into ONE standalone question "
    "that preserves all prior filters. If the new question already specifies all "
    "criteria (tables, filters, time window) on its own, return it unchanged. "
    "Reply with the rewritten question only — no preface, no explanation."
)


def looks_like_followup(question: str) -> bool:
    """Cheap heuristic: only treat clearly-referential questions as follow-ups."""
    q = question.lower().strip()
    return (
        q.startswith(("what about", "how about", "what if", "and ", "now ", "then "))
        or any(token in q for token in ["them", "those", "same", "previous", "only", "instead"])
        or q in {"sort them", "group them", "show more"}
    )


def rewrite_followup(llm, prior_turn: Optional[dict], new_question: str) -> str:
    if not prior_turn:
        return new_question.strip()
    prior_q = prior_turn.get("resolved_question") or prior_turn.get("original_question") or ""
    prior_sql = prior_turn.get("final_sql") or ""
    user_msg = (
        f"PRIOR QUESTION:\n{prior_q}\n\n"
        f"PRIOR SQL (filters previously applied):\n{prior_sql or '(none captured)'}\n\n"
        f"FOLLOW-UP:\n{new_question.strip()}\n\n"
        "Rewritten standalone question:"
    )
    try:
        resp = llm.invoke([SystemMessage(content=_REWRITE_SYSTEM), HumanMessage(content=user_msg)])
        text_out = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        text_out = text_out.strip("`").strip().strip('"').strip()
        # Drop a leading label like "Rewritten:" if the model added one.
        text_out = re.sub(r"^(rewritten[: ]*|standalone[: ]*)", "", text_out, flags=re.IGNORECASE).strip()
        return text_out or new_question.strip()
    except Exception:
        return new_question.strip()


_GROUNDING_SYSTEM = (
    "You summarize a SQL query result for a business user. "
    "Rules: (1) Start with the exact row count. (2) Reference only the columns "
    "listed below — do NOT invent fields. (3) Cite at most 2 concrete examples "
    "using values from the rows shown. (4) If 0 rows, say so plainly. "
    "(5) Keep the answer to 1-4 sentences. No tables, no markdown headers."
)


def generate_grounded_answer(
    llm,
    question: str,
    sql: str,
    df: pd.DataFrame,
    fallback_answer: str,
) -> str:
    try:
        n = len(df)
        cols = list(df.columns)
        if n == 0:
            preview = "(no rows)"
        else:
            preview = df.head(5).to_string(index=False, max_colwidth=40)
        user_msg = (
            f"QUESTION:\n{question}\n\n"
            f"SQL EXECUTED:\n{sql}\n\n"
            f"RESULT: {n} row(s). Columns: {cols}\n\n"
            f"TOP {min(n, 5)} ROW(S):\n{preview}\n\n"
            "Answer:"
        )
        resp = llm.invoke([SystemMessage(content=_GROUNDING_SYSTEM), HumanMessage(content=user_msg)])
        out = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        return out or fallback_answer
    except Exception:
        return fallback_answer


# ============================================================================
# Canonical SQL routing — deterministic SQL for high-value demo metrics.
# Bypasses the LLM agent for known GTM questions, eliminating drift.
# ============================================================================

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _extract_days(question: str, default: int = 21) -> int:
    """Parse 'past N days', 'last N days', etc.; return default if absent."""
    m = re.search(r"(?:past|last|previous|over the (?:past|last))\s+(\d+)\s*days?",
                  question, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*[-\s]?\s*day(?:s)?\s+(?:window|period)",
                  question, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return default


def _extract_amount_threshold(question: str, default: int = 50000) -> int:
    """Parse $50K / 50K / $50000 / 50000. Ignore small numbers (day counts, etc.)."""
    for m in re.finditer(r"\$?\s*(\d+(?:[\.,]\d+)?)\s*([KkMm])?\b", question):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        suffix = (m.group(2) or "").lower()
        if suffix == "k":
            val *= 1_000
        elif suffix == "m":
            val *= 1_000_000
        if val >= 1_000:   # filter out day counts, top-N numbers, etc.
            return int(val)
    return default


def _detect_segment(q_lower: str) -> Optional[str]:
    if "enterprise" in q_lower:
        return "Enterprise"
    if "mid-market" in q_lower or "mid market" in q_lower:
        return "Mid-Market"
    if re.search(r"\bsmb\b", q_lower):
        return "SMB"
    return None


# ----- Canonical SQL builders ----------------------------------------------

def _sql_lowest_cpl_by_channel() -> str:
    return """SELECT
    channel,
    SUM(spend) * 1.0 / NULLIF(SUM(qualified_leads), 0) AS cost_per_qualified_lead,
    SUM(spend) AS total_spend,
    SUM(qualified_leads) AS total_qualified_leads
FROM campaigns
GROUP BY channel
HAVING SUM(qualified_leads) > 0
ORDER BY cost_per_qualified_lead ASC, channel ASC
LIMIT 1"""


def _sql_rank_channels_by_cpl() -> str:
    return """SELECT
    channel,
    SUM(spend) * 1.0 / NULLIF(SUM(qualified_leads), 0) AS cost_per_qualified_lead,
    SUM(spend) AS total_spend,
    SUM(qualified_leads) AS total_qualified_leads
FROM campaigns
GROUP BY channel
HAVING SUM(qualified_leads) > 0
ORDER BY cost_per_qualified_lead ASC, channel ASC"""


def _sql_cost_per_opportunity_by_channel() -> str:
    return """SELECT
    channel,
    SUM(spend) * 1.0 / NULLIF(SUM(opportunities_created), 0) AS cost_per_opportunity,
    SUM(spend) AS total_spend,
    SUM(opportunities_created) AS total_opportunities_created
FROM campaigns
GROUP BY channel
HAVING SUM(opportunities_created) > 0
ORDER BY cost_per_opportunity ASC, channel ASC"""


def _sql_lead_to_qualified_rate_by_channel() -> str:
    return """SELECT
    channel,
    SUM(qualified_leads) * 1.0 / NULLIF(SUM(leads), 0) AS lead_to_qualified_rate,
    SUM(leads) AS total_leads,
    SUM(qualified_leads) AS total_qualified_leads
FROM campaigns
GROUP BY channel
HAVING SUM(leads) > 0
ORDER BY lead_to_qualified_rate DESC, channel ASC"""


def _sql_stalled_open_opps(ref_date: str, days: int, amount: int,
                            segment: Optional[str]) -> str:
    segment_clause = f"\n  AND acc.segment = '{segment}'" if segment else ""
    return (
        "WITH reference_window AS (\n"
        "    SELECT\n"
        f"        DATE('{ref_date}', '-{days} days') AS window_start,\n"
        f"        DATE('{ref_date}') AS as_of_date\n"
        ")\n"
        "SELECT\n"
        "    o.opportunity_id,\n"
        "    o.account_id,\n"
        "    acc.segment,\n"
        "    acc.region,\n"
        "    acc.account_owner,\n"
        "    o.stage,\n"
        "    o.amount,\n"
        "    o.created_date,\n"
        "    o.close_date,\n"
        "    o.probability,\n"
        "    r.window_start,\n"
        "    r.as_of_date\n"
        "FROM opportunities o\n"
        "JOIN accounts acc\n"
        "    ON o.account_id = acc.account_id\n"
        "CROSS JOIN reference_window r\n"
        f"WHERE o.amount > {amount}\n"
        "  AND o.stage NOT IN ('Closed Won', 'Closed Lost')"
        f"{segment_clause}\n"
        "  AND NOT EXISTS (\n"
        "      SELECT 1\n"
        "      FROM activities a\n"
        "      WHERE a.account_id = o.account_id\n"
        "        AND DATE(a.activity_date) BETWEEN r.window_start AND r.as_of_date\n"
        "  )\n"
        "ORDER BY o.amount DESC, o.opportunity_id ASC"
    )


def _sql_enterprise_accounts_with_unresolved_high_sev() -> str:
    """Account-level aggregation: each row is ONE distinct Enterprise account.

    This matches the user phrasing "which accounts" — the row count IS the
    distinct-account count, so the grounded answer can correctly report
    "Y enterprise accounts" instead of confusing ticket rows with accounts.

    Note: 'Closed' is excluded along with 'Resolved' because in this dataset
    Closed tickets are operationally resolved (not open customer-risk items).
    """
    return """SELECT
    acc.account_id,
    acc.segment,
    acc.region,
    acc.account_owner,
    COUNT(st.ticket_id) AS unresolved_high_severity_ticket_count,
    MIN(st.created_date) AS oldest_ticket_date
FROM accounts acc
JOIN support_tickets st
    ON acc.account_id = st.account_id
WHERE acc.segment = 'Enterprise'
  AND st.status NOT IN ('Resolved', 'Closed')
  AND st.severity IN ('High', 'Critical')
GROUP BY
    acc.account_id,
    acc.segment,
    acc.region,
    acc.account_owner
ORDER BY
    unresolved_high_severity_ticket_count DESC,
    oldest_ticket_date ASC"""


def _sql_owners_by_open_pipeline() -> str:
    return """SELECT
    acc.account_owner,
    SUM(o.amount) AS total_open_pipeline,
    COUNT(o.opportunity_id) AS open_opportunity_count
FROM accounts acc
JOIN opportunities o
    ON acc.account_id = o.account_id
WHERE o.stage NOT IN ('Closed Won', 'Closed Lost')
GROUP BY acc.account_owner
ORDER BY total_open_pipeline DESC, acc.account_owner ASC"""


def canonical_sql_for_question(question: str, ref_date: str) -> Optional[str]:
    """Return canonical SQL for a recognized GTM question, else None.

    Matchers are ordered most-specific first so multi-keyword questions route
    to the more precise template.
    """
    q = question.lower().strip()

    # G — Enterprise + unresolved high-severity support tickets
    if (
        "enterprise" in q
        and "ticket" in q
        and ("unresolved" in q or "open" in q or "active" in q or "not resolved" in q)
        and ("high" in q or "critical" in q or "severity" in q)
    ):
        return _sql_enterprise_accounts_with_unresolved_high_sev()

    # H — Account owners ranked by open pipeline
    if ("account owner" in q or "owners" in q) and (
        "open pipeline" in q or ("pipeline" in q and "open" in q)
    ):
        return _sql_owners_by_open_pipeline()

    # E / F — Stalled high-value open opportunities (with optional segment filter)
    if "stalled" in q or (
        "open opportunit" in q
        and ("no" in q and ("activity" in q or "activities" in q))
    ):
        if not _DATE_RE.match(ref_date or ""):
            return None   # need a real reference date for the window
        amount = _extract_amount_threshold(question, 50_000)
        days = _extract_days(question, 21)
        segment = _detect_segment(q)
        return _sql_stalled_open_opps(ref_date, days, amount, segment)

    # D — Lead-to-qualified-lead conversion rate by channel
    if (
        "channel" in q
        and "conversion" in q
        and ("lead" in q and "qualified" in q)
    ):
        return _sql_lead_to_qualified_rate_by_channel()

    # C — Cost per opportunity by channel
    if "channel" in q and "cost per opportunit" in q:
        return _sql_cost_per_opportunity_by_channel()

    # B — Rank channels by CPL
    if (
        "channel" in q
        and ("cost per qualified lead" in q or "cpl" in q)
        and any(t in q for t in ("rank", "ranking", "list all", "all channels", "channels by"))
    ):
        return _sql_rank_channels_by_cpl()

    # A — Lowest CPL by channel
    if (
        "channel" in q
        and ("cost per qualified lead" in q or "cpl" in q)
        and any(t in q for t in ("lowest", "minimum", "best", "least"))
    ):
        return _sql_lowest_cpl_by_channel()

    return None


# ============================================================================
# Turn pipeline
# ============================================================================

def run_turn(
    agent,
    llm,
    db_path: str,
    prior_turn: Optional[dict],
    user_question: str,
    turn_id: str,
    ref_date: str,
) -> dict:
    turn: dict = {
        "turn_id": turn_id,
        "original_question": user_question,
        "resolved_question": user_question,
        "grounded_answer": "",
        "raw_agent_answer": "",
        "final_sql": None,
        "sql_queries": [],
        "tool_calls": [],
        "result_preview": None,   # {"columns","row_count","rows":[{...}]}
        "error": None,
        "timing": {
            "rewrite_sec": None,
            "agent_sec": None,
            "sql_exec_sec": None,
            "grounding_sec": None,
            "total_sec": None,
        },
    }
    t_total = time.perf_counter()

    # 1. Rewrite ONLY when there is prior context AND the question reads like a
    #    referential follow-up. Standalone questions skip the extra LLM call.
    if prior_turn is not None and looks_like_followup(user_question):
        t = time.perf_counter()
        resolved = rewrite_followup(llm, prior_turn, user_question)
        turn["timing"]["rewrite_sec"] = time.perf_counter() - t
    else:
        resolved = user_question.strip()
    turn["resolved_question"] = resolved

    # 1b. Canonical SQL routing — deterministic SQL for known GTM questions.
    #     Skips the agent entirely, eliminating LLM drift on definitions.
    canonical_sql = canonical_sql_for_question(resolved, ref_date)
    if canonical_sql:
        turn["final_sql"] = canonical_sql
        turn["sql_queries"] = [canonical_sql]
        turn["tool_calls"] = [{
            "name": "canonical_sql_router",
            "input": {"resolved_question": resolved, "sql": canonical_sql},
        }]
        t = time.perf_counter()
        try:
            df = execute_validated(canonical_sql, db_path)
            turn["result_preview"] = {
                "columns": list(df.columns),
                "row_count": int(len(df)),
                "rows": df.head(RESULT_PREVIEW_ROWS).to_dict(orient="records"),
            }
            turn["timing"]["sql_exec_sec"] = time.perf_counter() - t
            t2 = time.perf_counter()
            turn["grounded_answer"] = generate_grounded_answer(
                llm=llm,
                question=resolved,
                sql=canonical_sql,
                df=df,
                fallback_answer=f"The query returned {len(df):,} row(s).",
            )
            turn["timing"]["grounding_sec"] = time.perf_counter() - t2
        except Exception as e:
            turn["timing"]["sql_exec_sec"] = time.perf_counter() - t
            turn["error"] = f"Canonical SQL execution failed: {type(e).__name__}: {e}"
            turn["grounded_answer"] = "I could not execute the canonical SQL for this question."
        turn["timing"]["total_sec"] = time.perf_counter() - t_total
        return turn

    # 2. Stateless agent invocation. Fresh capture handler. No thread_id.
    t = time.perf_counter()
    capture = SQLCaptureHandler()
    new_messages: List[Any] = []
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=resolved)]},
            config={"callbacks": [capture], "recursion_limit": AGENT_RECURSION_LIMIT},
        )
        new_messages = result.get("messages", [])
        turn["raw_agent_answer"] = (
            new_messages[-1].content if new_messages else ""
        )
    except Exception as e:
        turn["error"] = f"{type(e).__name__}: {e}"
    turn["timing"]["agent_sec"] = time.perf_counter() - t

    # 3. Backstop ONLY over this turn's messages (no MemorySaver => no history bleed).
    backstop_capture_scoped(new_messages, capture)
    turn["tool_calls"] = list(capture.tool_calls)
    turn["sql_queries"] = list(capture.queries)

    # 4. Pick the SQL that was actually executed for the final answer.
    final_sql = pick_final_sql(turn["tool_calls"], turn["sql_queries"])
    turn["final_sql"] = final_sql

    # 5. Validate and execute deterministically.
    df: Optional[pd.DataFrame] = None
    if final_sql:
        t = time.perf_counter()
        try:
            df = execute_validated(final_sql, db_path)
            turn["result_preview"] = {
                "columns": list(df.columns),
                "row_count": int(len(df)),
                "rows": df.head(RESULT_PREVIEW_ROWS).to_dict(orient="records"),
            }
        except Exception as e:
            turn["error"] = (turn["error"] or "") + f" | SQL execution failed: {e}"
        turn["timing"]["sql_exec_sec"] = time.perf_counter() - t

    # 6. Ground the user-facing answer in the dataframe.
    if df is not None:
        t = time.perf_counter()
        turn["grounded_answer"] = generate_grounded_answer(
            llm, resolved, final_sql or "", df,
            fallback_answer=turn["raw_agent_answer"] or "(no answer)",
        )
        turn["timing"]["grounding_sec"] = time.perf_counter() - t
    else:
        # No SQL executed (e.g. schema question) — fall back to the agent's text.
        turn["grounded_answer"] = turn["raw_agent_answer"] or "(no answer)"

    turn["timing"]["total_sec"] = time.perf_counter() - t_total
    return turn


# ============================================================================
# Rendering
# ============================================================================

def _preview_df(preview: Optional[dict]) -> Optional[pd.DataFrame]:
    if not preview or not preview.get("rows"):
        return pd.DataFrame(columns=preview.get("columns", [])) if preview else None
    return pd.DataFrame(preview["rows"], columns=preview.get("columns"))


def render_turn(turn: dict, *, show_sql: bool, show_debug: bool) -> None:
    st.write(turn.get("grounded_answer") or "(no answer)")

    if turn.get("error"):
        st.warning(turn["error"])

    if turn.get("resolved_question") and turn["resolved_question"] != turn["original_question"]:
        st.caption(f"Interpreted as: *{turn['resolved_question']}*")

    if show_sql:
        final_sql = turn.get("final_sql")
        preview = turn.get("result_preview")
        if final_sql:
            with st.expander("Generated SQL", expanded=False):
                st.code(final_sql, language="sql")
                df = _preview_df(preview)
                if df is not None and preview is not None:
                    rc = preview.get("row_count", len(df))
                    st.caption(f"Result · {rc:,} row(s)" + (f" (showing first {len(df)})" if rc > len(df) else ""))
                    st.dataframe(df, use_container_width=True)
                else:
                    st.caption("Query was captured but not executed (validation failed or error above).")
        else:
            st.caption("No SQL query was captured for this response.")

    if show_debug:
        with st.expander(f"Tool Call Debug · {turn.get('turn_id', '?')}", expanded=False):
            st.markdown(f"**Original:** {turn.get('original_question', '')}")
            st.markdown(f"**Resolved:** {turn.get('resolved_question', '')}")
            timing = turn.get("timing") or {}
            if timing:
                def _fmt(v):
                    return f"{v:.2f}s" if isinstance(v, (int, float)) else "skipped"
                st.markdown(
                    "**Timing:** "
                    f"rewrite {_fmt(timing.get('rewrite_sec'))} · "
                    f"agent {_fmt(timing.get('agent_sec'))} · "
                    f"sql exec {_fmt(timing.get('sql_exec_sec'))} · "
                    f"grounding {_fmt(timing.get('grounding_sec'))} · "
                    f"total {_fmt(timing.get('total_sec'))}"
                )
            st.markdown(f"**Raw agent answer:**")
            st.code(turn.get("raw_agent_answer", "") or "(empty)", language="text")
            tcs = turn.get("tool_calls") or []
            if not tcs:
                st.write("No tool calls were recorded for this response.")
            else:
                st.markdown(f"**Tool calls ({len(tcs)}):**")
                for i, tc in enumerate(tcs, 1):
                    st.markdown(f"{i}. `{tc.get('name', '<unknown>')}`")
                    st.code(repr(tc.get("input")), language="python")


# ============================================================================
# Cached read-only views (keyed by db_path + db_hash so they refresh on upload)
# ============================================================================

@st.cache_data(show_spinner=False)
def cached_schema_overview(db_path: str, db_hash: str) -> pd.DataFrame:
    return schema_overview(db_path)


@st.cache_data(show_spinner=False)
def cached_reference_date(db_path: str, db_hash: str) -> Optional[str]:
    return compute_reference_date(db_path)


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Chat with CSVs", layout="wide")
st.title("Chat with your CSVs")

# Identity: DB session (stable across uploads of the same files) vs conversation.
if "db_session_id" not in st.session_state:
    st.session_state.db_session_id = uuid.uuid4().hex
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = uuid.uuid4().hex
if "history" not in st.session_state:
    st.session_state.history = []  # list of turn dicts
if "db_hash" not in st.session_state:
    st.session_state.db_hash = None
if "turn_counter" not in st.session_state:
    st.session_state.turn_counter = 0

DB_PATH = str(Path(tempfile.gettempdir()) / f"chat_csv_{st.session_state.db_session_id}.db")

with st.sidebar:
    st.markdown("### 1) Upload CSV files")
    uploaded_files = st.file_uploader(
        "Choose one or more CSV files", type=["csv"], accept_multiple_files=True
    )

    st.markdown("### 2) Model")
    model_name = st.selectbox(
        "Ollama model",
        options=["qwen3:8b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b"],
        index=0,
        help="Must already be pulled in your local Ollama.",
    )

    st.markdown("### 3) Options")
    keep_history = st.checkbox("Keep chat history", value=True)
    show_sql = st.checkbox("Show generated SQL", value=True)
    show_debug = st.checkbox("Show tool-call debug", value=False,
                             help="Temporary aid while developing; hide before final demo.")

    if st.button("Reset conversation"):
        st.session_state.history = []
        st.session_state.conversation_id = uuid.uuid4().hex
        st.session_state.turn_counter = 0
        # Note: db_session_id and DB_PATH remain stable — data is NOT reloaded.
        st.rerun()

    st.markdown("---")
    st.caption(
        "Try cross-table questions:\n"
        "- Total revenue by industry from closed-won opportunities\n"
        "- Open opportunities above $50K with no activity in the past 21 days\n"
        "- Then follow up: \"What about enterprise accounts only?\""
    )

if not uploaded_files:
    st.info("Upload one or more CSV files to start.")
    st.stop()

# (Re)build DB only when uploads change. Reset clears history but NOT the DB.
curr_hash = files_fingerprint(uploaded_files)
try:
    if st.session_state.db_hash != curr_hash:
        with st.spinner("Loading CSVs into SQLite..."):
            loaded = load_csvs(uploaded_files, DB_PATH)
            create_generic_indexes(DB_PATH)
            st.session_state.db_hash = curr_hash
            st.session_state.loaded_preview = loaded
            st.session_state.history = []
            st.session_state.turn_counter = 0
            build_agent.clear()
            build_llm.clear()
        st.success(f"Loaded {len(loaded)} table(s).")

    loaded = st.session_state.get("loaded_preview", [])
    if loaded:
        tabs = st.tabs([t for t, _ in loaded])
        for (t, df_preview), tab in zip(loaded, tabs):
            with tab:
                st.caption(f"Preview of `{t}` (first {len(df_preview)} rows)")
                st.dataframe(df_preview, use_container_width=True)

    with st.expander("Schema overview", expanded=False):
        st.dataframe(cached_schema_overview(DB_PATH, curr_hash), use_container_width=True)
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

ref_date = cached_reference_date(DB_PATH, curr_hash) or ""
if ref_date:
    st.caption(f"Dataset reference date (used for relative time windows): **{ref_date}**")

agent = build_agent(DB_PATH, curr_hash, model_name, ref_date)
llm = build_llm(model_name)

# Render prior turns.
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["original_question"])
    with st.chat_message("assistant"):
        render_turn(turn, show_sql=show_sql, show_debug=show_debug)

# Chat input.
user_q = st.chat_input("Ask anything about your data")
if user_q:
    with st.chat_message("user"):
        st.write(user_q)

    st.session_state.turn_counter += 1
    turn_id = f"t_{st.session_state.turn_counter:03d}"
    prior_turn = st.session_state.history[-1] if st.session_state.history else None

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            turn = run_turn(
                agent=agent,
                llm=llm,
                db_path=DB_PATH,
                prior_turn=prior_turn,
                user_question=user_q,
                turn_id=turn_id,
                ref_date=ref_date,
            )
        render_turn(turn, show_sql=show_sql, show_debug=show_debug)

    if keep_history:
        st.session_state.history.append(turn)
