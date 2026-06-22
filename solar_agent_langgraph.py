"""
solar_agent_langgraph.py
========================
Block 1A Solar+BESS Analytics Agent for the Jamnagar pilot project
(Reliance Industries Limited).

Scope: STRICTLY Block 1A (Monofacial Fixed-Tilt Solar + BESS, AC-coupled).
       Block 1B / 1C are explicitly out of scope.

Reads:
  - config.yaml             (Mongo URI, DB name, Google API key, plots dir)
  - block_1a_knowledge.yaml (meter inventory, tag catalog, plant context)

Database shape expected (per the SCADA gateway):
  {
    "meterId": "JAMNAGAR_VIRTUAL_GATEWAY_<short>",
    "meter": {
        "tms": <epoch milliseconds, UTC>,
        "<tag1>": <value>,
        "<tag2>": <value>,
        ...
    },
    ...other top-level fields like fnm, ignitionStatus, prt, fcm...
  }
"""

from __future__ import annotations
from typing import Optional, List
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression

# LangChain / LangGraph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

# ======================================================================
# CONFIG + KNOWLEDGE LOADING
# ======================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.yaml")
KNOWLEDGE_PATH = os.path.join(HERE, "block_1a_knowledge.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    knowledge = yaml.safe_load(f)

os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
MONGO_URI       = config["MONGO_URI"]
DB_NAME         = config["DB_NAME"]
COLLECTION_NAME = config["COLLECTION_NAME"]
PLOTS_DIR       = config["PLOTS_DIR"]
os.makedirs(PLOTS_DIR, exist_ok=True)

# Strip any "private" anchor blocks (keys starting with underscore) from meters
METERS: dict = {k: v for k, v in knowledge["meters"].items() if not k.startswith("_")}
METER_SHORT_NAMES: list = list(METERS.keys())
PLANT = knowledge["plant"]
BLOCK_1A_META = knowledge["block_1a"]

# Plant discriminator used in MongoDB queries (events collection is multi-tenant).
EID = PLANT.get("eid", "jamnagarsolar")

# ======================================================================
# MONGO CLIENT
# ======================================================================
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# ======================================================================
# TIMEZONE HELPERS — agent input/output is in IST; DB stores UTC epoch ms.
# ======================================================================
IST = ZoneInfo("Asia/Kolkata")


def parse_ist_date(date_str: str, end_of_day: bool = False) -> datetime:
    """Parse 'YYYY-MM-DD' (or full ISO) as IST. end_of_day -> 23:59:59.999."""
    dt = datetime.fromisoformat(date_str.strip())
    if end_of_day and dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999_000)
    return dt.replace(tzinfo=IST)


def to_epoch_ms(dt: datetime) -> int:
    """IST-aware datetime -> UTC epoch milliseconds."""
    return int(dt.timestamp() * 1000)


def epoch_ms_to_ist(ms) -> pd.Timestamp:
    """UTC epoch milliseconds -> tz-naive IST pd.Timestamp (for plotting/display)."""
    if ms is None:
        return pd.NaT
    return pd.to_datetime(int(ms), unit="ms", utc=True).tz_convert(IST).tz_localize(None)


# ======================================================================
# METER LOOKUP
# ======================================================================
def get_meter(short: Optional[str]) -> Optional[dict]:
    """Resolve meter by short name (case-insensitive). Returns dict with 'short' set."""
    if not short:
        return None
    target = short.strip().upper()
    for s, m in METERS.items():
        if s.upper() == target:
            return {"short": s, **m}
    return None


def detect_meter_id_field(full_id: str) -> Optional[str]:
    """Some documents store the meter identifier under 'meterId', others under 'did'.
    Probe both and return whichever has data for this meter (or None if neither does).
    Uses count_documents with limit=1 for a fast existence check.
    """
    try:
        if collection.count_documents({"meterId": full_id, "eid": EID}, limit=1) > 0:
            return "meterId"
    except Exception:
        pass
    try:
        if collection.count_documents({"did": full_id, "eid": EID}, limit=1) > 0:
            return "did"
    except Exception:
        pass
    return None


def probe_all_meters(verbose: bool = True) -> None:
    """Probe each Block 1A meter once at startup. Caches the discovered identifier
    field on the meter dict as `_id_field` so subsequent queries skip the probe.
    """
    if verbose:
        print("🔍 Probing MongoDB for Block 1A meter availability...")
    for short, m in METERS.items():
        field = detect_meter_id_field(m["full_id"])
        m["_id_field"] = field
        if verbose:
            if field:
                print(f"   ✓ {short:<8} via '{field}'  ({m['full_id']})")
            else:
                print(f"   ⚠ {short:<8} NO DATA       ({m['full_id']})")


# ======================================================================
# DATA LOADER
# ======================================================================
def load_meter_tag_data(meter_short: str, tags: List[str],
                        start_ist: datetime, end_ist: datetime) -> pd.DataFrame:
    """
    Load specified tags for a meter within an IST time window.
    Returns DataFrame with 'timestamp' (tz-naive IST) and one column per tag.
    """
    meter = get_meter(meter_short)
    if not meter:
        return pd.DataFrame()

    id_field = meter.get("_id_field")
    if not id_field:
        # Probe lazily if startup probe was skipped
        id_field = detect_meter_id_field(meter["full_id"])
        if not id_field:
            return pd.DataFrame()
        # Cache result back on the master METERS dict
        METERS[meter["short"]]["_id_field"] = id_field

    start_ms = to_epoch_ms(start_ist)
    end_ms   = to_epoch_ms(end_ist)

    projection = {"_id": 0, "time": 1}
    for t in tags:
        projection[f"meter.{t}"] = 1

    cursor = collection.find(
        {id_field: meter["full_id"],
         "eid": EID,
         "time": {"$gte": start_ms, "$lte": end_ms}},
        projection,
    ).sort("time", 1)

    rows = []
    for doc in cursor:
        m = doc.get("meter", {})
        row = {"timestamp": epoch_ms_to_ist(doc.get("time"))}
        for t in tags:
            row[t] = m.get(t)
        rows.append(row)

    return pd.DataFrame(rows)


def get_meter_time_range_ms(meter_short: str):
    """Return (first_ms, last_ms) for a meter, or (None, None)."""
    meter = get_meter(meter_short)
    if not meter:
        return None, None
    id_field = meter.get("_id_field") or detect_meter_id_field(meter["full_id"])
    if not id_field:
        return None, None
    METERS[meter["short"]]["_id_field"] = id_field

    first = collection.find_one(
        {id_field: meter["full_id"], "eid": EID},
        {"time": 1, "_id": 0},
        sort=[("time", 1)],
    )
    last = collection.find_one(
        {id_field: meter["full_id"], "eid": EID},
        {"time": 1, "_id": 0},
        sort=[("time", -1)],
    )
    if not first or not last:
        return None, None
    return first.get("time"), last.get("time")


# ======================================================================
# TOOLS
# ======================================================================

@tool
def list_meters() -> str:
    """List all Block 1A meters with their role and short description.
    Call this when you don't yet know which meter holds the data the user is asking about.
    """
    lines = [
        f"📋 **Block 1A Meter Inventory** — {len(METERS)} meters",
        f"Plant: {PLANT['name']} ({PLANT['location']}) — operator {PLANT['operator']}",
        f"Block 1A: {BLOCK_1A_META.get('solar_dc_mw')} MW DC solar + "
        f"{BLOCK_1A_META.get('bess_total_mwh')} MWh BESS "
        f"({BLOCK_1A_META.get('bess_container_count')} × "
        f"{BLOCK_1A_META.get('bess_container_mwh')} MWh containers)",
        "",
    ]
    for short, m in METERS.items():
        desc = (m.get("description") or "").strip().replace("\n", " ")
        lines.append(f"• **{short}** — {m['role']}: {desc}")
    return "\n".join(lines)


@tool
def describe_meter(meter_short_name: str) -> str:
    """Describe a Block 1A meter in detail: role, notable tags with units and friendly names,
    and the full list of available tag names.

    Args:
        meter_short_name: e.g. 'B1INV1', 'B1PQM2', 'B1PCS1', 'B1BCT1', 'WMS'.
    """
    m = get_meter(meter_short_name)
    if not m:
        return (f"❌ Meter '{meter_short_name}' is not in Block 1A.\n"
                f"Available: {', '.join(METER_SHORT_NAMES)}.")

    lines = [
        f"📟 **{m['short']}** — {m['role']}",
        f"MongoDB meterId: `{m['full_id']}`",
        "",
        (m.get("description") or "").strip(),
        "",
        "**Notable tags** (friendly name → tag field used in tools):",
    ]
    notable = m.get("notable_tags") or {}
    for friendly, tag in notable.items():
        lines.append(f"  • {friendly} → `{tag}`")

    all_tags = m.get("all_tags") or []
    lines.append("")
    lines.append(f"**All available tags** ({len(all_tags)} total):")
    lines.append(", ".join(f"`{t}`" for t in all_tags))

    if m.get("note"):
        lines.append("")
        lines.append(f"📝 Note: {m['note'].strip()}")

    return "\n".join(lines)


@tool
def get_meter_time_range(meter_short_name: str) -> str:
    """Return the earliest and latest IST timestamps available in MongoDB
    for a given Block 1A meter. Use this before querying or plotting to
    confirm data exists for your desired window.

    Args:
        meter_short_name: e.g. 'B1INV1'.
    """
    m = get_meter(meter_short_name)
    if not m:
        return f"❌ Meter '{meter_short_name}' not in Block 1A. Use: {', '.join(METER_SHORT_NAMES)}."

    first_ms, last_ms = get_meter_time_range_ms(m["short"])
    if first_ms is None:
        return f"⚠️ No documents found for {m['short']} ({m['full_id']}) in the database."

    first = epoch_ms_to_ist(first_ms)
    last  = epoch_ms_to_ist(last_ms)
    span_days = (last - first).total_seconds() / 86400.0
    return (f"📅 {m['short']} data availability (IST):\n"
            f"  • First record: {first}\n"
            f"  • Last record:  {last}\n"
            f"  • Span:         {span_days:.1f} days")


@tool
def query_tag(meter_short_name: str, tag: str, start_date: str,
              end_date: str, agg_type: str = "stats") -> str:
    """Query a single tag from one Block 1A meter over a date range, with optional aggregation.

    Args:
        meter_short_name: e.g. 'B1INV1', 'B1PQM2', 'WMS'.
        tag: Tag name WITHOUT the 'meter.' prefix, e.g. 'outputPower',
             'bessSOC', 'globalHorizontalIrradiance'.
        start_date: 'YYYY-MM-DD' (IST).
        end_date:   'YYYY-MM-DD' (IST). Use same as start_date for a single day.
        agg_type:   One of 'stats' (default — min/max/avg/sum/first/last/count),
                    'sum', 'avg', 'min', 'max', 'first', 'last'.
    """
    m = get_meter(meter_short_name)
    if not m:
        return f"❌ Meter '{meter_short_name}' not in Block 1A. Use: {', '.join(METER_SHORT_NAMES)}."
    if tag not in m.get("all_tags", []):
        return (f"❌ Tag '{tag}' not found on {m['short']}. "
                f"Call describe_meter('{m['short']}') to see the available tags.")

    try:
        start = parse_ist_date(start_date)
        end   = parse_ist_date(end_date, end_of_day=True)
    except Exception:
        return "❌ Use ISO dates like '2024-06-15'."

    df = load_meter_tag_data(m["short"], [tag], start, end)
    if df.empty:
        return f"⚠️ No data for {m['short']}.{tag} between {start_date} and {end_date} (IST)."

    s = pd.to_numeric(df[tag], errors="coerce").dropna()
    if s.empty:
        return (f"⚠️ {m['short']}.{tag} returned {len(df)} rows but none are numeric. "
                f"This tag may be a status code or non-numeric field.")

    a = (agg_type or "stats").lower().strip()
    if a == "stats":
        return (f"📊 **{m['short']}.{tag}** from {start_date} to {end_date} (IST)\n"
                f"  • Samples:  {len(s)}\n"
                f"  • Min:      {s.min():.3f}\n"
                f"  • Max:      {s.max():.3f}\n"
                f"  • Average:  {s.mean():.3f}\n"
                f"  • Sum:      {s.sum():.3f}\n"
                f"  • First:    {s.iloc[0]:.3f}  @ {df['timestamp'].iloc[0]}\n"
                f"  • Last:     {s.iloc[-1]:.3f}  @ {df['timestamp'].iloc[-1]}")

    ops = {"sum": s.sum(), "avg": s.mean(), "min": s.min(), "max": s.max(),
           "first": s.iloc[0], "last": s.iloc[-1]}
    if a not in ops:
        return "❌ agg_type must be one of: stats, sum, avg, min, max, first, last."
    return (f"📊 {a.upper()}({m['short']}.{tag}) "
            f"from {start_date} to {end_date} (IST) = {float(ops[a]):.3f} "
            f"  (n = {len(s)})")


@tool
def plot_tag(meter_short_name: str, tag: str, start_date: str, end_date: str) -> str:
    """Plot a single tag over time and save as an interactive Plotly HTML chart.

    Args:
        meter_short_name: e.g. 'B1INV1'.
        tag: Tag name without 'meter.' prefix, e.g. 'outputPower'.
        start_date, end_date: 'YYYY-MM-DD' (IST).
    """
    m = get_meter(meter_short_name)
    if not m:
        return f"❌ Meter '{meter_short_name}' not in Block 1A. Use: {', '.join(METER_SHORT_NAMES)}."
    if tag not in m.get("all_tags", []):
        return f"❌ Tag '{tag}' not on {m['short']}. Use describe_meter('{m['short']}')."

    try:
        start = parse_ist_date(start_date)
        end   = parse_ist_date(end_date, end_of_day=True)
    except Exception:
        return "❌ Use ISO dates."

    df = load_meter_tag_data(m["short"], [tag], start, end)
    if df.empty:
        return f"⚠️ No data for {m['short']}.{tag} between {start_date} and {end_date}."

    df = df.sort_values("timestamp")
    df[tag] = pd.to_numeric(df[tag], errors="coerce")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[tag], mode="lines",
                             name=f"{m['short']}.{tag}"))
    fig.update_layout(
        title=f"{m['short']}.{tag}  —  {start_date} to {end_date} (IST)",
        xaxis_title="Time (IST)",
        yaxis_title=tag,
        hovermode="x unified",
        template="plotly_white",
    )
    fp = os.path.join(PLOTS_DIR, f"{m['short']}_{tag}_{start_date}_to_{end_date}.html")
    fig.write_html(fp, include_plotlyjs="cdn", auto_open=False)
    vmin, vmax = df[tag].min(), df[tag].max()
    return (f"📈 Chart saved: {fp}\n"
            f"   {len(df)} points. Range: {vmin:.2f} … {vmax:.2f}")


@tool
def compare_tags(tag_specs: List[str], start_date: str, end_date: str) -> str:
    """Plot multiple tags on the same chart for comparison. Examples:
       - solar power vs irradiance: ['B1INV1:outputPower', 'WMS:globalHorizontalIrradiance']
       - BESS SOC across all 3 containers: ['B1BCT1:bessSOC', 'B1BCT2:bessSOC', 'B1BCT3:bessSOC']
       - Solar vs BESS vs POC: ['B1MFM2:activePower', 'B1PCS1:activePower', 'B1PQM2:activePower']

    Args:
        tag_specs: List of 'METER:TAG' strings. Use 1 to 6 specs.
                   The second tag uses a secondary Y-axis automatically
                   when exactly 2 specs are provided (good for dual-scale).
        start_date, end_date: 'YYYY-MM-DD' (IST).
    """
    if not tag_specs:
        return "❌ Provide at least one 'METER:TAG' spec."
    if len(tag_specs) > 6:
        return "❌ Compare at most 6 tags at once for readability."

    try:
        start = parse_ist_date(start_date)
        end   = parse_ist_date(end_date, end_of_day=True)
    except Exception:
        return "❌ Use ISO dates."

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plotted, errors = [], []

    for i, spec in enumerate(tag_specs):
        if ":" not in spec:
            errors.append(f"Bad spec '{spec}' — expected 'METER:TAG'.")
            continue
        meter_short, tag = spec.split(":", 1)
        m = get_meter(meter_short)
        if not m:
            errors.append(f"Meter '{meter_short}' not in Block 1A.")
            continue
        if tag not in m.get("all_tags", []):
            errors.append(f"Tag '{tag}' not on {m['short']}.")
            continue
        df = load_meter_tag_data(m["short"], [tag], start, end)
        if df.empty:
            errors.append(f"No data for {m['short']}.{tag}.")
            continue
        df = df.sort_values("timestamp")
        df[tag] = pd.to_numeric(df[tag], errors="coerce")
        use_secondary = (i == 1 and len(tag_specs) == 2)
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[tag], mode="lines",
                       name=f"{m['short']}.{tag}"),
            secondary_y=use_secondary,
        )
        plotted.append(f"{m['short']}.{tag} ({len(df)} pts)")

    if not plotted:
        return "⚠️ Nothing plotted. " + " ".join(errors)

    fig.update_layout(
        title=f"Comparison — {start_date} to {end_date} (IST)",
        xaxis_title="Time (IST)",
        hovermode="x unified",
        template="plotly_white",
    )
    safe = "_vs_".join(s.replace(":", "-").replace("/", "-") for s in tag_specs)[:80]
    fp = os.path.join(PLOTS_DIR, f"compare_{safe}_{start_date}_to_{end_date}.html")
    fig.write_html(fp, include_plotlyjs="cdn", auto_open=False)
    out = f"📈 Comparison chart saved: {fp}\n   Plotted: {', '.join(plotted)}"
    if errors:
        out += "\n⚠️ Skipped: " + "; ".join(errors)
    return out


@tool
def tag_correlation(x_meter: str, x_tag: str, y_meter: str, y_tag: str,
                    start_date: str, end_date: str) -> str:
    """Fit a linear regression y_tag ~ x_tag between two Block 1A tags
    (possibly from different meters), aligning samples on the nearest
    timestamp (within 1 minute). Returns R², slope, intercept, sample count.

    Typical uses:
      • Solar power vs GHI:   ('B1INV1','outputPower') vs ('WMS','globalHorizontalIrradiance')
      • PV vs module temp:    ('WMS','moduleTemperatureSensor01') vs ('B1INV1','outputPower')
      • BESS SOC vs power:    ('B1PCS1','activePower') vs ('B1BCT1','bessSOC')
    """
    xm, ym = get_meter(x_meter), get_meter(y_meter)
    if not xm:
        return f"❌ x meter '{x_meter}' not in Block 1A."
    if not ym:
        return f"❌ y meter '{y_meter}' not in Block 1A."
    if x_tag not in xm.get("all_tags", []):
        return f"❌ Tag '{x_tag}' not on {xm['short']}."
    if y_tag not in ym.get("all_tags", []):
        return f"❌ Tag '{y_tag}' not on {ym['short']}."

    try:
        start = parse_ist_date(start_date)
        end   = parse_ist_date(end_date, end_of_day=True)
    except Exception:
        return "❌ Use ISO dates."

    df_x = load_meter_tag_data(xm["short"], [x_tag], start, end).rename(columns={x_tag: "x"})
    df_y = load_meter_tag_data(ym["short"], [y_tag], start, end).rename(columns={y_tag: "y"})
    if df_x.empty or df_y.empty:
        return "⚠️ No data for at least one of the tags in this range."

    df_x = df_x.sort_values("timestamp")
    df_y = df_y.sort_values("timestamp")
    merged = pd.merge_asof(df_x, df_y, on="timestamp",
                           tolerance=pd.Timedelta("1min"),
                           direction="nearest")
    merged["x"] = pd.to_numeric(merged["x"], errors="coerce")
    merged["y"] = pd.to_numeric(merged["y"], errors="coerce")
    merged = merged.dropna(subset=["x", "y"])
    if len(merged) < 5:
        return f"⚠️ Too few aligned numeric samples ({len(merged)}) — need at least 5."

    X = merged[["x"]].values
    y = merged["y"].values
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return (f"📐 Linear relationship\n"
            f"   {ym['short']}.{y_tag} ≈ "
            f"{model.coef_[0]:.4f} × {xm['short']}.{x_tag} + {model.intercept_:.3f}\n"
            f"   R² = {r2:.3f}   (n = {len(merged)} aligned samples)\n"
            f"   ⚠️ Linear assumption — verify visually with compare_tags or a scatter.")


@tool
def block_1a_daily_summary(date: str) -> str:
    """Compute a same-day energy summary for Block 1A on a given IST date,
    pulling the right tag from the right meter:
      • Solar generation       — B1INV1.energyToday, B1MFM2.dailyActiveEnergyExport
      • BESS charge/discharge  — B1PCS1.dailyAcEnergyImport / dailyAcEnergyExport
      • POC export to grid     — B1PQM2.dailyActiveEnergyExport
      • Auxiliary consumption  — B1MFM1.dailyActiveEnergyImport
      • Peak GHI & module temp — WMS

    Also performs a quick energy-balance sanity check.

    Args:
        date: 'YYYY-MM-DD' (IST).
    """
    try:
        start = parse_ist_date(date)
        end   = parse_ist_date(date, end_of_day=True)
    except Exception:
        return "❌ Use ISO date YYYY-MM-DD."

    def last_value(meter_short: str, tag: str):
        df = load_meter_tag_data(meter_short, [tag], start, end)
        if df.empty:
            return None
        s = pd.to_numeric(df[tag], errors="coerce").dropna()
        return None if s.empty else float(s.iloc[-1])

    def max_value(meter_short: str, tag: str):
        df = load_meter_tag_data(meter_short, [tag], start, end)
        if df.empty:
            return None
        s = pd.to_numeric(df[tag], errors="coerce").dropna()
        return None if s.empty else float(s.max())

    inv_energy = last_value("B1INV1", "energyToday")
    sol_export = last_value("B1MFM2", "dailyActiveEnergyExport")
    bess_chg   = last_value("B1PCS1", "dailyAcEnergyImport")
    bess_dis   = last_value("B1PCS1", "dailyAcEnergyExport")
    poc_export = last_value("B1PQM2", "dailyActiveEnergyExport")
    poc_import = last_value("B1PQM2", "dailyActiveEnergyImport")
    aux_cons   = last_value("B1MFM1", "dailyActiveEnergyImport")
    peak_ghi   = max_value("WMS", "globalHorizontalIrradiance")
    peak_modT  = max_value("WMS", "moduleTemperatureSensor01")
    peak_amb   = max_value("WMS", "ambientTemperature")

    def fmt(v, unit, prec=2):
        return f"{v:.{prec}f} {unit}" if v is not None else "n/a"

    lines = [
        f"📋 **Block 1A Daily Summary — {date} (IST)**",
        "",
        "🌞 **Solar generation**",
        f"   • Inverter daily energy (B1INV1.energyToday):           {fmt(inv_energy, 'kWh')}",
        f"   • Solar MFM daily export (B1MFM2.dailyExport):          {fmt(sol_export, 'kWh')}",
        "",
        "🔋 **BESS dispatch (B1PCS1)**",
        f"   • Daily AC charge (import):                              {fmt(bess_chg, 'kWh')}",
        f"   • Daily AC discharge (export):                           {fmt(bess_dis, 'kWh')}",
        f"   • Net BESS (discharge − charge):                         "
        f"{fmt((bess_dis - bess_chg) if (bess_dis is not None and bess_chg is not None) else None, 'kWh')}",
        "",
        "⚡ **Plant output**",
        f"   • POC daily export to grid (B1PQM2):                     {fmt(poc_export, 'kWh')}",
        f"   • POC daily import from grid (B1PQM2):                   {fmt(poc_import, 'kWh')}",
        f"   • Auxiliary consumption (B1MFM1):                        {fmt(aux_cons, 'kWh')}",
        "",
        "🌤  **Weather (WMS)**",
        f"   • Peak GHI:                                              {fmt(peak_ghi, 'W/m²')}",
        f"   • Peak module temperature (sensor 01):                   {fmt(peak_modT, '°C')}",
        f"   • Peak ambient temperature:                              {fmt(peak_amb, '°C')}",
    ]

    # Energy balance: solar + bess_discharge ≈ poc_export + aux + bess_charge
    if all(v is not None for v in [sol_export, poc_export, bess_chg, bess_dis, aux_cons]):
        balance = sol_export + bess_dis - bess_chg - aux_cons - poc_export
        verdict = "✅ balanced" if abs(balance) < 100 else "⚠️ off — check meter scaling or reset times"
        lines += [
            "",
            "🔎 **Energy balance check**",
            "   (solar export + BESS discharge − BESS charge − aux − POC export)",
            f"   Residual = {balance:+.2f} kWh   {verdict}",
        ]

    return "\n".join(lines)


@tool
def bess_soc_profile(container: str, start_date: str, end_date: str) -> str:
    """Plot SOC profile for one or all three BESS containers in Block 1A,
    with min/max/avg/final stats per container.

    Args:
        container: 'B1BCT1', 'B1BCT2', 'B1BCT3', or 'all' for all three on one chart.
        start_date, end_date: 'YYYY-MM-DD' (IST).
    """
    try:
        start = parse_ist_date(start_date)
        end   = parse_ist_date(end_date, end_of_day=True)
    except Exception:
        return "❌ Use ISO dates."

    container = container.strip().upper()
    if container == "ALL":
        targets = ["B1BCT1", "B1BCT2", "B1BCT3"]
    elif container in {"B1BCT1", "B1BCT2", "B1BCT3"}:
        targets = [container]
    else:
        return "❌ container must be one of B1BCT1, B1BCT2, B1BCT3, or 'all'."

    fig = go.Figure()
    stats_lines: list = []
    for c in targets:
        df = load_meter_tag_data(c, ["bessSOC"], start, end)
        if df.empty:
            stats_lines.append(f"   • {c}: no data")
            continue
        df = df.sort_values("timestamp")
        df["bessSOC"] = pd.to_numeric(df["bessSOC"], errors="coerce")
        s = df["bessSOC"].dropna()
        if s.empty:
            stats_lines.append(f"   • {c}: no numeric SOC values")
            continue
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bessSOC"], mode="lines", name=c))
        stats_lines.append(
            f"   • {c}: min {s.min():.1f}%  max {s.max():.1f}%  "
            f"avg {s.mean():.1f}%  final {s.iloc[-1]:.1f}%  (n={len(s)})"
        )

    if not fig.data:
        return "⚠️ No SOC data for the requested container(s) and range.\n" + "\n".join(stats_lines)

    fig.update_layout(
        title=f"BESS SOC profile — {start_date} to {end_date} (IST)",
        xaxis_title="Time (IST)",
        yaxis_title="SOC (%)",
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(range=[0, 100]),
    )
    fp = os.path.join(PLOTS_DIR, f"bess_soc_{container}_{start_date}_to_{end_date}.html")
    fig.write_html(fp, include_plotlyjs="cdn", auto_open=False)
    return f"🔋 SOC chart saved: {fp}\n" + "\n".join(stats_lines)


@tool
def inspect_sample_document(meter_short_name: str) -> str:
    """Fetch ONE recent sample document from MongoDB for a Block 1A meter,
    so you can see the actual document structure and confirm tag names / values.
    Use this when a query unexpectedly returns no data or you want to verify
    what's actually stored.

    Args:
        meter_short_name: e.g. 'B1INV1', 'B1BCT1', 'WMS'.
    """
    import json
    m = get_meter(meter_short_name)
    if not m:
        return f"❌ Meter '{meter_short_name}' not in Block 1A. Use: {', '.join(METER_SHORT_NAMES)}."
    id_field = m.get("_id_field") or detect_meter_id_field(m["full_id"])
    if not id_field:
        return (f"⚠️ No documents found for {m['short']} under either 'meterId' or 'did' "
                f"with eid='{EID}'.\n   Checked: {m['full_id']}")
    METERS[m["short"]]["_id_field"] = id_field

    doc = collection.find_one(
        {id_field: m["full_id"], "eid": EID},
        {"_id": 0},
        sort=[("time", -1)],  # most recent
    )
    if not doc:
        return f"⚠️ No documents found for {m['short']} (id field '{id_field}', eid '{EID}')."
    t = doc.get("time")
    ts_ist = epoch_ms_to_ist(t) if t else "n/a"
    snippet = json.dumps(doc, indent=2, default=str)
    if len(snippet) > 2500:
        snippet = snippet[:2500] + "\n  ... [truncated]"
    return (f"📄 Most-recent sample document for {m['short']} "
            f"(identifier field: '{id_field}', time = {ts_ist} IST):\n"
            f"```json\n{snippet}\n```")


# ======================================================================
# TOOL REGISTRATION
# ======================================================================
TOOLS = [
    list_meters,
    describe_meter,
    get_meter_time_range,
    inspect_sample_document,
    query_tag,
    plot_tag,
    compare_tags,
    tag_correlation,
    block_1a_daily_summary,
    bess_soc_profile,
]

# ======================================================================
# LLM SETUP
# ======================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

# Build a concise, knowledge-rich system prompt from the YAML.
_meter_inventory_lines = []
for short, m in METERS.items():
    _meter_inventory_lines.append(f"  • {short} — {m['role']}")
_METER_INVENTORY_STR = "\n".join(_meter_inventory_lines)

SYSTEM_PROMPT = f"""You are the **Block 1A Solar+BESS Analytics Assistant** for the
Jamnagar pilot plant operated by Reliance Industries Limited.

# SCOPE — STRICTLY BLOCK 1A
You ONLY have data and knowledge for **Block 1A** (Monofacial Fixed-Tilt
Solar + BESS, AC-coupled architecture).
  - Solar PV: {BLOCK_1A_META.get('solar_dc_mw')} MW DC (1 inverter).
  - BESS: {BLOCK_1A_META.get('bess_total_mwh')} MWh total
    = {BLOCK_1A_META.get('bess_container_count')} containers
    × {BLOCK_1A_META.get('bess_container_mwh')} MWh each.

If the user asks about **Block 1B** or **Block 1C**, respond clearly:
"Block 1B and Block 1C are not yet loaded into my knowledge base. I can
only help with Block 1A analysis right now." Do not attempt to query them.

# METER INVENTORY (Block 1A)
{_METER_INVENTORY_STR}

(WMS — the Weather Monitoring Station — is physically shared across blocks
but here is treated as a Block 1A asset only.)

# DATA MODEL
- The MongoDB collection is multi-tenant. Every Block 1A query filters on `eid: "jamnagarsolar"`.
- The meter identifier may be stored under EITHER **`meterId`** OR **`did`** depending on
  the device — the agent probes once at startup per meter and caches the right field.
- **`time`** (top-level, UTC epoch ms) is the timestamp you filter on — it's when the
  gateway received the sample. `meter.tms` (nested) is the device-side Modbus timestamp;
  it's not what we query by.
- Tag values are nested under `meter.<tag>` — tools accept tag names WITHOUT the
  `meter.` prefix.
- IST dates ('YYYY-MM-DD') are accepted/returned by every tool; you never deal with epoch ms.
- If a query unexpectedly returns no data, call `inspect_sample_document` to see
  the actual stored fields for that meter.

# WORKFLOW
1. If unsure which meter has the data, call `list_meters` or `describe_meter`.
2. Before a query, call `get_meter_time_range` to confirm data exists in the window.
3. For single values or distributions: `query_tag` (start with agg_type='stats').
4. For visual analysis: `plot_tag` (one tag) or `compare_tags` (multiple).
5. For relationships: `tag_correlation` (also pair with `compare_tags` to inspect visually).
6. For a daily energy P&L picture: `block_1a_daily_summary`.
7. For BESS state-of-charge analysis: `bess_soc_profile`.

# DOMAIN POINTERS
- **Solar AC power** — `B1INV1.outputPower` (instantaneous kW)
  or `B1MFM2.activePower` (at the inverter AC output).
- **Solar daily energy** — `B1INV1.energyToday` (resets at midnight)
  or `B1MFM2.dailyActiveEnergyExport`.
- **POC export to grid** — `B1PQM2.dailyActiveEnergyExport` (daily)
  or `B1PQM2.totalActiveEnergyExport` (lifetime). This is the meter for PPA / DSM.
- **BESS state-of-charge** — `bessSOC` from B1BCT1 / B1BCT2 / B1BCT3 (0–100%).
- **BESS dispatch (AC side)** — `B1PCS1.activePower`. Sign convention
  per the PCS docs: positive = discharge, negative = charge — but
  verify against `dailyAcEnergyImport` (charging) vs `dailyAcEnergyExport`
  (discharging) for the day in question.
- **Irradiance** — `WMS.globalHorizontalIrradiance` (GHI, W/m²) or
  `WMS.planeOfArraySensor01` (POA, W/m²).
- **Module temperature** — `WMS.moduleTemperatureSensor01`. PV power
  derates with module temperature (~ −0.35 %/°C above 25 °C, typical).
- **Auxiliary consumption** — `B1MFM1.dailyActiveEnergyImport` is the
  parasitic load that should be deducted from gross to get net export.

# OUTPUT STYLE
- Always include units (kW, kWh, MWh, °C, W/m², %, kVAR, kVA).
- Explain results in solar/BESS engineering terms rather than raw DB vocabulary.
- If a query returns no data, say so plainly. Never invent numbers.
- After answering, suggest 1–2 sensible follow-ups
  (e.g. "Want me to plot this against GHI for the same day?").
"""


# ======================================================================
# LANGGRAPH WIRING
# ======================================================================
def call_model(state: MessagesState):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")


# ======================================================================
# MAIN CLI LOOP
# ======================================================================
def main():
    # Sanity-check Mongo connectivity up front so the user sees errors clearly.
    try:
        mongo_client.admin.command("ping")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB at {MONGO_URI[:60]}...")
        print(f"   {type(e).__name__}: {e}")
        print("   Check your VPN / firewall / config.yaml MONGO_URI and try again.")
        sys.exit(1)

    # Probe each Block 1A meter once to discover whether the identifier field
    # is 'meterId' or 'did', and to verify data presence.
    probe_all_meters(verbose=True)
    available = sum(1 for m in METERS.values() if m.get("_id_field"))
    if available == 0:
        print("\n❌ No data found for ANY Block 1A meter. "
              "Check eid value and database/collection in config.yaml.\n")
        sys.exit(1)

    with SqliteSaver.from_conn_string("solar_agent_memory.db") as memory:
        graph = builder.compile(checkpointer=memory)
        print("\n💡 Block 1A Solar+BESS Analytics Agent (LangGraph + Gemini 2.5 Flash)")
        print(f"   Plant: {PLANT['name']} | Meters loaded: {len(METERS)}")
        print("   Type 'exit' to quit.\n")

        thread_id = "block_1a_session"

        while True:
            try:
                q = input("💬 You: ")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye.")
                break
            if not q.strip():
                continue
            if q.strip().lower() in {"exit", "quit", ":q"}:
                print("👋 Bye.")
                break

            try:
                out = graph.invoke(
                    {"messages": [HumanMessage(content=q)]},
                    config={"configurable": {"thread_id": thread_id},
                            "recursion_limit": 20},
                )
                print("🤖 Assistant:", out["messages"][-1].content, "\n")
            except Exception as e:
                print(f"❌ Error during agent run: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
