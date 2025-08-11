# app.py — US Equities Copilot (v9)
# - Fractional years support (0.5 step)
# - TTM charts limited by selected history horizon
# - Label density toggle for discrete TTM charts
# - Single wide statement per section with Annual/Quarterly toggle
# - Canonical ordering for IS/BS rows
# - Peers fallback + Risk/Options analytics

import os, math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests
import streamlit as st

# ---------- Page config & theme ----------
st.set_page_config(page_title="US Equities Copilot", page_icon="📈", layout="wide")
try:
    pio.templates.default = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
except Exception:
    pass

# ---------- Keys & constants ----------
FMP_KEY = os.getenv("FMP_KEY")
if not FMP_KEY:
    try:
        FMP_KEY = st.secrets["FMP_KEY"]
    except Exception:
        FMP_KEY = None

FMP_BASE = "https://financialmodelingprep.com/api/v3"
if not FMP_KEY:
    st.info("Tip: add FMP_KEY (env/Streamlit secrets) to enrich peer list; otherwise we'll use Yahoo/S&P500 fallbacks.")

SEC_HEADERS = {
    "User-Agent": "US-Equities-Copilot/1.0 (contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}


# ---------- Optional FMP helpers (extended quarterly history) ----------
def fmp_get_json(path: str, params: dict | None = None):
    if not FMP_KEY:
        return None
    try:
        p = params.copy() if params else {}
        p["apikey"] = FMP_KEY
        url = f"{FMP_BASE}/{path}"
        r = requests.get(url, params=p, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fmp_quarterly_statement(symbol: str, kind: str) -> pd.DataFrame:
    """
    kind: 'income-statement' | 'cash-flow-statement' | 'balance-sheet-statement'
    Returns DataFrame indexed by date (oldest -> newest) if available.
    """
    data = fmp_get_json(f"{kind}/{symbol}", {"period": "quarter", "limit": 120})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" not in df:  # try calendarYear/period fallback
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date")
    return df


# ---------- Peers helpers (sturdier) ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fmp_screen_symbols(sector: str | None, industry: str | None, limit: int = 150) -> list[str]:
    if not FMP_KEY:
        return []
    try:
        params = {"limit": limit}
        if sector: params["sector"] = sector
        if industry: params["industry"] = industry
        out = fmp_get_json("stock-screener", params) or []
        syms = [o.get("symbol") for o in out if o.get("symbol")]
        # if still tiny, broaden by market cap
        if len(syms) < 10:
            out2 = fmp_get_json("stock-screener", {"marketCapMoreThan": 5_000_000_000, "limit": limit}) or []
            syms += [o.get("symbol") for o in out2 if o.get("symbol")]
        return list(dict.fromkeys(syms))
    except Exception:
        return []


# Fallback peer seeds for common tickers (used if sector is missing and no FMP key)
DEFAULT_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "AMZN", "NVDA", "META", "IBM", "HPQ", "DELL"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "NVDA", "CRM", "ADBE", "ORCL", "NOW"],
    "NVDA": ["AMD", "INTC", "AVGO", "QCOM", "TSM", "MU", "ASML", "TXN"],
    "AMZN": ["WMT", "COST", "TGT", "BABA", "SHOP", "MELI", "JD", "EBAY"],
    "GOOGL": ["MSFT", "META", "AAPL", "AMZN", "NFLX", "SNOW", "CRM", "ADBE"],
}


# ---------- Utils ----------
def extract_series(stmt_df: pd.DataFrame, aliases) -> pd.Series:
    """Extract a single line-item series from a statement DF (annual or quarterly) regardless of orientation."""
    if stmt_df is None or stmt_df.empty:
        return pd.Series(dtype=float)
    cols_dt = pd.to_datetime(stmt_df.columns, errors="coerce")
    idx_dt = pd.to_datetime(stmt_df.index, errors="coerce")
    # columns-as-dates case (yfinance typical)
    if cols_dt.notna().sum() >= idx_dt.notna().sum():
        name = get_alias(stmt_df.index, aliases)
        if not name:
            return pd.Series(dtype=float)
        s = pd.to_numeric(stmt_df.loc[name], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index, errors="coerce")
        return s.sort_index()
    # rows-as-dates (rare)
    else:
        name = get_alias(stmt_df.columns, aliases)
        if not name:
            return pd.Series(dtype=float)
        s = pd.to_numeric(stmt_df[name], errors="coerce").dropna()
        s.index = pd.to_datetime(stmt_df.index, errors="coerce")
        return s.sort_index()


# Sector fallbacks to guarantee peers
SECTOR_FALLBACKS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "INTC", "AMD",
                   "QCOM", "NOW", "TXN", "SAP", "IBM", "SNOW", "INTU", "PANW"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "T", "SPOT", "CRWD", "TTD"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG", "TJX", "SBUX", "RIVN"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BLK", "C", "AXP"],
    "Healthcare": ["UNH", "JNJ", "PFE", "LLY", "MRK", "ABBV", "TMO", "ABT", "BMY", "MDT"],
    "Industrials": ["BA", "CAT", "DE", "GE", "HON", "RTX", "UPS", "ETN", "NSC", "MMM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "PXD", "VLO", "KMI"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "CTVA", "MLM", "VMC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "O", "SPG", "CCI", "PSA", "DLR", "VTR", "WELL"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PCG"],
}


def _to_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def fmt_billions(x):
    v = _to_float(x)
    if np.isnan(v): return "—"
    if abs(v) >= 1_000_000_000: return f"{v / 1_000_000_000:,.1f}B"
    if abs(v) >= 1_000_000: return f"{v / 1_000_000:,.0f}M"
    return f"{v:,.0f}"


def parse_tickers(s: str):
    if not s: return []
    s = s.replace(",", " ").upper()
    return [t for t in s.split() if t.isalnum()]


def months_from_years(years_float: float) -> int:
    """Convert a float 'years' to nearest whole months (>=1)."""
    m = int(round(float(years_float) * 12))
    return max(m, 1)


def filter_last_years(obj, years: float):
    if obj is None or len(obj) == 0: return obj
    idx = pd.to_datetime(obj.index)
    cutoff = idx.max() - pd.DateOffset(months=months_from_years(years))
    return obj.loc[idx >= cutoff]


def drawdown_series(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return s
    roll_max = s.cummax()
    dd = (s / roll_max) - 1.0
    dd.name = "drawdown"
    return dd


def ttm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").sort_index()
    return s.rolling(4).sum()


def get_alias(row_index: pd.Index, candidates):
    low_map = {str(x).lower(): x for x in row_index}
    for c in candidates:
        if c.lower() in low_map: return low_map[c.lower()]
    for real in row_index:
        rl = str(real).lower()
        for c in candidates:
            if c.lower() in rl:
                return real
    return None


def to_period_str(dt_index: pd.DatetimeIndex, freq: str):
    if freq == "Quarterly":
        q = ((dt_index.month - 1) // 3) + 1
        return [f"{y}-Q{qq}" for y, qq in zip(dt_index.year, q)]
    else:
        return [str(y) for y in dt_index.year]


def limit_series_years(s: pd.Series, years: float) -> pd.Series:
    """Keep only the last `years` of data based on the series DatetimeIndex."""
    if s is None or len(s) == 0:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()
    if s.empty:
        return s
    cutoff = s.index.max() - pd.DateOffset(months=months_from_years(years))
    return s[s.index >= cutoff]


def bar_discrete_from_series(series: pd.Series, title: str, ylabel: str, freq_hint: str = "Quarterly",
                             label_every: int = 1):
    """Build a discrete bar chart from a Series with datetime-like index."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return px.bar(pd.DataFrame({"period": [], "value": []}), x="period", y="value", title=title)
    if not isinstance(s.index, pd.DatetimeIndex):
        idx = pd.to_datetime(s.index, errors="coerce")
        s.index = idx
    s = s.sort_index()
    periods = to_period_str(s.index, "Quarterly" if freq_hint == "Quarterly" else "Annual")
    df = pd.DataFrame({"period": periods, "value": s.values / 1_000_000_000.0})
    fig = px.bar(df, x="period", y="value", title=title, text="value")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(title=ylabel, tickformat=".1f")
    fig.update_xaxes(type="category")
    # label density
    if label_every > 1 and len(periods) > 2:
        fig.update_xaxes(tickmode="array", tickvals=periods[::label_every], ticktext=periods[::label_every])
    return fig


def _as_series(x, prefer_col="close"):
    if isinstance(x, pd.Series): return x
    if isinstance(x, pd.DataFrame):
        if prefer_col in x.columns:
            s = x[prefer_col]
        elif x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            num = x.select_dtypes(include=["number"])
            s = num.iloc[:, 0] if num.shape[1] else x.iloc[:, 0]
        return pd.Series(pd.to_numeric(s.squeeze().values, errors="coerce"), index=s.index)
    return pd.Series(np.asarray(x).reshape(-1))


# ---------- Canonical ordering for financial statements ----------
IS_ORDER = [
    ["total revenue", "revenue", "operating revenue"],
    ["cost of revenue", "cost of goods"],
    ["gross profit"],
    ["research and development", "r&d"],
    ["selling general and administrative", "sg&a"],
    ["operating expense", "total operating expenses"],
    ["operating income", "operating income or loss"],
    ["interest expense"],
    ["interest income"],
    ["other income", "non operating", "non-operating"],
    ["income before tax", "pretax income", "income before taxes", "ebt"],
    ["income tax", "provision for income taxes"],
    ["net income", "net income common stockholders"],
]

BS_ORDER = [
    # Assets (current -> non-current)
    ["cash and cash equivalents", "cash equivalents", "cash"],
    ["short term investments"],
    ["accounts receivable", "receivables"],
    ["inventory"],
    ["other current assets"],
    ["total current assets"],
    ["property plant equipment", "pp&e", "net property"],
    ["goodwill"],
    ["intangible", "intangible assets"],
    ["other assets"],
    ["total assets"],
    # Liabilities & Equity
    ["accounts payable"],
    ["deferred revenue"],
    ["short long term debt", "short term debt", "current debt"],
    ["other current liabilities"],
    ["total current liabilities"],
    ["long term debt"],
    ["other liabilities"],
    ["total liabilities"],
    ["common stock equity", "shareholders equity", "stockholders equity"],
    ["retained earnings"],
    ["treasury stock"],
    ["accumulated other comprehensive income", "aoci"],
    ["total equity"],
    ["total liabilities and stockholders equity", "total liabilities & stockholders equity"],
]


def sort_lines(df: pd.DataFrame, spec: list[list[str]]) -> pd.DataFrame:
    """Sort index (line items) by a list of keyword groups. Unmatched lines stay at the end, alphabetical."""
    if df is None or df.empty: return df
    idx = [str(x) for x in df.index]
    low = [x.lower() for x in idx]

    ranks = []
    for name in low:
        rank = 10_000  # default large
        for i, group in enumerate(spec):
            if any(kw in name for kw in group):
                rank = i
                break
        ranks.append(rank)

    order = np.lexsort((np.array(idx), np.array(ranks)))  # tie-break alphabetical
    return df.iloc[order]


# ---------- Data loaders ----------
@st.cache_data(show_spinner=False, ttl=3600)
def yf_history(ticker: str, start: str, end: str):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, auto_adjust=False)
    if df is None or df.empty: return pd.DataFrame()
    df = df.reset_index().rename(
        columns={"Date": "date", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    df["ticker"] = ticker
    return df[["date", "ticker", "close", "adj_close", "volume"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices_yf(tickers, start, end):
    frames = []
    for t in tickers:
        try:
            frames.append(yf_history(t, start, end))
        except Exception:
            pass
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "ticker", "close", "adj_close", "volume"])


@st.cache_data(show_spinner=False, ttl=1800)
def yf_info(ticker: str):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    try:
        return tk.get_info() or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def yf_statements(ticker: str):
    """Robust statements with orientation check."""
    import yfinance as yf
    tk = yf.Ticker(ticker)

    def grab(attr_names):
        for a in attr_names:
            try:
                df = getattr(tk, a)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df.copy()
            except Exception:
                continue
        return pd.DataFrame()

    is_a = grab(["income_stmt", "financials"])
    is_q = grab(["quarterly_income_stmt", "quarterly_financials"])
    bs_a = grab(["balance_sheet"])
    bs_q = grab(["quarterly_balance_sheet"])
    cf_a = grab(["cash_flow_stmt", "cashflow"])
    cf_q = grab(["quarterly_cashflow"])

    def normalize(df):
        if df is None or df.empty: return df
        cols_dt = pd.to_datetime(df.columns, errors="coerce")
        idx_dt = pd.to_datetime(df.index, errors="coerce")
        if cols_dt.notna().sum() < idx_dt.notna().sum():
            df = df.T
        return df

    return {
        "is_annual": normalize(is_a),
        "is_quarterly": normalize(is_q),
        "bs_annual": normalize(bs_a),
        "bs_quarterly": normalize(bs_q),
        "cf_annual": normalize(cf_a),
        "cf_quarterly": normalize(cf_q),
    }


# ---------- SEC helpers (robust) ----------
@st.cache_data(show_spinner=False, ttl=86400)
def sec_ticker_map():
    urls = [
        "https://www.sec.gov/files/company_tickers.json",
        "https://www.sec.gov/files/company_tickers_exchange.json",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            if isinstance(j, dict):
                df = pd.DataFrame(j).T
            else:
                df = pd.DataFrame(j)
            # Normalize columns
            cols = {c.lower(): c for c in df.columns}
            if "ticker" not in cols:
                for c in df.columns:
                    if str(c).lower() in ("tickers", "symbol"): cols["ticker"] = c
                    if str(c).lower() in ("cik", "cik_str", "cikstr"): cols["cik_str"] = c
                    if str(c).lower() in ("title", "name", "company"): cols["title"] = c
            if "ticker" in cols:
                out = pd.DataFrame({
                    "ticker": df[cols["ticker"]].astype(str).str.upper(),
                    "cik_str": df[cols.get("cik_str", cols.get("cik", df.columns[0]))].astype(str).str.zfill(10),
                    "title": df[cols.get("title", df.columns[-1])].astype(str),
                })
                return out.dropna(subset=["ticker", "cik_str"])
        except Exception:
            continue
    return pd.DataFrame(columns=["ticker", "cik_str", "title"])


def sec_lookup_cik(ticker: str):
    df = sec_ticker_map()
    if df.empty or "ticker" not in df.columns: return None
    row = df[df["ticker"] == str(ticker).upper()]
    return row["cik_str"].iloc[0] if not row.empty else None


@st.cache_data(show_spinner=False, ttl=3600)
def sec_recent_filings(ticker: str, limit: int = 5):
    cik = sec_lookup_cik(ticker)
    if not cik: return pd.DataFrame()
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(sub_url, headers=SEC_HEADERS, timeout=30)
        if r.status_code != 200: return pd.DataFrame()
        j = r.json()
        forms = pd.DataFrame(j.get("filings", {}).get("recent", {}))
        if forms.empty: return pd.DataFrame()
        take = forms.head(limit).copy()

        def _link(row):
            acc = str(row["accessionNumber"]).replace("-", "")
            primary = row["primaryDocument"]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary}"

        take["link"] = take.apply(_link, axis=1)
        return take[["filingDate", "form", "link"]].rename(
            columns={"filingDate": "Date", "form": "Form", "link": "Link"})
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def sec_latest_filing_link(ticker: str, form_type: str = "10-Q"):
    cik = sec_lookup_cik(ticker)
    if not cik: return None
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(sub_url, headers=SEC_HEADERS, timeout=30)
        if r.status_code != 200: return None
        j = r.json()
        forms = pd.DataFrame(j.get("filings", {}).get("recent", {}))
        if forms.empty: return None
        target = forms[forms["form"] == form_type]
        if target.empty: return None
        row = target.iloc[0]
        acc = str(row["accessionNumber"]).replace("-", "")
        primary = row["primaryDocument"]
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary}"
    except Exception:
        return None


# ---------- Statements display ----------
def format_statements_for_display(df: pd.DataFrame, freq: str, years: float) -> pd.DataFrame:
    """Normalize orientation; filter by years; scale to billions; dates in columns."""
    if df is None or df.empty: return pd.DataFrame()
    cols_dt = pd.to_datetime(df.columns, errors="coerce")
    idx_dt = pd.to_datetime(df.index, errors="coerce")
    if cols_dt.notna().sum() < idx_dt.notna().sum():
        df = df.T
        cols_dt = pd.to_datetime(df.columns, errors="coerce")

    mask = ~cols_dt.isna()
    df = df.loc[:, mask];
    cols_dt = cols_dt[mask]
    if len(cols_dt) == 0: return pd.DataFrame()

    cutoff = cols_dt.max() - pd.DateOffset(months=months_from_years(min(years, 5) if freq == "Quarterly" else years))
    keep = cols_dt >= cutoff
    df = df.loc[:, keep];
    cols_dt = cols_dt[keep]
    if len(cols_dt) == 0: return pd.DataFrame()

    order = np.argsort(cols_dt.values)
    ordered = pd.to_datetime(cols_dt.values[order])
    df = df.iloc[:, order]

    df = df.apply(pd.to_numeric, errors="coerce") / 1_000_000_000.0  # to Billions

    if freq == "Quarterly":
        q = ((ordered.month - 1) // 3) + 1
        new_cols = [f"{y}-Q{qq}" for y, qq in zip(ordered.year, q)]
    else:
        new_cols = [str(y) for y in ordered.year]
    df.columns = new_cols
    return df


# ---------- Peers ----------
@st.cache_data(show_spinner=True, ttl=86400)
def sp500_universe():
    """Get S&P 500 tickers via yfinance helper; fallback to a small hardcoded list if unavailable."""
    try:
        import yfinance as yf
        tickers = yf.tickers_sp500()
        if isinstance(tickers, (list, tuple)) and len(tickers) > 10:
            return [t.upper() for t in tickers]
    except Exception:
        pass
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "JNJ", "V", "XOM", "UNH"]


@st.cache_data(show_spinner=True, ttl=3600)
def build_sector_metrics(ticker_for_context: str, sector: str | None, industry: str | None):
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor

    # Seed candidates from FMP screener and yfinance peers
    peers = fmp_screen_symbols(sector, industry, limit=250) if FMP_KEY else []
    try:
        yp = yf.Ticker(ticker_for_context).get_peers() or []
        peers += yp
    except Exception:
        pass
    if sector in SECTOR_FALLBACKS:
        peers += SECTOR_FALLBACKS[sector]

    # Unique, keep order, remove self
    peers = [s for s in dict.fromkeys(peers) if s and s != ticker_for_context]

    # Always include the context ticker at the front
    tickers = [ticker_for_context] + peers[:180]

    def fetch_metrics(t):
        try:
            info = yf.Ticker(t).get_info() or {}
            sec = info.get("sector")
            ind = info.get("industry")
            ev = _to_float(info.get("enterpriseValue"))
            ebitda = _to_float(info.get("ebitda"))
            sales = _to_float(info.get("totalRevenue"))
            pe = _to_float(info.get("trailingPE"))
            pm = _to_float(info.get("profitMargins"))
            ev_ebitda = ev / ebitda if (np.isfinite(ev) and np.isfinite(ebitda) and ebitda) else np.nan
            ev_sales = ev / sales if (np.isfinite(ev) and np.isfinite(sales) and sales) else np.nan
            return {"ticker": t, "sector": sec, "industry": ind,
                    "ev": ev, "ebitda": ebitda, "sales": sales,
                    "ev_ebitda": ev_ebitda, "ev_sales": ev_sales, "pe": pe, "pm": pm}
        except Exception:
            return {"ticker": t, "sector": None, "industry": None,
                    "ev": np.nan, "ebitda": np.nan, "sales": np.nan,
                    "ev_ebitda": np.nan, "ev_sales": np.nan, "pe": np.nan, "pm": np.nan}

    with ThreadPoolExecutor(max_workers=24) as ex:
        rows = list(ex.map(fetch_metrics, tickers))

    df = pd.DataFrame(rows)

    # Try to filter peers by exact industry first; if too few, fall back to sector
    main_sec = sector
    main_ind = industry
    # Ensure the context row keeps its industry/sector values if missing from inputs
    if df.loc[df["ticker"] == ticker_for_context, "industry"].isna().all() and main_ind:
        df.loc[df["ticker"] == ticker_for_context, "industry"] = main_ind
    if df.loc[df["ticker"] == ticker_for_context, "sector"].isna().all() and main_sec:
        df.loc[df["ticker"] == ticker_for_context, "sector"] = main_sec

    peers_filtered = df.copy()
    if main_ind:
        peers_filtered = peers_filtered[peers_filtered["industry"] == main_ind]
    elif main_sec:
        peers_filtered = peers_filtered[peers_filtered["sector"] == main_sec]

    # If still too few (e.g., FMP key missing), relax stepwise
    if peers_filtered["ticker"].nunique() < 5 and main_sec:
        peers_filtered = df[df["sector"] == main_sec]
    if peers_filtered["ticker"].nunique() < 3:
        # last resort: add S&P 500 top caps but mark that they are broad-market
        universe = sp500_universe()
        more = [s for s in universe if s not in df["ticker"].tolist()][:30]
        if more:
            with ThreadPoolExecutor(max_workers=16) as ex:
                rows2 = list(ex.map(fetch_metrics, more))
            df2 = pd.DataFrame(rows2)
            peers_filtered = pd.concat([peers_filtered, df2], ignore_index=True)

    # Compute medians on the filtered set
    peers_filtered = peers_filtered.dropna(how="all", subset=["ev_ebitda", "ev_sales", "pe"]).reset_index(drop=True)

    meds = {}
    for col in ["ev_ebitda", "ev_sales", "pe", "pm"]:
        col_vals = pd.to_numeric(peers_filtered[col], errors="coerce")
        meds[col] = float(np.nanmedian(col_vals)) if col_vals.notna().any() else np.nan

    # Keep only useful display columns
    display_cols = ["ticker", "sector", "industry", "ev_ebitda", "ev_sales", "pe", "pm"]
    peers_filtered = peers_filtered[display_cols].copy()

    return peers_filtered, meds


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    try:
        qp = dict(st.query_params)
    except Exception:
        try:
            qp = st.experimental_get_query_params()
        except Exception:
            qp = {}


    def _qp_get(name, default):
        v = qp.get(name, default)
        if isinstance(v, list): return v[0] if v else default
        return v


    default_tickers = _qp_get("tickers", "AAPL")
    try:
        default_years = float(_qp_get("years", "5"))
    except Exception:
        default_years = 5.0

    tickers_input = st.text_input("Tickers (space/comma separated)", value=default_tickers)
    years_hist = st.slider("History (years)", 1.0, 10.0, float(default_years), step=0.5)

    try:
        st.query_params.update({"tickers": " ".join(parse_tickers(tickers_input)), "years": f"{years_hist}"})
    except Exception:
        try:
            st.experimental_set_query_params(tickers=" ".join(parse_tickers(tickers_input)), years=f"{years_hist}")
        except Exception:
            pass

# ---------- Main ----------
st.title("US Equities Copilot")

tickers = parse_tickers(tickers_input)
if not tickers: st.stop()

end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.DateOffset(months=months_from_years(years_hist))

for t in tickers:
    st.markdown(f"## {t}")
    info = yf_info(t)
    sector = info.get("sector")
    industry = info.get("industry")
    mc = info.get("marketCap");
    pe_val = info.get("trailingPE");
    pm_val = info.get("profitMargins")

    cols = st.columns(4)
    cols[0].metric("Market Cap", fmt_billions(mc))
    cols[1].metric("P/E (trailing)", f"{pe_val:.1f}" if np.isfinite(_to_float(pe_val)) else "—")
    cols[2].metric("Profit Margin", f"{pm_val:.1%}" if np.isfinite(_to_float(pm_val)) else "—")
    cols[3].metric("Sector", sector or "—")

    tabs = st.tabs(["Overview", "Price", "Fundamentals", "Graphs", "Valuation", "Risk", "SEC"])

    # Overview
    with tabs[0]:
        ov = {"Name": info.get("longName") or info.get("shortName"),
              "Country": info.get("country"),
              "Industry": info.get("industry"),
              "Website": info.get("website")}
        st.table(pd.Series({k: v for k, v in ov.items() if v}).rename("Value"))

    # Price
    with tabs[1]:
        all_prices = load_prices_yf([t], str(start_date.date()), str(end_date.date()))
        p = all_prices[all_prices["ticker"] == t].sort_values("date").set_index("date")["close"]
        p = filter_last_years(p, years_hist)
        if p.empty:
            st.warning("No price data.")
        else:
            bench = st.selectbox("Benchmark", ["SPY", "QQQ", "None"], index=0, key=f"bench_{t}")
            if bench != "None":
                bdf = load_prices_yf([bench], str(start_date.date()), str(end_date.date()))
                b = bdf[bdf["ticker"] == bench].sort_values("date").set_index("date")["close"].reindex(p.index).ffill()
                p_s = _as_series(p, "close");
                b_s = _as_series(b, "close")
                if p_s.empty or b_s.empty:
                    st.warning("Not enough data to plot benchmark overlay.")
                else:
                    df_p = p_s.reset_index();
                    df_p.columns = ["date", "price"];
                    df_p["series"] = t
                    df_b = b_s.reset_index();
                    df_b.columns = ["date", "price"];
                    df_b["series"] = bench
                    long = pd.concat([df_p, df_b], ignore_index=True)
                    st.plotly_chart(
                        px.line(long, x="date", y="price", color="series", title=f"{t} vs {bench} ({years_hist:g}Y)"),
                        use_container_width=True)

                    aligned = pd.concat([p_s.astype(float), b_s.astype(float)], axis=1, join="inner").dropna()
                    if not aligned.empty:
                        ratio = aligned.iloc[:, 0] / aligned.iloc[:, 1]
                        rs_long = ratio.reset_index();
                        rs_long.columns = ["date", "value"]
                        st.plotly_chart(px.line(rs_long, x="date", y="value", title=f"Relative Strength: {t}/{bench}"),
                                        use_container_width=True)
            else:
                st.plotly_chart(px.line(p, title=f"{t} — Price ({years_hist:g}Y)"), use_container_width=True)

            dd = drawdown_series(p)
            fig2 = px.area(dd.reset_index(), x="date", y="drawdown", title=f"{t} — Drawdown");
            fig2.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)

    # Fundamentals
    with tabs[2]:
        stmts = yf_statements(t)

        # ---------- Income Statement (single, wide) ----------
        st.subheader("Income Statement")
        is_a = format_statements_for_display(stmts["is_annual"], "Annual", years_hist)
        is_q = format_statements_for_display(stmts["is_quarterly"], "Quarterly", years_hist)
        if is_a is not None and not is_a.empty: is_a = sort_lines(is_a, IS_ORDER)
        if is_q is not None and not is_q.empty: is_q = sort_lines(is_q, IS_ORDER)
        is_freq = st.radio("Frequency", ["Annual", "Quarterly"], index=0, key=f"is_freq_{t}", horizontal=True)
        is_df = is_a if is_freq == "Annual" else is_q
        if is_df is None or is_df.empty:
            alt = is_q if is_freq == "Annual" else is_a
            if alt is not None and not alt.empty:
                st.info(f"No {is_freq.lower()} statement available; showing the other frequency.")
                is_df = alt
        st.caption(f"{is_freq} (Billions)")
        st.dataframe(is_df if is_df is not None and not is_df.empty else pd.DataFrame({'': ['empty']}),
                     use_container_width=True, height=460)

        # ---------- Balance Sheet (single, wide) ----------
        st.subheader("Balance Sheet")
        bs_a = format_statements_for_display(stmts["bs_annual"], "Annual", years_hist)
        bs_q = format_statements_for_display(stmts["bs_quarterly"], "Quarterly", years_hist)
        if bs_a is not None and not bs_a.empty: bs_a = sort_lines(bs_a, BS_ORDER)
        if bs_q is not None and not bs_q.empty: bs_q = sort_lines(bs_q, BS_ORDER)
        bs_freq = st.radio("Frequency  ", ["Annual", "Quarterly"], index=0, key=f"bs_freq_{t}", horizontal=True)
        bs_df = bs_a if bs_freq == "Annual" else bs_q
        if bs_df is None or bs_df.empty:
            alt = bs_q if bs_freq == "Annual" else bs_a
            if alt is not None and not alt.empty:
                st.info(f"No {bs_freq.lower()} balance sheet; showing the other frequency.")
                bs_df = alt
        st.caption(f"{bs_freq} (Billions)")
        st.dataframe(bs_df if bs_df is not None and not bs_df.empty else pd.DataFrame({'': ['empty']}),
                     use_container_width=True, height=460)

        # ---------- Cash Flow (single, wide) ----------
        st.subheader("Cash Flow")
        cf_a = format_statements_for_display(stmts["cf_annual"], "Annual", years_hist)
        cf_q = format_statements_for_display(stmts["cf_quarterly"], "Quarterly", years_hist)
        cf_freq = st.radio("Frequency   ", ["Annual", "Quarterly"], index=0, key=f"cf_freq_{t}", horizontal=True)
        cf_df = cf_a if cf_freq == "Annual" else cf_q
        if cf_df is None or cf_df.empty:
            alt = cf_q if cf_freq == "Annual" else cf_a
            if alt is not None and not alt.empty:
                st.info(f"No {cf_freq.lower()} cash flow; showing the other frequency.")
                cf_df = alt
        st.caption(f"{cf_freq} (Billions)")
        st.dataframe(cf_df if cf_df is not None and not cf_df.empty else pd.DataFrame({'': ['empty']}),
                     use_container_width=True, height=460)

        st.markdown("---")

        # --- Chart menu (below financial statements) ---
        chart_options = [
            "Revenue (TTM)", "Net Income (TTM)", "Operating Income (TTM)",
            "FCF (TTM)",
            "Revenue (Annual)", "Net Income (Annual)", "FCF (Annual)",
            "P/E (TTM)"
        ]
        selected_charts = st.multiselect(
            "Choose charts to display",
            chart_options,
            default=[],
            key=f"fund_chart_sel_v2_{t}"
        )
        chart_queue = []


        def plot_sel(fig, label: str):
            chart_queue.append((label, fig))


        def _map_chart_key_from_title(title: str) -> str:
            if not isinstance(title, str):
                return ""
            t = title.lower()
            # Robust contains-based mapping
            if "revenue" in t and "ttm" in t:
                return "Revenue (TTM)"
            if "net income" in t and "ttm" in t:
                return "Net Income (TTM)"
            if "operating" in t and "income" in t and "ttm" in t:
                return "Operating Income (TTM)"
            if "fcf" in t and "ttm" in t:
                return "FCF (TTM)"
            if "revenue" in t and "annual" in t:
                return "Revenue (Annual)"
            if "net income" in t and "annual" in t:
                return "Net Income (Annual)"
            if "fcf" in t and "annual" in t:
                return "FCF (Annual)"
            if "p/e" in t and "ttm" in t:
                return "P/E (TTM)"
            return title


        def plot_maybe(fig, *args, **kwargs):
            try:
                title = getattr(fig.layout.title, "text", "") or ""
            except Exception:
                title = ""
            label = _map_chart_key_from_title(title)
            return plot_sel(fig, label)


        # Label density control removed; default to every label
        every = 1

        # Try extended quarterly from FMP first (if key available), else yfinance raw quarterlies
        if FMP_KEY:
            fmp_is_q = fmp_quarterly_statement(t, "income-statement")
            fmp_cf_q = fmp_quarterly_statement(t, "cash-flow-statement")
        else:
            fmp_is_q = pd.DataFrame()
            fmp_cf_q = pd.DataFrame()

    with tabs[4]:
        peers_df, meds = build_sector_metrics(t, sector, industry)
        ev = _to_float(info.get("enterpriseValue"))
        ebitda = _to_float(info.get("ebitda"))
        sales = _to_float(info.get("totalRevenue"))
        pe_v = _to_float(info.get("trailingPE"))
        pm_v = _to_float(info.get("profitMargins"))
        ev_eb = ev / ebitda if (np.isfinite(ev) and np.isfinite(ebitda) and ebitda) else np.nan
        ev_s = ev / sales if (np.isfinite(ev) and np.isfinite(sales) and sales) else np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("EV/EBITDA", f"{ev_eb:.1f}" if np.isfinite(ev_eb) else "—")
        m2.metric("EV/Sales", f"{ev_s:.1f}" if np.isfinite(ev_s) else "—")
        m3.metric("P/E", f"{pe_v:.1f}" if np.isfinite(pe_v) else "—")
        m4.metric("Profit Margin", f"{pm_v:.1%}" if np.isfinite(pm_v) else "—")

        if not peers_df.empty and len(peers_df) >= 2:
            def pct_of(series, val):
                s = pd.to_numeric(series, errors="coerce").dropna()
                return float((s <= val).mean()) if np.isfinite(val) and len(s) else np.nan


            cpc = st.columns(4)
            cpc[0].metric("EV/EBITDA Percentile",
                          f"{pct_of(peers_df['ev_ebitda'], ev_eb):.0%}" if np.isfinite(ev_eb) else "—")
            cpc[1].metric("EV/Sales Percentile",
                          f"{pct_of(peers_df['ev_sales'], ev_s):.0%}" if np.isfinite(ev_s) else "—")
            cpc[2].metric("P/E Percentile", f"{pct_of(peers_df['pe'], pe_v):.0%}" if np.isfinite(pe_v) else "—")
            cpc[3].metric("Profit Margin Percentile",
                          f"{pct_of(peers_df['pm'], pm_v):.0%}" if np.isfinite(pm_v) else "—")
        else:
            st.caption("Peer percentiles require at least 2 companies; table still shown if available.")

        if not peers_df.empty:
            st.dataframe(peers_df.sort_values(["industry", "sector", "ticker"]).reset_index(drop=True),
                         use_container_width=True, height=360)
            st.download_button("Download peers (CSV)", peers_df.to_csv(index=False).encode("utf-8"),
                               f"{t}_peers.csv", "text/csv")
        else:
            st.info("No peers found. Add FMP_KEY or use a broader universe.")

    # Risk
    with tabs[5]:
        all_prices = load_prices_yf([t], str(start_date.date()), str(end_date.date()))
        p = all_prices[all_prices["ticker"] == t].sort_values("date").set_index("date")["close"]
        p = filter_last_years(p, years_hist)
        if p.empty:
            st.info("No prices for risk section.")
        else:
            rets = p.pct_change().dropna()
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() else np.nan
            sortino = (rets.mean() * np.sqrt(252)) / (rets[rets < 0].std() * np.sqrt(252)) if (
                rets[rets < 0].std()) else np.nan
            var95_1d = np.percentile(rets, 5)
            var99_1d = np.percentile(rets, 1)
            r5 = p.pct_change(5).dropna()
            var95_5d = np.percentile(r5, 5) if len(r5) else np.nan
            var99_5d = np.percentile(r5, 1) if len(r5) else np.nan
            cvar95 = rets[rets <= var95_1d].mean() if len(rets[rets <= var95_1d]) else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Vol (ann.)", f"{ann_vol:.1%}" if np.isfinite(ann_vol) else "—")
            c2.metric("Sharpe (daily->ann.)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "—")
            c3.metric("Sortino", f"{sortino:.2f}" if np.isfinite(sortino) else "—")
            c4.metric("Max Drawdown", f"{drawdown_series(p).min():.1%}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("1D VaR 95%", f"{var95_1d:.2%}" if np.isfinite(var95_1d) else "—")
            c6.metric("1D VaR 99%", f"{var99_1d:.2%}" if np.isfinite(var99_1d) else "—")
            c7.metric("5D VaR 95%", f"{var95_5d:.2%}" if np.isfinite(var95_5d) else "—")
            c8.metric("5D VaR 99%", f"{var99_5d:.2%}" if np.isfinite(var99_5d) else "—")
            st.caption(f"CVaR 95% (Expected Shortfall): {cvar95:.2%}" if np.isfinite(cvar95) else "CVaR 95%: —")

            st.plotly_chart(px.histogram(rets, nbins=60, title="Daily Returns Histogram"), use_container_width=True)

            rv = rets.rolling(30).std() * np.sqrt(252)
            rv60 = rets.rolling(60).std() * np.sqrt(252)
            rv_df = pd.DataFrame({"date": rv.index, "Vol30": rv.values, "Vol60": rv60.values}).dropna()
            rv_long = rv_df.melt(id_vars="date", var_name="window", value_name="vol")
            st.plotly_chart(
                px.line(rv_long, x="date", y="vol", color="window", title="Rolling Ann. Volatility (30/60d)"),
                use_container_width=True)

            # Options
            try:
                import yfinance as yf

                tk = yf.Ticker(t)
                exps = tk.options
                if exps:
                    exp = exps[0]
                    chain = tk.option_chain(exp)
                    calls, puts = chain.calls.copy(), chain.puts.copy()
                    last_px = float(p.iloc[-1])


                    def near_atm_iv(df):
                        if df is None or df.empty or "impliedVolatility" not in df: return np.nan
                        pos = (df["strike"] - last_px).abs().idxmin()
                        return float(pd.to_numeric(df.loc[pos, "impliedVolatility"], errors="coerce"))


                    atm_call_iv = near_atm_iv(calls);
                    atm_put_iv = near_atm_iv(puts)


                    def safe_mean(s):
                        s = pd.to_numeric(s, errors="coerce");
                        return float(s.mean()) if s.notna().any() else np.nan


                    pcr_oi = safe_mean(puts.get("openInterest", pd.Series())) / safe_mean(
                        calls.get("openInterest", pd.Series()))
                    pcr_vol = safe_mean(puts.get("volume", pd.Series())) / safe_mean(calls.get("volume", pd.Series()))
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ATM IV (call)", f"{atm_call_iv:.1%}" if np.isfinite(atm_call_iv) else "—",
                              help=f"Nearest expiry: {exp}")
                    m2.metric("ATM IV (put)", f"{atm_put_iv:.1%}" if np.isfinite(atm_put_iv) else "—")
                    m3.metric("Put/Call OI", f"{pcr_oi:.2f}" if np.isfinite(pcr_oi) else "—")
                    m4.metric("Put/Call Volume", f"{pcr_vol:.2f}" if np.isfinite(pcr_vol) else "—")
                    if "impliedVolatility" in calls and not calls["impliedVolatility"].isna().all():
                        calls_plot = calls[["strike", "impliedVolatility"]].dropna();
                        calls_plot["type"] = "Call"
                        puts_plot = puts[["strike", "impliedVolatility"]].dropna();
                        puts_plot["type"] = "Put"
                        smile = pd.concat([calls_plot, puts_plot], ignore_index=True)
                        st.plotly_chart(
                            px.line(smile, x="strike", y="impliedVolatility", color="type", title=f"IV Smile — {exp}"),
                            use_container_width=True)
                else:
                    st.info("No options expirations returned by Yahoo for this ticker.")
            except Exception as e:
                st.info(f"Options data unavailable: {e}")

    # SEC
    with tabs[6]:
        link_10q = sec_latest_filing_link(t, "10-Q")
        link_10k = sec_latest_filing_link(t, "10-K")
        c1, c2 = st.columns(2)
        with c1:
            if link_10q: st.markdown(f"**Latest 10-Q:** [{link_10q}]({link_10q})")
        with c2:
            if link_10k: st.markdown(f"**Latest 10-K:** [{link_10k}]({link_10k})")
        filings = sec_recent_filings(t, limit=10)
        if not filings.empty:
            st.dataframe(filings, use_container_width=True, height=360)
        else:
            st.info("No recent filings found via SEC API.")
    # Graphs
    with tabs[3]:
        st.subheader("Graphs")
        # Label density default
        every = 1
        # --- Revenue & Net Income TTM ---
        done_any = False
        if fmp_is_q is not None and not fmp_is_q.empty:
            if "revenue" in fmp_is_q:
                rev_series = pd.to_numeric(fmp_is_q["revenue"], errors="coerce").dropna()
                rev_ttm = limit_series_years(ttm(rev_series).dropna(), years_hist)
                if len(rev_ttm) >= 3:
                    st.plotly_chart(
                        bar_discrete_from_series(rev_ttm, "Revenue (TTM, Billions)", "Revenue (B)", "Quarterly",
                                                 label_every=every), use_container_width=True);
            if "netIncome" in fmp_is_q:
                ni_series = pd.to_numeric(fmp_is_q["netIncome"], errors="coerce").dropna()
                ni_ttm = limit_series_years(ttm(ni_series).dropna(), years_hist)
                if len(ni_ttm) >= 3:
                    st.plotly_chart(
                        bar_discrete_from_series(ni_ttm, "Net Income (TTM, Billions)", "Net Income (B)", "Quarterly",
                                                 label_every=every), use_container_width=True);
            isq_raw = stmts["is_quarterly"]
            if isq_raw is not None and not isq_raw.empty:
                rev_series = extract_series(isq_raw, ["Total Revenue", "Revenue"])
                ni_series = extract_series(isq_raw, ["Net Income", "Net Income Common Stockholders"])
                if not rev_series.empty:
                    rev_ttm = limit_series_years(ttm(rev_series).dropna(), years_hist)
                    if len(rev_ttm) >= 3:
                        st.plotly_chart(
                            bar_discrete_from_series(rev_ttm, "Revenue (TTM, Billions)", "Revenue (B)", "Quarterly",
                                                     label_every=every), use_container_width=True);
                if not ni_series.empty:
                    ni_ttm = limit_series_years(ttm(ni_series).dropna(), years_hist)
                    if len(ni_ttm) >= 3:
                        st.plotly_chart(bar_discrete_from_series(ni_ttm, "Net Income (TTM, Billions)", "Net Income (B)",
                                                                 "Quarterly",
                                                                 label_every=every), use_container_width=True);

        # --- FCF TTM ---
        fcf_done = False
        if fmp_cf_q is not None and not fmp_cf_q.empty and {"operatingCashFlow", "capitalExpenditure"} <= set(
                fmp_cf_q.columns):
            ocf = pd.to_numeric(fmp_cf_q["operatingCashFlow"], errors="coerce")
            capex = pd.to_numeric(fmp_cf_q["capitalExpenditure"], errors="coerce")
            fcf_series = (ocf - capex).dropna()
            fcf_ttm = limit_series_years(ttm(fcf_series).dropna(), years_hist)
            if len(fcf_ttm) >= 3:
                st.plotly_chart(
                    bar_discrete_from_series(fcf_ttm, "FCF (TTM, Billions)", "FCF (B)", "Quarterly", label_every=every),
                    use_container_width=True);
                fcf_done = True

        if not fcf_done:
            cfq_raw = stmts["cf_quarterly"]
            if cfq_raw is not None and not cfq_raw.empty:
                ocf = extract_series(cfq_raw, ["Total Cash From Operating Activities", "Operating Cash Flow"])
                capex = extract_series(cfq_raw, ["Capital Expenditures", "Capital Expenditure"])
                if not ocf.empty and not capex.empty:
                    fcf_series = (ocf - capex).dropna()
                    fcf_ttm = limit_series_years(ttm(fcf_series).dropna(), years_hist)
                    if len(fcf_ttm) >= 3:
                        st.plotly_chart(bar_discrete_from_series(fcf_ttm, "FCF (TTM, Billions)", "FCF (B)", "Quarterly",
                                                                 label_every=every), use_container_width=True);
                        fcf_done = True
        # P/E (TTM)
        try:
            # Price series
            try:
                p = all_prices[all_prices["ticker"] == t].sort_values("date").set_index("date")["close"]
            except Exception:
                ap = load_prices_yf([t], str(start_date.date()), str(end_date.date()))
                p = ap[ap["ticker"] == t].sort_values("date").set_index("date")["close"]
            p = filter_last_years(p, years_hist)

            # EPS TTM from quarterly NI and diluted shares
            eps_ttm = None
            # Prefer FMP fields when available
            if fmp_is_q is not None and not fmp_is_q.empty and "netIncome" in fmp_is_q:
                ni_q = pd.to_numeric(fmp_is_q["netIncome"], errors="coerce")
                shs = None
                for cand in ["weightedAverageShsOutDil", "weightedAverageShsOut"]:
                    if cand in fmp_is_q:
                        shs = pd.to_numeric(fmp_is_q[cand], errors="coerce").replace(0, pd.NA)
                        break
                if shs is not None:
                    eps_q = (ni_q / shs).dropna()
                    eps_ttm = limit_series_years(ttm(eps_q).dropna(), years_hist)

            # Yahoo statements fallback for EPS
            if eps_ttm is None or (hasattr(eps_ttm, "__len__") and len(eps_ttm) < 3):
                isq_raw = stmts["is_quarterly"]
                ni_q2 = extract_series(isq_raw, ["Net Income", "Net Income Common Stockholders"])
                shs_q2 = extract_series(isq_raw, [
                    "Diluted Shares Outstanding",
                    "Weighted Average Diluted Shares",
                    "Weighted Average Shares Diluted",
                    "Weighted Average Shares Outstanding (Diluted)"
                ])
                if not ni_q2.empty and not shs_q2.empty:
                    eps_q2 = (ni_q2 / shs_q2).dropna()
                    eps_ttm = limit_series_years(ttm(eps_q2).dropna(), years_hist)

            if eps_ttm is not None and hasattr(eps_ttm, "__len__") and len(eps_ttm) >= 3 and not p.empty:
                # Align price to EPS dates
                p_aligned = p.reindex(eps_ttm.index, method="ffill").dropna()
                common_idx = eps_ttm.index.intersection(p_aligned.index)
                if len(common_idx) >= 3:
                    pe_series = (p_aligned.reindex(common_idx) / eps_ttm.reindex(common_idx)).dropna()
                    df_pe = pe_series.reset_index()
                    df_pe.columns = ["date", "P/E"]
                    fig_pe = px.line(df_pe, x="date", y="P/E", title="P/E (TTM)")
                    st.plotly_chart(fig_pe, use_container_width=True)
        except Exception:
            pass
        # --- Annual charts ---
        is_a_raw = stmts["is_annual"]
        rev_a = extract_series(is_a_raw, ["Total Revenue", "Revenue"])
        ni_a = extract_series(is_a_raw, ["Net Income", "Net Income Common Stockholders"])
        if not rev_a.empty:
            rev_a = limit_series_years(rev_a, years_hist)
            st.plotly_chart(bar_discrete_from_series(rev_a, "Revenue (Annual, Billions)", "Revenue (B)", "Annual",
                                                     label_every=1), use_container_width=True)
        if not ni_a.empty:
            ni_a = limit_series_years(ni_a, years_hist)
            st.plotly_chart(bar_discrete_from_series(ni_a, "Net Income (Annual, Billions)", "Net Income (B)", "Annual",
                                                     label_every=1), use_container_width=True)

        if not fcf_done:
            cf_a_raw = stmts["cf_annual"]
            ocf_a = extract_series(cf_a_raw, ["Total Cash From Operating Activities", "Operating Cash Flow"])
            capex_a = extract_series(cf_a_raw, ["Capital Expenditures", "Capital Expenditure"])
            if not ocf_a.empty and not capex_a.empty:
                fcf_a = (ocf_a - capex_a).dropna()
                fcf_a = limit_series_years(fcf_a, years_hist)
                st.plotly_chart(
                    bar_discrete_from_series(fcf_a, "FCF (Annual, Billions)", "FCF (B)", "Annual", label_every=1),
                    use_container_width=True)


