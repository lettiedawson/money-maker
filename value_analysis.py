#!/usr/bin/env python3
"""
value_analysis.py — Value-investing analysis worksheet

Modes:
- Offline JSON: python value_analysis.py --input sample_inputs.json --out report
- FMP fetch:    python value_analysis.py --ticker AAPL --fetch --out apple
- Interactive:  python value_analysis.py  (menu)

Env:
- export FMP_KEY=HP7hZOlM9PCWdhoqZlYSlL0sqm2c2eJJ
Docs:
- Full as-reported financials:
  https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{SYMBOL}?period=annual&limit=50
- Quote:
  https://financialmodelingprep.com/api/v3/quote/{SYMBOL}
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import requests
import pathlib
import time
import shutil
import subprocess
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import ListFlowable, ListItem 
import re

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------------------
# Data models
# -------------------------------

@dataclass
class Fundamentals:
    company_name: str
    ticker: str
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    revenue_ttm: Optional[float] = None
    eps_ttm: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    fcf_series: Optional[List[float]] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

@dataclass
class ValuationInputs:
    price: float
    pe: Optional[float] = None
    forward_pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    shares_outstanding: Optional[float] = None
    eps_normalized: Optional[float] = None
    growth_rate: Optional[float] = None
    discount_rate: Optional[float] = None

@dataclass
class QualitativeNotes:
    moat: Optional[str] = None
    management: Optional[str] = None
    understandable: Optional[bool] = None
    earnings_quality: Optional[str] = None
    risks: Optional[str] = None

@dataclass
class Worksheet:
    fundamentals: Fundamentals
    valuation: ValuationInputs
    notes: QualitativeNotes

# -------------------------------
# Helpers
# -------------------------------

def _try_float(x) -> Optional[float]:
    try:
        if x in (None, "", "None"):
            return None
        return float(x)
    except Exception:
        return None

def _avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return None

def _safe_div(a, b) -> Optional[float]:
    try:
        return float(a) / float(b)
    except Exception:
        return None

def _sum_safely(items) -> Optional[float]:
    vals = []
    for it in items:
        v = _try_float(it)
        if v is not None:
            vals.append(v)
    return sum(vals) if vals else None

def cagr(series: List[float]) -> Optional[float]:
    if not series or len(series) < 2:
        return None
    first, last = series[0], series[-1]
    n = len(series) - 1
    if first <= 0 or last <= 0:
        return None
    return (last / first) ** (1 / n) - 1

def graham_intrinsic(eps: float, growth_rate: float, aa_yield: float = 0.04) -> Optional[float]:
    if eps is None or eps <= 0:
        return None
    if growth_rate is None or growth_rate < 0:
        growth_rate = 0
    g_whole = growth_rate * 100.0
    Y = aa_yield * 100.0 if aa_yield > 0 else 4.0
    return eps * (8.5 + (2 * g_whole)) * (4.4 / Y)

def margin_of_safety(current_price: float, intrinsic_value: Optional[float]) -> Optional[float]:
    if intrinsic_value is None or intrinsic_value <= 0:
        return None
    return (intrinsic_value - current_price) / intrinsic_value

def safe_pct(x: Optional[float]) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) and math.isfinite(x) else "—"

def safe_num(x: Optional[float], ndigits: int = 2) -> str:
    return f"{x:,.{ndigits}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "—"

def trend_arrow(series: Optional[List[float]]) -> str:
    if not series or len(series) < 2:
        return "—"
    if series[-1] > series[0]:
        return "↑"
    if series[-1] < series[0]:
        return "↓"
    return "→"

# -------------------------------
# Caching
# -------------------------------
def load_from_cache(cache_file: pathlib.Path, max_age_sec: int = 86400) -> Optional[dict]:
    """Return JSON data if cache exists and is recent enough, else None."""
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < max_age_sec:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None

def save_to_cache(cache_file: pathlib.Path, data: dict) -> None:
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save cache {cache_file}: {e}")

def clear_cache() -> None:
    """Delete the entire .cache directory (if it exists)."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print("🧹 Cache cleared (.cache/ removed).")
    else:
        print("No cache directory found. Nothing to clear.")

# -------------------------------
# Benchmarks
# -------------------------------

BENCHMARKS = {
    "roe_min": 0.15,
    "de_max": 1.0,
    "pb_max": 1.5,
    "ps_max": 2.0,
    "mos_target": 0.25,
    "div_payout_max": 0.60,
    "fcf_cagr_min": 0.05
}

# -------------------------------
# FMP (as-reported) client
# -------------------------------

FMP_BASE = "https://financialmodelingprep.com/api/v3"

def _get(url: str) -> Any:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def fmp_quote(ticker: str, key: str) -> Dict[str, Any]:
    cache_file = CACHE_DIR / f"{ticker}_quote.json"
    cached = load_from_cache(cache_file, max_age_sec=3600)  # refresh hourly
    if cached:
        return cached

    url = f"{FMP_BASE}/quote/{ticker}?apikey={key}"
    data = _get(url)
    if isinstance(data, list) and data:
        save_to_cache(cache_file, data[0])
        return data[0]
    return {}

def fmp_full_as_reported(ticker: str, key: str, period="annual", limit=10) -> List[Dict[str, Any]]:
    cache_file = CACHE_DIR / f"{ticker}_{period}_{limit}_as_reported.json"

    # try cache first
    cached = load_from_cache(cache_file)
    if cached:
        return cached

    url = f"{FMP_BASE}/financial-statement-full-as-reported/{ticker}?period={period}&limit={limit}&apikey={key}"
    data = _get(url)
    if isinstance(data, list):
        save_to_cache(cache_file, data)
    return data if isinstance(data, list) else []

# Field access helpers (as-reported tags vary)
import re

def _normalize_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _try_number(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def find_number(obj: Any, keys: List[str]) -> Optional[float]:
    """
    Search dict/list recursively for the first numeric value whose key matches any target.
    Matching is:
      1) exact normalized key match, then
      2) fuzzy: key contains the target (normalized).
    """
    targets = [_normalize_key(k) for k in keys]

    def _search(o: Any) -> Optional[float]:
        if isinstance(o, dict):
            # 1) exact matches
            norm_map = { _normalize_key(k): (k, v) for k, v in o.items() }
            for t in targets:
                if t in norm_map:
                    v = _try_number(norm_map[t][1])
                    if v is not None:
                        return v
            # 2) fuzzy contains
            for k, v in o.items():
                nk = _normalize_key(k)
                if any(t in nk for t in targets):
                    val = _try_number(v)
                    if val is not None:
                        return val
            # 3) recurse deeper
            for v in o.values():
                hit = _search(v)
                if hit is not None:
                    return hit
        elif isinstance(o, list):
            for it in o:
                hit = _search(it)
                if hit is not None:
                    return hit
        return None

    return _search(obj)


def build_ws_from_as_reported(ticker: str, key: str, period="annual", limit=10) -> Worksheet:
    # Price & shares (from quote)
    quote = fmp_quote(ticker, key)
    price = _try_float(quote.get("price"))
    market_cap = _try_float(quote.get("marketCap"))
    shares_out = _try_float(quote.get("sharesOutstanding"))
    company_name = quote.get("name") or ticker
    industry = None  # not available on this endpoint; could add /profile if desired

    # Pull full as-reported statements (list of dicts by period; [0] is most recent)
    rows = fmp_full_as_reported(ticker, key, period=period, limit=limit)

    # Try to collect key items from latest 2 periods for ratios that need averages
    latest = rows[0] if rows else {}
    prev   = rows[1] if len(rows) > 1 else {}

    # Income
    revenue = find_number(latest, [
        "revenuefromcontractwithcustomerexcludingassessedtax",
        "totalrevenue", "salesrevenuenet", "revenue"
    ])
    net_income = find_number(latest, [
        "netincomeloss", "profitloss", "netincome"
    ])

    # EPS
    eps_ttm = find_number(latest, [
        "earningspersharediluted", "earningspersharebasic", "eps"
    ])

    # Equity (book)
    equity_curr = find_number(latest, [
        "stockholdersequity", "totalstockholdersequity",
        "stockholdersequityincludingportionattributabletononcontrollinginterest"
    ])
    equity_prev = find_number(prev, [
        "stockholdersequity", "totalstockholdersequity",
        "stockholdersequityincludingportionattributabletononcontrollinginterest"
    ])

    # Debt: try direct, else components
    total_debt = find_number(latest, [
        "totaldebt", "longtermdebtnoncurrent", "longtermdebtcurrent", "debt"
    ])
    if total_debt is None:
        total_debt = _sum_safely([
            latest.get("longtermdebtnoncurrent"),
            latest.get("longtermdebtcurrent"),
            latest.get("commercialpaper"),
            latest.get("shorttermdebt"),
            latest.get("shorttermborrowings"),
        ])

    # Dividends / payout
    dividends_paid = find_number(latest, [
        "paymentsofdividends", "dividends", "paymentsofdividendscommonstock"
    ])

    # Shares (fallback if quote missing)
    if not shares_out:
        shares_out = find_number(latest, [
            "commonstocksharesoutstanding",
            "weightedaveragenumberofdilutedsharesoutstanding",
            "weightedaveragenumberofsharesoutstandingbasic"
        ])

    # Cash flow items (to compute FCF)
    fcf_series = []
    for row in reversed(rows):  # oldest -> latest
        ocf = find_number(row, [
            "netcashprovidedbyusedinoperatingactivities",
            "netcashprovidedbyusedinoperatingactivitiescontinuingoperations",
            "cashfromoperatingactivities"
        ])
        capex = find_number(row, [
            "paymentstoacquirepropertyplantandequipment",
            "capitalexpenditures", "purchaseofpropertyandequipment"
        ])
        if ocf is None or capex is None:
            continue
        # Note: capex is usually positive outflow here; FCF = OCF - CapEx
        fcf_series.append(ocf - capex)
    fcf_series = fcf_series or None

    # Multiples
    pe = _safe_div(price, eps_ttm) if price and eps_ttm else None
    ps = (market_cap / revenue) if market_cap and revenue and revenue > 0 else None
    pb = (market_cap / equity_curr) if market_cap and equity_curr and equity_curr > 0 else None

    # ROE and Debt/Equity
    roe = _safe_div(net_income, equity_prev) if net_income and equity_prev else None
    debt_to_equity = _safe_div(total_debt, equity_curr) if total_debt and equity_curr else None

    # Dividend payout ratio
    payout_ratio = _safe_div(dividends_paid, net_income) if dividends_paid and net_income else None

    fundamentals = Fundamentals(
        company_name=company_name,
        ticker=ticker,
        industry=industry,
        market_cap=market_cap,
        revenue_ttm=revenue,
        eps_ttm=eps_ttm,
        roe=roe,
        debt_to_equity=debt_to_equity,
        fcf_series=fcf_series,
        dividend_yield=None,     # could compute from trailing dividends & price using dividend history
        payout_ratio=payout_ratio
    )

    growth_rate = cagr(fcf_series) if fcf_series else None

    valuation = ValuationInputs(
        price=price or 0.0,
        pe=pe,
        forward_pe=None,
        pb=pb,
        ps=ps,
        shares_outstanding=shares_out,
        eps_normalized=eps_ttm,
        growth_rate=growth_rate,
        discount_rate=None
    )

    notes = QualitativeNotes()
    return Worksheet(fundamentals, valuation, notes)

# -------------------------------
# Report building
# -------------------------------

def build_report(ws: Worksheet) -> Dict[str, Any]:
    f = ws.fundamentals
    v = ws.valuation

    fcf_cagr_val = cagr(f.fcf_series) if f.fcf_series else None
    graham_iv = graham_intrinsic(
        eps=v.eps_normalized or f.eps_ttm or 0,
        growth_rate=(v.growth_rate if v.growth_rate is not None else (fcf_cagr_val or 0))
    )
    mos = margin_of_safety(v.price, graham_iv)

    checks = {
        "ROE > 15%": f.roe is not None and f.roe >= BENCHMARKS["roe_min"],
        "D/E < 1": f.debt_to_equity is not None and f.debt_to_equity < BENCHMARKS["de_max"],
        "P/B < 1.5": v.pb is not None and v.pb < BENCHMARKS["pb_max"],
        "P/S < 2": v.ps is not None and v.ps < BENCHMARKS["ps_max"],
        "Dividend payout < 60%": f.payout_ratio is not None and f.payout_ratio < BENCHMARKS["div_payout_max"],
        "FCF CAGR ≥ 5%": fcf_cagr_val is not None and fcf_cagr_val >= BENCHMARKS["fcf_cagr_min"],
        "MOS ≥ 25%": mos is not None and mos >= BENCHMARKS["mos_target"]
    }

    passed = sum(1 for ok in checks.values() if ok)
    if passed >= 5:
        decision = "Buy"
    elif passed >= 3:
        decision = "Consider"
    else:
        decision = "Pass"

    return {
        "computed": {"fcf_cagr": fcf_cagr_val, "graham_intrinsic": graham_iv, "margin_of_safety": mos},
        "checks": checks,
        "decision": decision
    }

def write_markdown(ws: Worksheet, results: Dict[str, Any], out_path: str) -> None:
    f = ws.fundamentals
    v = ws.valuation
    c = results["computed"]
    checks = results["checks"]

    lines = []
    lines.append(f"# Value Investing Analysis — {f.company_name} ({f.ticker})")
    lines.append(f"_Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    lines.append("## 1) Company Fundamentals")
    lines.append("| Metric | Value | Benchmark/Notes |")
    lines.append("|---|---:|---|")
    lines.append(f"| Industry | {f.industry or '—'} |  |")
    lines.append(f"| Market Cap ($) | {safe_num(f.market_cap)} |  |")
    lines.append(f"| Revenue (TTM, $) | {safe_num(f.revenue_ttm)} | positive & steady |")
    lines.append(f"| EPS (TTM) | {safe_num(f.eps_ttm)} | growth steady |")
    lines.append(f"| ROE | {safe_pct(f.roe)} | > 15% preferred |")
    lines.append(f"| Debt/Equity | {safe_num(f.debt_to_equity)} | < 1 conservative |")
    lines.append(f"| Free Cash Flow trend | {trend_arrow(f.fcf_series)} | positive & growing |")
    lines.append(f"| FCF CAGR | {safe_pct(c['fcf_cagr'])} | ≥ 5% |")
    lines.append(f"| Dividend Yield | {safe_pct(f.dividend_yield)} |  |")
    lines.append(f"| Payout Ratio | {safe_pct(f.payout_ratio)} | < 60% sustainable |\n")

    lines.append("## 2) Valuation")
    lines.append("| Metric | Value | Benchmark/Notes |")
    lines.append("|---|---:|---|")
    lines.append(f"| Price | {safe_num(v.price)} |  |")
    lines.append(f"| P/E (TTM) | {safe_num(v.pe)} | compare to industry & history |")
    lines.append(f"| Forward P/E | {safe_num(v.forward_pe)} |  |")
    lines.append(f"| P/B | {safe_num(v.pb)} | < 1.5 often attractive |")
    lines.append(f"| P/S | {safe_num(v.ps)} | < 2 often attractive |")
    lines.append(f"| EPS (normalized) | {safe_num(v.eps_normalized or f.eps_ttm)} |  |")
    lines.append(f"| Growth rate (assumed) | {safe_pct(v.growth_rate)} | can use FCF CAGR |")
    lines.append(f"| Intrinsic Value (Graham) | {safe_num(c['graham_intrinsic'])} | heuristic |")
    lines.append(f"| Margin of Safety | {safe_pct(c['margin_of_safety'])} | target 20–40% |\n")

    lines.append("## 3) Checklist")
    lines.append("| Rule | Pass? |")
    lines.append("|---|:--:|")
    for k, ok in checks.items():
        lines.append(f"| {k} | {'✅' if ok else '❌'} |")

    lines.append("\n## 4) Qualitative Notes")
    lines.append(f"- **Moat**: {'—'}")
    lines.append(f"- **Management**: {'—'}")
    lines.append(f"- **Understandable?**: {'—'}")
    lines.append(f"- **Earnings Quality**: {'—'}")
    lines.append(f"- **Risks**: {'—'}\n")

    lines.append("## 5) Decision")
    lines.append(f"- **Decision**: **{results['decision']}**")
    lines.append(f"- **Current Price**: {safe_num(v.price)}")
    lines.append(f"- **MOS Target Met?**: {'Yes' if checks.get('MOS ≥ 25%') else 'No'}\n")

    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines))

def open_file(path: str) -> None:
    """Open a file with the system default app."""
    try:
        if sys.platform.startswith("darwin"):      # macOS
            subprocess.run(["open", path], check=False)
        elif sys.platform.startswith("win"):       # Windows
            os.startfile(path)  # type: ignore
        elif sys.platform.startswith("linux"):     # Linux
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print(f"⚠️ Could not auto-open {path}: {e}")

def write_csv(ws: Worksheet, results: Dict[str, Any], out_path: str) -> None:
    f = ws.fundamentals
    v = ws.valuation
    c = results["computed"]
    checks = results["checks"]

    rows = [
        ["company_name", f.company_name],
        ["ticker", f.ticker],
        ["industry", f.industry],
        ["market_cap", f.market_cap],
        ["revenue_ttm", f.revenue_ttm],
        ["eps_ttm", f.eps_ttm],
        ["roe", f.roe],
        ["debt_to_equity", f.debt_to_equity],
        ["dividend_yield", f.dividend_yield],
        ["payout_ratio", f.payout_ratio],
        ["price", v.price],
        ["pe", v.pe],
        ["forward_pe", v.forward_pe],
        ["pb", v.pb],
        ["ps", v.ps],
        ["eps_normalized", v.eps_normalized or f.eps_ttm],
        ["growth_rate", v.growth_rate],
        ["graham_intrinsic", c["graham_intrinsic"]],
        ["margin_of_safety", c["margin_of_safety"]],
        ["fcf_cagr", c["fcf_cagr"]],
        ["decision", results["decision"]],
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

def sanitize(s: str) -> str:
    if s is None:
        return ""
    # Replace problematic Unicode with ASCII-friendly versions
    replacements = {
        "—": "-", "–": "-", "•": "-", "•": "-",
        "↑": "^", "↓": "v", "→": "->",
        "✅": "Yes", "❌": "No",
        "’": "'", "“": '"', "”": '"', " ": " ",
        "…": "...", "■": "-", "•": "-"
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # strip other non-ascii to be safe
    return re.sub(r"[^\x00-\x7F]", "", s)

def write_pdf(ws: Worksheet, results: Dict[str, Any], out_path: str) -> None:
    f = ws.fundamentals
    v = ws.valuation
    c = results["computed"]
    checks = results["checks"]

    # Add gentle margins to reduce overflow risk
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    flow = []

    # Title
    flow.append(Paragraph(sanitize(f"Value Investing Analysis — {f.company_name} ({f.ticker})"), styles["Title"]))
    flow.append(Paragraph(sanitize(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"), styles["Normal"]))
    flow.append(Spacer(1, 12))

    # Fundamentals table
    flow.append(Paragraph(sanitize("Company Fundamentals"), styles["Heading2"]))
    fundamentals_data = [
        [sanitize("Metric"), sanitize("Value"), sanitize("Benchmark/Notes")],
        [sanitize("Industry"), sanitize(f.industry or "—"), sanitize("")],
        [sanitize("Market Cap ($)"), sanitize(safe_num(f.market_cap)), sanitize("")],
        [sanitize("Revenue (TTM, $)"), sanitize(safe_num(f.revenue_ttm)), sanitize("positive & steady")],
        [sanitize("EPS (TTM)"), sanitize(safe_num(f.eps_ttm)), sanitize("growth steady")],
        [sanitize("ROE"), sanitize(safe_pct(f.roe)), sanitize("> 15% preferred")],
        [sanitize("Debt/Equity"), sanitize(safe_num(f.debt_to_equity)), sanitize("< 1 conservative")],
        [sanitize("FCF CAGR"), sanitize(safe_pct(c["fcf_cagr"])), sanitize(">= 5%")],
        [sanitize("Dividend Yield"), sanitize(safe_pct(f.dividend_yield)), sanitize("")],
        [sanitize("Payout Ratio"), sanitize(safe_pct(f.payout_ratio)), sanitize("< 60% sustainable")],
    ]
    t = Table(fundamentals_data, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey)]))
    flow.append(t)
    flow.append(Spacer(1, 12))

    # Valuation table
    flow.append(Paragraph(sanitize("Valuation"), styles["Heading2"]))
    valuation_data = [
        [sanitize("Metric"), sanitize("Value"), sanitize("Benchmark/Notes")],
        [sanitize("Price"), sanitize(safe_num(v.price)), sanitize("")],
        [sanitize("P/E (TTM)"), sanitize(safe_num(v.pe)), sanitize("compare to industry & history")],
        [sanitize("Forward P/E"), sanitize(safe_num(v.forward_pe)), sanitize("")],
        [sanitize("P/B"), sanitize(safe_num(v.pb)), sanitize("< 1.5 often attractive")],
        [sanitize("P/S"), sanitize(safe_num(v.ps)), sanitize("< 2 often attractive")],
        [sanitize("EPS (normalized)"), sanitize(safe_num(v.eps_normalized or f.eps_ttm)), sanitize("")],
        [sanitize("Growth rate (assumed)"), sanitize(safe_pct(v.growth_rate)), sanitize("can use FCF CAGR")],
        [sanitize("Intrinsic Value (Graham)"), sanitize(safe_num(c["graham_intrinsic"])), sanitize("heuristic")],
        [sanitize("Margin of Safety"), sanitize(safe_pct(c["margin_of_safety"])), sanitize("target 20-40%")],
    ]
    t = Table(valuation_data, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey)]))
    flow.append(t)
    flow.append(Spacer(1, 12))

    # Checklist
    flow.append(Paragraph(sanitize("Checklist"), styles["Heading2"]))
    check_data = [[sanitize("Rule"), sanitize("Pass?")]]
    for k, ok in checks.items():
        check_data.append([sanitize(k), sanitize("Yes" if ok else "No")])  # avoid emoji
    t = Table(check_data, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey)]))
    flow.append(t)
    flow.append(Spacer(1, 18))

    # Decision
    flow.append(Paragraph(sanitize("Decision"), styles["Heading2"]))
    flow.append(Paragraph(sanitize(f"Decision: {results['decision']}"), styles["Normal"]))
    flow.append(Paragraph(sanitize(f"Current Price: {safe_num(v.price)}"), styles["Normal"]))
    flow.append(Paragraph(sanitize(f"MOS Target Met?: {'Yes' if checks.get('MOS ≥ 25%') else 'No'}"), styles["Normal"]))
    flow.append(Spacer(1, 18))

    # How to Interpret
    flow.append(Paragraph(sanitize("How to Interpret the Checklist"), styles["Heading2"]))
    interpret_items = [
        "5-7 passes -> Buy: quality + value likely align.",
        "3-4 passes -> Consider: dig deeper; some risks/valuation flags.",
        "0-2 passes -> Pass: quality or valuation probably insufficient now.",
        "Rules are guardrails, not guarantees. Prefer multi-year consistency over single-period spikes.",
        "If business quality is exceptional (durable moat, steady reinvestment), you may accept slightly higher P/B or lower MOS—but document why."
    ]
    # Sanitize each bullet and render as simple Paragraphs (ListFlowable is fine too)
    for item in interpret_items:
        flow.append(Paragraph(sanitize("• " + item), styles["BodyText"]))
    flow.append(Spacer(1, 18))

    # Appendix (sanitized Paragraphs; fixed widths to ensure wrapping)
    flow.append(Paragraph(sanitize("Appendix: Definitions & Why They Matter"), styles["Heading2"]))
    defs = [
        ("Return on Equity (ROE)",
         "Net income / average shareholders' equity. Gauges how efficiently the company turns owner capital into profits. "
         "Higher & consistent ROE (e.g., >15%) often signals a durable advantage or strong execution."),
        ("Debt-to-Equity (D/E)",
         "Total debt / shareholders' equity. Indicates leverage. Lower D/E (<1) gives more flexibility in downturns and reduces financial risk."),
        ("Earnings per Share (EPS)",
         "Profit attributable to common shareholders / diluted shares. Common input for valuation (e.g., P/E, Graham formula)."),
        ("Price-to-Book (P/B)",
         "Market cap / shareholders' equity. Rough proxy for paying above/below accounting book value. Sub-1.5 is a classic value screen (industry-dependent)."),
        ("Price-to-Sales (P/S)",
         "Market cap / trailing 12-month revenue. Useful when earnings are noisy; lower P/S indicates cheaper revenue but margins matter."),
        ("Free Cash Flow (FCF)",
         "Operating cash flow - capital expenditures. Cash available after maintaining/growing the asset base. Fuels buybacks, dividends, and debt paydown."),
        ("FCF CAGR",
         "Compound annual growth rate of FCF over multiple years. Measures direction & durability of cash generation; sustained positive CAGR is a quality signal."),
        ("Payout Ratio",
         "Dividends paid / net income. Tests dividend sustainability. Lower ratios (<60%) leave room for reinvestment or shocks."),
        ("Graham Intrinsic Value (heuristic)",
         "Ben Graham's formula using EPS and growth (adjusted by bond yield). It's a rough anchor, not a precise valuation."),
        ("Margin of Safety (MOS)",
         "(Intrinsic value - price) / intrinsic value. A discount that protects against errors or adverse surprises; 20-40% is a common target.")
    ]

    def_table = [[sanitize("Metric"), sanitize("Definition & Rationale")]]
    for name, text in defs:
        def_table.append([
            Paragraph(sanitize(name), styles["BodyText"]),
            Paragraph(sanitize(text), styles["BodyText"])
        ])

    # Safer column widths to avoid overflow; text will wrap inside Paragraphs
    t = Table(def_table, hAlign="LEFT", colWidths=[150, 360])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 12))

    # Build PDF
    doc.build(flow)
   

# -------------------------------
# I/O helpers
# -------------------------------

def load_offline_inputs(path: str) -> Worksheet:
    with open(path, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
    fundamentals = Fundamentals(**data["fundamentals"])
    valuation = ValuationInputs(**data["valuation"])
    notes = QualitativeNotes(**data.get("notes", {}))
    return Worksheet(fundamentals, valuation, notes)

def generate_and_save(ws: Worksheet, out_stem: str) -> None:
    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    results = build_report(ws)

    pdf_path = f"{out_stem}.pdf"
    csv_path = f"{out_stem}.csv"

    write_pdf(ws, results, pdf_path)
    write_csv(ws, results, csv_path)

    print(f"✅ Wrote {pdf_path} and {csv_path}")

    # Auto-open PDF
    open_file(pdf_path)


# -------------------------------
# Interactive menu
# -------------------------------

def interactive_menu(fmp_key: Optional[str]):
    print("=== Value Investing Analysis Menu ===")
    print("1) Load from JSON input file")
    print("2) Fetch by ticker via FMP (as reported)")
    print("3) Delete cache (.cache/)")
    print("Q) Quit")
    choice = input("Choose an option: ").strip().lower()

    if choice == "1":
        # Load from local JSON and generate report
        input_path = input("Enter path to JSON input file: ").strip()
        if not input_path:
            print("No path provided.")
            return
        try:
            ws = load_offline_inputs(input_path)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return
        out_stem = input("Enter output file stem (default: report): ").strip() or "report"
        generate_and_save(ws, out_stem)

    elif choice == "2":
        # Fetch via FMP as-reported, then generate report
        ticker = input("Enter ticker (e.g., AAPL): ").strip().upper()
        if not ticker:
            print("No ticker provided.")
            return
        key = fmp_key or input("Enter FMP API key (leave blank to cancel): ").strip()
        if not key:
            print("No API key provided.")
            return
        period = (input("Period [annual/quarter] (default: annual): ").strip().lower() or "annual")
        if period not in {"annual", "quarter"}:
            print("Invalid period; using 'annual'.")
            period = "annual"
        try:
            limit_str = input("How many periods? (default: 10): ").strip()
            limit = int(limit_str) if limit_str else 10
        except ValueError:
            print("Invalid number; using 10.")
            limit = 10

        try:
            ws = build_ws_from_as_reported(ticker, key, period=period, limit=limit)
        except Exception as e:
            print(f"Error fetching data from FMP: {e}")
            return

        out_stem = input("Enter output file stem (default: report): ").strip() or "report"
        generate_and_save(ws, out_stem)

    elif choice == "3":
        # Delete cache directory
        confirm = input("Are you sure you want to delete the cache? (y/N): ").strip().lower()
        if confirm == "y":
            clear_cache()
        else:
            print("Canceled.")

    elif choice == "q":
        print("Goodbye!")
    else:
        print("Invalid choice.")

# -------------------------------
# CLI
# -------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate a value-investing analysis report (Markdown + CSV). "
                    "Run with no arguments for an interactive menu."
    )
    p.add_argument("--input", help="Path to JSON with offline inputs")
    p.add_argument("--ticker", help="Ticker to fetch via FMP (as reported) — requires --fetch")
    p.add_argument("--fetch", action="store_true", help="Use FMP as-reported endpoint")
    p.add_argument("--out", default="report", help="Output file stem (no extension)")
    p.add_argument("--fmp-key", help="FMP API key (overrides env FMP_KEY)")
    p.add_argument("--period", default="annual", choices=["annual", "quarter"], help="Financials period")
    p.add_argument("--limit", type=int, default=10, help="How many periods to pull")
    p.add_argument("--clear-cache", action="store_true", help="Delete the local .cache folder and exit")  # NEW

    args = p.parse_args()
    fmp_key = args.fmp_key or os.getenv("FMP_KEY")

    # NEW: handle cache clearing as a one-off action
    if args.clear_cache:
        clear_cache()
        return

    if args.input:
        ws = load_offline_inputs(args.input)
        generate_and_save(ws, args.out)
    elif args.fetch and args.ticker:
        if not fmp_key:
            raise SystemExit("No FMP API key. Use --fmp-key or export FMP_KEY.")
        ws = build_ws_from_as_reported(args.ticker.upper(), fmp_key, period=args.period, limit=args.limit)
        generate_and_save(ws, args.out)
    else:
        interactive_menu(fmp_key)


if __name__ == "__main__":
    main()
