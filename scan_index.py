"""Scan a stock-market index for buy/sell opportunities.

Runs the complete TradingAgents pipeline (market, social, news and fundamentals
analysts, the bull/bear debate, the trader, the risk debate and the portfolio
manager) over a chosen slice of an index's constituents and writes one ranked
markdown report of the opportunities, next to the full per-ticker agent
reports the framework already saves.

Usage (from the repo root, with the venv python):

    python scan_index.py sp500 --num 10                # top 10 by market cap
    python scan_index.py nasdaq100 --num 20 --rank first
    python scan_index.py dow --num 5 --rank random
    python scan_index.py russell2000 --num 15          # ranks via the IWM ETF holdings list
    python scan_index.py --tickers NVDA,AAPL,MSFT      # an explicit custom list
    python scan_index.py sp500 --file my_tickers.csv   # one ticker per line
    python scan_index.py sp500 --num 20 --resume       # continue an interrupted scan

Model/provider configuration comes from the environment exactly as for any
other run: put the provider key (OPENAI_API_KEY, ...) and the optional
FRED_API_KEY in .env, and switch models with TRADINGAGENTS_LLM_PROVIDER /
TRADINGAGENTS_DEEP_THINK_LLM / TRADINGAGENTS_QUICK_THINK_LLM. No key in a
.env: the script fails fast with a pointer, before any ticker is analyzed.

Cost warning: every analyzed ticker is a complete multi-agent run. With the
default analyst lineup expect minutes and a nontrivial API spend per ticker;
``--num`` and ``--analysts`` are the budget knobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

import requests

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.rating import run_rating
from tradingagents.dataflows.symbols import safe_ticker_component

logger = logging.getLogger("scan_index")

WIKIPEDIA = "https://en.wikipedia.org/wiki"

_NASDAQ_LIST_API = "https://api.nasdaq.com/api/quote/list-type/{list_id}"

# index key -> how to fetch its constituents. Wikipedia dropped the
# constituents tables from its Nasdaq-100 / Dow pages and free listings for
# Russell 2000 do not exist, so each index gets the source that still works:
# the official Nasdaq list API for NASDAQ-100 (exact 101 entries, with market
# cap included), Wikipedia for the S&P 500 with SlickCharts pages as a
# fallback, and an explicit --file for Russell 2000.
_INDEX_SOURCES: dict[str, dict] = {
    "sp500": {
        "type": "wikipedia",
        "url": f"{WIKIPEDIA}/List_of_S%26P_500_companies",
        "columns": ("Symbol", "Ticker"),
        "fallback": {"type": "slickcharts", "url": "https://www.slickcharts.com/sp500"},
    },
    "nasdaq100": {
        "type": "nasdaq_api",
        "url": _NASDAQ_LIST_API.format(list_id="nasdaq100"),
    },
    "dow": {
        "type": "slickcharts",
        "url": "https://www.slickcharts.com/dowjones",
    },
    "russell2000": {
        "type": "file_only",
        "help": (
            "Russell 2000 has no freely maintained constituents list online.\n"
            "Export one (FTSE Russell, your broker's screener, or any CSV with a\n"
            "Symbol or Ticker column) and pass it with --file:\n"
            "    python scan_index.py russell2000 --file russell2000.csv --num 15"
        ),
    },
}

# Index/ETF tickers a symbol page may emit as navigation links, not members.
_NAV_SYMBOLS = frozenset({"SPY", "QQQ", "DIA", "SPX", "NDX", "IWM", "IWB", "IWC"})

_HTTP_HEADERS = {"User-Agent": "TradingAgents index scanner (research)"}

# Market-cap snapshots could change at open/close, never minute to minute.
_CAP_CACHE_TTL = 5 * 24 * 3600

# Ranker order, most actionable first: a strong buy call and a strong sell
# call are both opportunities; holds and unparseable decisions trail.
_OPPORTUNITY_ORDER = ("Buy", "Sell", "Overweight", "Underweight", "Hold", "REVIEW")


def _fetch_url(url: str, timeout: int = 60, headers: dict | None = None) -> bytes:
    response = requests.get(url, headers=headers or _HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.content


def _clean_text(chunks: list[str]) -> str:
    return " ".join(" ".join(chunks).split())


class _WikiTableParser(HTMLParser):
    """Collect every ``wikitable`` row: one row per ``<tr>``, one cell per ``<th>``/``<td>``.

    Uses only the stdlib so the scanner needs no lxml/bs4 on top of the
    framework's own dependencies. Nested tables are ignored, keeping only the
    outermost level of any wikitable.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_stack: list[dict] = []   # every <table>, wiki or not, nested
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            wiki = any(k == "class" and "wikitable" in (v or "") for k, v in attrs)
            self._table_stack.append({"wiki": wiki, "rows": [], "cells": None})
        elif self._table_stack:
            table = self._table_stack[-1]
            if table["wiki"] and tag in ("tr", "th", "td"):
                if tag == "tr":
                    table["cells"] = []
                self._chunks = []

    def handle_endtag(self, tag):
        if tag in ("table", "tr", "th", "td") and self._table_stack:
            table = self._table_stack[-1]
            if not table["wiki"]:
                if tag == "table":
                    self._table_stack.pop()
                return
            if tag in ("th", "td"):
                table["cells"].append(_clean_text(self._chunks))
            elif tag == "tr" and table["cells"] is not None and table["cells"]:
                table["rows"].append(table["cells"])
            elif tag == "table":
                if table["rows"]:
                    self.tables.append(table["rows"])
                self._table_stack.pop()
            self._chunks = []

    def handle_data(self, data):
        if self._table_stack and self._table_stack[-1]["wiki"]:
            self._chunks.append(data)


def _per_share_symbol(symbol: str) -> str:
    """Wikipedia spells share classes ``BRK.B``; Yahoo wants ``BRK-B``."""
    return re.sub(r"(?<=[A-Z])\.([A-Z]{1,2})$", r"-\1", symbol.strip().upper())


def _constituents_from_wikipedia(url: str, accepted_columns: tuple[str, ...],
                                 title_hint: str | None) -> list[str]:
    """The ticker column of an index's constituents wikitable, in page order."""
    parser = _WikiTableParser()
    parser.feed(_fetch_url(url).decode("utf-8", errors="replace"))
    best: list[str] = []
    for rows in parser.tables:
        if title_hint and title_hint not in _clean_text(rows[0]).lower():
            continue
        header = [cell.lower() for cell in rows[0]]
        column = next(
            (header.index(c.lower()) for c in accepted_columns if c.lower() in header),
            None,
        )
        if column is None:
            continue
        tickers = [row[column] for row in rows[1:] if column < len(row)]
        if len(tickers) > len(best):
            best = tickers
    if not best:
        raise ValueError(f"no constituents table found on {url}; its layout may have changed")
    # Drop any stray trailing annotate rows (".X" share classes kept apart).
    return [_per_share_symbol(t) for t in best if re.fullmatch(r"[A-Z0-9.\-]{1,10}", t)]


def _constituents_from_nasdaq_api(url: str) -> list[str]:
    """Official Nasdaq list API: exact constituent symbols in index order."""
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    payload = json.loads(_fetch_url(url, headers=browser_headers).decode("utf-8"))
    rows = payload.get("data", {}).get("data", {}).get("rows") or []
    tickers = [
        _per_share_symbol((row.get("Symbol") or row.get("symbol") or "").strip())
        for row in rows
    ]
    tickers = [t for t in tickers if re.fullmatch(r"[A-Z0-9.\-]{1,10}", t)]
    if not tickers:
        raise ValueError(f"the Nasdaq list API at {url} returned no rows; it may have changed")
    return list(dict.fromkeys(tickers))


def _constituents_from_slickcharts(url: str) -> list[str]:
    """Index page tickers, embedded as ``/symbol/<ticker>`` links in page order."""
    html = _fetch_url(url).decode("utf-8", errors="replace")
    tickers = [
        _per_share_symbol(s)
        for s in re.findall(r'href="/symbol/([A-Za-z0-9.\-]+)"', html)
        if s.upper() not in _NAV_SYMBOLS
    ]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError(f"no /symbol/ links found on {url}; the page layout may have changed")
    return tickers


def get_constituents(index: str, file: Path | None = None) -> list[str]:
    """The index's tickers, from --file (one per line / optional first CSV column),
    an explicit well-known index source, or Russell via IWM."""
    if file is not None:
        text = file.read_text()
        if file.suffix.lower() == ".csv":
            reader = csv.reader(text.splitlines())
            first = next(reader)
            column = next(
                (i for i, name in enumerate(first)
                 if name.strip().lower() in ("symbol", "ticker", "ticker symbol")),
                0,
            )
            return [_per_share_symbol(r[column]) for r in reader if r and r[column].strip()]
        return [_per_share_symbol(line) for line in text.splitlines() if line.strip()]

    source = _INDEX_SOURCES[index]
    if source["type"] == "file_only":
        raise ValueError(source["help"])
    try:
        if source["type"] == "wikipedia":
            return _constituents_from_wikipedia(source["url"], source["columns"], source.get("title_hint"))
        if source["type"] == "nasdaq_api":
            return _constituents_from_nasdaq_api(source["url"])
        return _constituents_from_slickcharts(source["url"])
    except Exception as error:
        fallback = source.get("fallback")
        if fallback is None:
            raise
        logger.warning("primary source for %s failed (%s); trying %s", index, error, fallback["url"])
        return _constituents_from_slickcharts(fallback["url"])


def _effective_symbol(symbol: str) -> str:
    """The symbol to quote to yfinance."""
    from tradingagents.dataflows.symbols import normalize_symbol

    return normalize_symbol(symbol)


def market_caps(symbols: list[str], cache_dir: Path, index_key: str) -> dict[str, float]:
    """Latest market cap per symbol, cached for a few days between scans.

    Missing quotes come back as absent keys rather than zeros, so a
    ``--rank market_cap`` selection never prefers a failed quote.
    """
    cache_file = cache_dir / "index_scan" / f"{index_key}_caps.json"
    caps: dict[str, float] = {}
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached.get("ts", 0) < _CAP_CACHE_TTL:
            for key in cached.get("caps", {}):
                if key in symbols:
                    caps[key] = cached["caps"][key]
            if len(caps) == len(symbols):
                return caps

    import yfinance as yf

    missing = [s for s in symbols if s not in caps]
    logger.info("fetching market caps for %d of %d symbols", len(missing), len(symbols))
    done = 0

    def one(symbol: str) -> tuple[str, float | None]:
        nonlocal done
        try:
            return symbol, float(yf.Ticker(_effective_symbol(symbol)).fast_info.get("marketCap") or 0)
        except Exception as exc:
            logger.debug("market cap for %s failed: %s", symbol, exc)
            return symbol, None
        finally:
            done += 1
            if done % 100 == 0:
                print(f"  market caps: {done}/{len(missing)} fetched...", flush=True)

    with ThreadPoolExecutor(max_workers=10) as pool:
        for symbol, cap in pool.map(one, missing):
            if cap:
                caps[symbol] = cap

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"ts": time.time(), "caps": caps}))
    return caps


def pick_tickers(index: str, tickers: list[str], rank: str, num: int,
                 cache_dir: Path) -> list[str]:
    """The slice of the index the scan analyzes."""
    if len(tickers) <= num:
        return tickers
    if rank == "random":
        return sorted(random.sample(tickers, num))
    if rank == "market_cap":
        print(f"ranking {len(tickers)} constituents by market cap ...", flush=True)
        caps = market_caps(tickers, cache_dir, index)
        ranked = sorted(tickers, key=lambda t: -caps.get(t, 0.0))
        quoted = sum(1 for t in ranked[:num] if t in caps)
        if quoted == 0:
            print("  no market caps could be fetched; falling back to list order")
        else:
            print(f"  {quoted}/{num} of the top slice quoted OK")
        return ranked[:num]
    return tickers[:num]


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# A decision opens with its rating, then "Executive Summary" — marked either
# as a markdown heading ("## Executive Summary") or as bold emphasis
# ("**Executive Summary**"), so both shapes start the summary block.
_SECTION_HEADING_RE = re.compile(r"^#{1,6}\s")
_EXEC_SUMMARY_RE = re.compile(r"^[#*_\s`>]*executive\s+summary[!.*_`:]*$", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^[\s\-*_>=]+$")


def _extract_executive_summary(decision_md: str, limit: int = 380) -> str:
    """The Portfolio Manager's executive summary, as one line, truncated."""
    out: list[str] = []
    capture = False
    for line in decision_md.splitlines():
        stripped = line.strip()
        if not stripped or _SEPARATOR_RE.match(stripped):
            continue
        if _EXEC_SUMMARY_RE.match(stripped):
            capture = True
            continue
        if _SECTION_HEADING_RE.match(stripped) or _EXEC_SUMMARY_RE.match(stripped.lstrip("*_# `")):
            if capture:
                break
            continue
        if capture:
            out.append(stripped)
    if not out:
        # No executive-summary block found: fall back to the first prose lines.
        out = [l for l in decision_md.splitlines()
               if l.strip() and not l.strip().startswith("#")][:3]
    text = " ".join(out)
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _report_line(row: dict) -> str:
    return json.dumps(
        {k: row.get(k) for k in ("ticker", "index", "trade_date", "signal",
                                 "exec_summary", "decision", "error")}
    )


def _load_rows(run_dir: Path) -> list[dict]:
    log = run_dir / "opportunities.jsonl"
    if not log.exists():
        return []
    rows = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _rank_key(row: dict) -> tuple[float, str]:
    try:
        order = _OPPORTUNITY_ORDER.index(row.get("signal") or "REVIEW")
    except ValueError:
        order = len(_OPPORTUNITY_ORDER)
    return (order, row.get("ticker", ""))


def write_report(index: str, trade_date: str, rows: list[dict], run_dir: Path) -> Path:
    """The ranked markdown report over every result row, coded or failed."""
    ok = [row for row in rows if row.get("decision") and not row.get("error")]
    failed = [row for row in rows if row.get("error")]
    ranked = sorted(ok, key=_rank_key)
    counts = {}
    for row in ranked:
        counts[row["signal"]] = counts.get(row["signal"], 0) + 1

    lines = [
        f"# {index.upper()} opportunities — {trade_date}",
        "",
        f"{len(ranked)} analyzed · "
        + " · ".join(f"{signal} {count}" for signal, count in
                     sorted(counts.items(), key=lambda c: _OPPORTUNITY_ORDER.index(c[0])))
        + (f" · {len(failed)} failed" if failed else ""),
        "",
        "Every line below is the Portfolio Manager's call after the full research",
        "pipeline; the reasoning column is its own executive summary. Ratings are",
        "relative to the other members of the scanned list, not an absolute",
        "recommendation.",
        "",
        "| # | Ticker | Rating | Key reasons |",
        "|---|--------|--------|-------------|",
    ]
    for n, row in enumerate(ranked, 1):
        summary = (row.get("exec_summary") or "").replace("|", "\\|")
        lines.append(f"| {n} | **{row['ticker']}** | {row['signal']} | {summary} |")
    if not ranked:
        lines.append("| — | none — every run failed or produced no decision | | |")

    lines += ["", "## Per-ticker detail", ""]
    for row in ranked:
        lines += [
            f"### {row['ticker']} — {row['signal']}",
            "",
            row.get("exec_summary") or "",
            "",
            "<details><summary>full trading decision</summary>",
            "",
            row.get("decision", "").strip() or "_no decision text_",
            "",
            "</details>",
            "",
        ]
    for row in sorted(failed, key=lambda r: r.get("ticker", "")):
        lines += [f"### {row['ticker']} — ERROR", "", f"`{row.get('error')}`", ""]
    if failed:
        lines += [
            "Failed cells kept their place in line for a re-run: pass ``--resume``",
            "with the same index and num to retry only these.",
            "",
        ]

    report = run_dir / f"opportunities_{index}_{trade_date}.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a stock-market index for buy/sell opportunities.",
    )
    parser.add_argument("index", nargs="?",
                        help="one of: " + ", ".join(_INDEX_SOURCES))
    parser.add_argument("--num", type=int, default=10,
                        help="how many tickers to analyze (default 10)")
    parser.add_argument("--rank", choices=("market_cap", "first", "random"),
                        default="market_cap",
                        help="which slice of the index to pick (default market_cap)")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="trade date, YYYY-MM-DD, no later than today")
    parser.add_argument("--tickers",
                        help="comma-separated explicit list; overrides INDEX/--file")
    parser.add_argument("--file", type=Path,
                        help="file of tickers (one per line, or a CSV with a Symbol column)")
    parser.add_argument("--analysts",
                        default="market,social,news,fundamentals",
                        help="comma-separated analyst lineup (default all four)")
    parser.add_argument("--debug", action="store_true", help="stream agent progress")
    parser.add_argument("--resume", action="store_true",
                        help="continue the newest interrupted scan of this index+date")
    args = parser.parse_args(argv)
    if not args.tickers and not args.index:
        parser.error("an index name is required (or pass --tickers)")
    if args.index and args.index not in _INDEX_SOURCES:
        parser.error(f"unknown index '{args.index}'; choose from: {', '.join(_INDEX_SOURCES)}")
    if args.tickers and args.num != 10:
        parser.error("--num does not apply to an explicit --tickers list")
    return args


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    index = args.tickers and "custom" or args.index
    if args.tickers:
        tickers = [_per_share_symbol(t) for t in args.tickers.split(",") if t.strip()]
    else:
        print(f"fetching '{index}' constituents ...", flush=True)
        tickers = get_constituents(index, args.file)
    if not tickers:
        print("no tickers resolved; nothing to do", file=sys.stderr)
        return 1

    scan_list = pick_tickers(index, tickers, args.rank, args.num, Path(DEFAULT_CONFIG["data_cache_dir"]))
    print(f"analyzing {len(scan_list)} tickers: {', '.join(scan_list)}", flush=True)

    results_dir = Path(DEFAULT_CONFIG["results_dir"]) / "index_scan"
    safe_index = safe_ticker_component(index)
    run_dir = None
    if args.resume:
        candidates = sorted(
            d for d in results_dir.glob(f"scan_{safe_index}_{args.date}_*")
            if (d / "opportunities.jsonl").exists()
        )
        if candidates:
            run_dir = candidates[-1]
            print(f"resuming {run_dir.name}", flush=True)
    if run_dir is None:
        run_dir = results_dir / f"scan_{safe_index}_{args.date}_{_run_stamp()}"
        run_dir.mkdir(parents=True, exist_ok=True)

    done = {row["ticker"] for row in _load_rows(run_dir)}
    todo = [t for t in scan_list if t not in done]

    config = {**DEFAULT_CONFIG,
              "results_dir": str(run_dir),
              "memory_log_path": str(run_dir / "trading_memory.md")}
    try:
        graph = TradingAgentsGraph(
            selected_analysts=tuple(a.strip() for a in args.analysts.split(",") if a.strip()),
            debug=args.debug,
            config=config,
        )
    except Exception as exc:
        print(
            f"could not initialize the agent graph: {exc}\n"
            "Check your provider key and model settings in .env "
            "(see .env.example).",
            file=sys.stderr,
        )
        return 1

    started = time.time()
    failures = 0
    for n, ticker in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {ticker}: full agent run starting...", flush=True)
        row = {"ticker": ticker, "index": index, "trade_date": args.date}
        try:
            final_state, signal = graph.propagate(ticker, args.date)
            decision_md = final_state.get("final_trade_decision", "")
            row.update(
                signal=signal,
                exec_summary=_extract_executive_summary(decision_md),
                decision=decision_md,
            )
            graph.save_reports(final_state, ticker)
            print(f"[{n}/{len(todo)}] {ticker}: {signal}", flush=True)
        except Exception as exc:
            failures += 1
            logger.warning("run for %s failed: %s", ticker, exc)
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{n}/{len(todo)}] {ticker}: FAILED ({exc})", flush=True)
        with (run_dir / "opportunities.jsonl").open("a") as log:
            log.write(_report_line(row) + "\n")
        print(f"  elapsed {time.time() - started:.0f}s", flush=True)

    rows = _load_rows(run_dir)
    report = write_report(index, args.date, rows, run_dir)
    print(f"\n{'='*60}\n{report}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
