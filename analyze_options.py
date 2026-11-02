"""Answer a question about one asset's option market with the agent team.

Runs the TradingAgents pipeline — now including an Options Analyst that pulls
live option chains and reads the structure across expiries and strikes (ATM
level, OTM/ITM wings, IV smile and skew, the ATM-IV term structure, volume
and open-interest flows, put/call ratios, max pain, approximate dealer gamma)
— over a single asset, and writes a focused markdown report that answers
your question, next to the full per-agent report tree the framework saves.

Usage (from the repo root, with the venv python):

    python analyze_options.py NVDA --question "Are the OTM puts rich into earnings?"
    python analyze_options.py GOLD --question "Where is max pain for the December expiry?"
    python analyze_options.py BTC   --question "What does the put/call skew imply for the next month?"
    python analyze_options.py SPY   --analysts options,market,news,fundamentals
    python analyze_options.py AAPL --expiry 2026-12-18 --question "..."
    python analyze_options.py NVDA --peek          # the deterministic data snapshot, no agents

Assets: a stock/ETF ticker (NVDA, TSLA, SPY, ...), GOLD / XAUUSD, or BTC.
Yahoo lists no option chains for spot crypto, futures or indexes, so those
assets are quoted through a listed proxy underlying — IBIT for BTC, GLD for
gold, SPY for ^GSPC — and every report says which underlying it actually
quoted. Override with TRADINGAGENTS_OPTION_UNDERLYING in .env or --underlying.

Model/provider configuration comes from the environment exactly as for any
other run (provider key in .env; models via TRADINGAGENTS_LLM_PROVIDER /
TRADINGAGENTS_DEEP_THINK_LLM / TRADINGAGENTS_QUICK_THINK_LLM). No key: the
script fails fast with a pointer, before any analysis is paid for.

Cost warning: every question is a complete multi-agent run (analysts, the
bull/bear debate, the trader, the risk debate, the portfolio manager) —
expect minutes and a nontrivial API spend. ``--analysts options`` is the
cheapest lineup that still answers; ``--peek`` is free.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.symbols import crypto_base, normalize_symbol, safe_ticker_component
from tradingagents.dataflows.vendors.yahoo.options import (
    OPTIONABLE_PROXIES,
    get_option_expirations_yfinance,
    get_option_term_structure_yfinance,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger("analyze_options")

# What a bare 'python analyze_options.py <ASSET>' asks when no --question is given.
_DEFAULT_QUESTION = (
    "Analyze this asset's option market: the ATM level, the ITM and OTM wings "
    "(IV, volume and open interest per band), the volatility skew, the ATM-IV "
    "term structure across expiries, the put/call positioning, max pain and the "
    "approximate gamma exposure, and what the whole picture implies for the "
    "near-term path of the asset. Tag every level ITM / ATM / OTM."
)

# Underlying quote types the futures aliases resolve to (the *_=F Yahoo symbols).
_FUTURES_SUFFIX = "=F"

# Metals/energy/index aliases worth labeling as non-equity asset classes
# beyond the futures suffix; the stock toolset serves any of them.
_COMMODITY_CANONICALS = frozenset(OPTIONABLE_PROXIES) - {"^GSPC", "^NDX", "^DJI", "BTC-USD", "ETH-USD"}


def detect_asset_type(canonical: str) -> str:
    """The pipeline flavor for an asset: crypto, commodity, or stock.

    Spot crypto sets the crypto pipeline (its prompts treat the instrument as
    an asset, not a company); a metals/energy future canonical (or any name
    that resolves to one) sets the commodity flavor; everything else — regular
    equities and the tradable ETF proxies — runs the stock pipeline.
    """
    if crypto_base(canonical) or canonical.endswith("-USD"):
        return "crypto"
    if canonical.endswith(_FUTURES_SUFFIX) or canonical in _COMMODITY_CANONICALS:
        return "commodity"
    return "stock"


def _preflight_data(canonical: str, config: dict, curr_date: str) -> str:
    """The expiration list the option tools will serve, fetched before any LLM spend.

    Returns the vendor's report text. A hold notice (historical run) or an
    absence (no listed chains anywhere) raises SystemExit with the reason:
    an options question without option data is not a run worth paying for.
    """
    try:
        with run_config(config):
            text = get_option_expirations_yfinance(canonical, curr_date)
    except Exception as exc:  # noqa: BLE001 - one message for every data failure
        proxies = OPTIONABLE_PROXIES.get(canonical, ())
        hint = (f" Known listed proxies for this asset class: {', '.join(proxies)} — "
                "force one with --underlying or TRADINGAGENTS_OPTION_UNDERLYING in .env."
                if proxies else
                " If you know a listed proxy (an ETF the asset tracks), force it with "
                "--underlying or TRADINGAGENTS_OPTION_UNDERLYING in .env.")
        print(f"no option data for {canonical}: {exc}{hint}", file=sys.stderr)
        raise SystemExit(1) from exc
    if text.startswith("OPTION DATA WITHHELD"):
        print(text, file=sys.stderr)
        raise SystemExit(1)
    return text


def _proxy_note(expirations_text: str) -> str:
    """Which underlying actually serves the chains, from the pre-flight text."""
    match = re.search(r"chains quoted on ([^\s\)]+)", expirations_text)
    return f"option data quoted on {match.group(1)}" if match else "option data quoted directly on the asset"


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _summary_path(run_dir: Path, canonical: str, trade_date: str) -> Path:
    return run_dir / f"options_analysis_{safe_ticker_component(canonical)}_{trade_date}.md"


def write_summary(run_dir: Path, canonical: str, trade_date: str, question: str,
                  proxy_note: str, data_note: str, final_state: dict, signal: str,
                  tree_path: Path) -> Path:
    """The focused headline report: the question, the options team's findings, the call."""
    options_report = (final_state.get("options_report") or "").strip()
    decision = (final_state.get("final_trade_decision") or "").strip()
    lines = [
        f"# Options analysis — {canonical} — {trade_date}",
        "",
        f"- **Question**: {question}",
        f"- **Rating**: {signal}",
        f"- **Option data**: {proxy_note}",
        f"- **Full agent report tree**: {tree_path}",
        "",
    ]
    lines += ["## Options market data (pre-flight snapshot)", "", data_note, ""]
    if options_report:
        lines += ["## Options Analyst's report", "", options_report, ""]
    else:
        lines += [
            "## Options Analyst's report",
            "",
            "_The options analyst returned no report — its data was unavailable or its "
            "run failed mid-way. The full report tree under the path above holds what "
            "the rest of the team produced._",
            "",
        ]
    if decision:
        lines += ["## Final decision (Portfolio Manager)", "", decision, ""]
    report = _summary_path(run_dir, canonical, trade_date)
    report.write_text("\n".join(lines) + "\n")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the agent team a question about one asset's option market.",
    )
    parser.add_argument("asset", help="the asset to analyze: a ticker (NVDA, TSLA, SPY), GOLD/XAUUSD, or BTC")
    parser.add_argument("--question", default="",
                        help="what to ask the option market; a standard full-structure question by default")
    parser.add_argument("--expiry",
                        help="an expiration date to focus the analysis on, YYYY-mm-dd (optional)")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="trade date, YYYY-MM-DD, no later than today (option data only exists for today)")
    parser.add_argument("--analysts", default="options,market,news",
                        help="comma-separated analyst lineup; 'options' should be kept "
                             "(default options,market,news)")
    parser.add_argument("--underlying",
                        help="force the optionable underlying whose chains are quoted "
                             "(e.g. GLD for gold, IBIT for BTC)")
    parser.add_argument("--peek", action="store_true",
                        help="print the deterministic option-data snapshot and exit (no agents, no API spend)")
    parser.add_argument("--debug", action="store_true", help="stream agent progress")
    args = parser.parse_args(argv)
    if not args.asset.strip():
        parser.error("an asset is required (a ticker, GOLD/XAUUSD, or BTC)")
    lineup = [a.strip() for a in args.analysts.split(",") if a.strip()]
    unknown = [a for a in lineup if a not in ANALYST_NODE_SPECS]
    if unknown:
        parser.error(
            "unknown analyst key(s) " + ", ".join(unknown)
            + "; available: " + ", ".join(ANALYST_NODE_SPECS)
        )
    if "options" not in lineup:
        parser.error("this script answers option questions; keep 'options' in --analysts")
    if args.expiry:
        try:
            datetime.strptime(args.expiry, "%Y-%m-%d")
        except ValueError:
            parser.error("--expiry must be YYYY-mm-dd")
    return args


def main(argv=None) -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    canonical = normalize_symbol(args.asset.strip())
    asset_type = detect_asset_type(canonical)

    config = {**DEFAULT_CONFIG}
    if args.underlying:
        config["option_underlying"] = args.underlying

    question = args.question.strip() or _DEFAULT_QUESTION
    if args.expiry:
        question += f" Focus the analysis especially on the expiry {args.expiry}."

    if args.peek:
        try:
            config["results_dir"] = str(Path(DEFAULT_CONFIG["results_dir"]))
            with run_config(config):
                print(get_option_expirations_yfinance(canonical, args.date))
                print("\n" + "-" * 60 + "\n")
                print(get_option_term_structure_yfinance(canonical, args.date))
        except Exception as exc:
            print(f"option data for {canonical} unavailable: {exc}", file=sys.stderr)
            return 1
        return 0

    # Pre-flight: prove option data exists before a single token is spent, and
    # keep the expirations snapshot for the summary report.
    expirations_text = _preflight_data(canonical, config, args.date)
    proxy_note = _proxy_note(expirations_text)

    results_dir = Path(DEFAULT_CONFIG["results_dir"]) / "options_analysis"
    run_dir = results_dir / f"opts_{safe_ticker_component(canonical)}_{args.date}_{_run_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config.update({
        "results_dir": str(run_dir),
        "memory_log_path": str(run_dir / "trading_memory.md"),
    })

    print(f"asset: {canonical} ({asset_type}) · underlying for chains: {proxy_note}")
    print(f"question: {question}\n", flush=True)

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
    print("agents starting...", flush=True)
    try:
        final_state, signal = graph.propagate(canonical, args.date, asset_type=asset_type,
                                              question=question)
    except Exception as exc:
        logger.error("run for %s failed: %s", canonical, exc)
        print(f"agent run FAILED ({type(exc).__name__}: {exc}); "
              f"partial state is lost for this graph", file=sys.stderr)
        return 2

    tree_path = graph.save_reports(final_state, canonical)
    summary = write_summary(
        run_dir, canonical, args.date, question, proxy_note, expirations_text,
        final_state, signal, tree_path,
    )

    print(f"\n{'=' * 60}")
    print(f"rating: {signal}")
    print(f"elapsed {time.time() - started:.0f}s")
    print(f"summary report: {summary}")
    print(f"full agent reports: {tree_path}")
    options_report = (final_state.get("options_report") or "").strip()
    if not options_report:
        print("WARNING: the options analyst produced no report (see the summary file).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
