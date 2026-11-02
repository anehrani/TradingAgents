"""Unit tests for the option-market data layer and its analyst plumbing (no network)."""

import math

import pandas as pd
import pytest

from tradingagents.dataflows.router import VENDOR_METHODS, get_category_for_method
from tradingagents.dataflows.vendors.yahoo import options as opt
from tradingagents.dataflows.vendors.yahoo.options import (
    MAX_CHAIN_ROWS_PER_SIDE,
    OPTIONABLE_PROXIES,
    approximate_delta,
    approximate_gamma,
    chain_metrics,
    get_option_chain_yfinance,
    get_option_expirations_yfinance,
    max_pain_strike,
    tag_moneyness,
)
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS, build_analyst_execution_plan
from tradingagents.graph.propagation import Propagator

SPOT = 105.0


def _call_row(strike, volume=0, oi=0, iv=0.30, bid=None, ask=None, last=None):
    return {"strike": strike, "volume": volume, "openInterest": oi,
            "impliedVolatility": iv, "bid": bid, "ask": ask, "lastPrice": last}


@pytest.fixture
def chain_frames():
    calls = pd.DataFrame([
        _call_row(100.0, volume=10, oi=500, iv=0.30, bid=5.1, ask=5.3, last=5.2),
        _call_row(105.0, oi=200, iv=0.28, bid=1.9, ask=2.1, last=2.0),
        _call_row(120.0, volume=40, oi=900, iv=0.22, bid=0.1, ask=0.2, last=0.15),
    ])
    puts = pd.DataFrame([
        _call_row(90.0, volume=5, oi=300, iv=0.34, bid=0.4, ask=0.6, last=0.5),
        _call_row(100.0, volume=25, oi=700, iv=0.32, bid=2.0, ask=2.2, last=2.1),
        _call_row(105.0, volume=60, oi=1000, iv=0.36, bid=4.4, ask=4.6, last=4.5),
    ])
    return calls, puts


def test_moneyness_tags():
    assert tag_moneyness("call", 100.0, SPOT) == "ITM"
    assert tag_moneyness("call", 120.0, SPOT) == "OTM"
    assert tag_moneyness("call", 105.0 * 1.009, SPOT) == "ATM"
    assert tag_moneyness("put", 110.0, SPOT) == "ITM"
    assert tag_moneyness("put", 90.0, SPOT) == "OTM"
    assert tag_moneyness("put", SPOT, SPOT) == "ATM"


def test_greek_approximations_sane():
    t_years = 30 / 365.25
    call = approximate_delta("call", SPOT, SPOT, 0.30, t_years)
    put = approximate_delta("put", SPOT, SPOT, 0.30, t_years)
    assert math.isclose(call, 1 + put, abs_tol=1e-9)
    assert 0.35 < call < 0.65          # an ATM call delta hovers around 0.5
    assert approximate_gamma(SPOT, SPOT, 0.30, t_years) > 0
    assert approximate_delta("call", SPOT, SPOT, 0.0, t_years) is None


def test_max_pain_strike():
    # 800 open calls at 100 pay only above 100; puts at 110/120 pay only below.
    pain = max_pain_strike([100.0, 110.0, 120.0, 130.0], {100.0: 800}, {110.0: 300, 120.0: 100})
    assert pain == 100.0
    assert max_pain_strike([100.0, 110.0], {100.0: 0}, {110.0: 0}) is None


def test_chain_metrics_readings(chain_frames):
    calls, puts = chain_frames
    m = chain_metrics(SPOT, "2026-12-18", 47, calls, puts)
    assert m["atm_strike"] == SPOT
    assert m["call_volume"] == 50 and m["put_volume"] == 90
    assert m["call_open_interest"] == 1600 and m["put_open_interest"] == 2000
    assert m["put_call_volume"] == pytest.approx(1.8)
    assert m["put_call_oi"] == pytest.approx(1.25)
    assert m["max_pain"] == SPOT
    assert m["top_put_oi"][0] == (105.0, 1000.0)
    assert m["net_gamma"] < 0          # the put wing carries the OI


def test_chain_metrics_survives_thin_chains():
    thin = pd.DataFrame([_call_row(100.0, volume=float("nan"), oi=float("nan"), iv=float("nan"))])
    m = chain_metrics(SPOT, "2026-12-18", 47, thin, thin.iloc[0:0].copy())
    assert m["call_volume"] == 0 and m["put_open_interest"] == 0
    assert m["put_call_volume"] is None and m["max_pain"] is None


def test_moneyness_filtering_and_row_cap(chain_frames):
    calls, _ = chain_frames
    otm = opt._rows_for_table(calls, "call", SPOT, 30 / 365.25, "otm")
    assert [r["money"] for r in otm] == ["OTM"] and otm[0]["strike"] == 120.0

    ladder = pd.DataFrame([_call_row(105.0 + i) for i in range(1, MAX_CHAIN_ROWS_PER_SIDE + 10)])
    capped = opt._rows_for_table(ladder, "call", SPOT, 30 / 365.25, "all")
    assert len(capped) == MAX_CHAIN_ROWS_PER_SIDE
    # The cap keeps the strikes nearest the spot: 106..305 stay, 306..315 go.
    assert capped[0]["strike"] == 106.0 and max(r["strike"] for r in capped) == 305.0


def test_historical_runs_are_refused():
    words = get_option_expirations_yfinance("AAPL", "2020-01-01")
    assert words.startswith("OPTION DATA WITHHELD")
    assert get_option_chain_yfinance("AAPL", "2026-12-18", "2020-01-01").startswith(
        "OPTION DATA WITHHELD")


def test_router_wiring():
    for method in ("get_option_expirations", "get_option_term_structure", "get_option_chain"):
        assert get_category_for_method(method) == "option_data"
        assert "yfinance" in VENDOR_METHODS[method]


def test_options_analyst_registered():
    spec = ANALYST_NODE_SPECS["options"]
    assert spec.agent_node == "Options Analyst"
    assert spec.report_key == "options_report"
    assert spec.tools
    plan = build_analyst_execution_plan(["options"])
    assert [s.key for s in plan.specs] == ["options"]


def test_optionable_proxies_cover_gold_and_btc():
    assert OPTIONABLE_PROXIES["BTC-USD"][0] == "IBIT"
    assert OPTIONABLE_PROXIES["GC=F"][0] == "GLD"


def test_question_reaches_the_analysts():
    state = Propagator().create_initial_state("BTC-USD", "2026-11-01", question="What do OTM puts say?")
    first = state["messages"][0]
    assert first[0] == "human" and "What do OTM puts say?" in first[1]
    assert "options_report" in state
    plain = Propagator().create_initial_state("AAPL", "2026-11-01")
    assert plain["messages"][0][1] == "AAPL"


def test_proxy_note_extracts_the_quoted_underlying():
    from analyze_options import _proxy_note

    # A parenthesized header note must not swallow the ')' into the ticker.
    assert _proxy_note("## Option expirations for GC=F (chains quoted on GLD)") == (
        "option data quoted on GLD"
    )
    assert _proxy_note("## Option expirations for SPY") == (
        "option data quoted directly on the asset"
    )
    assert _proxy_note("## Gold term structure — chains quoted on GLD") == (
        "option data quoted on GLD"
    )
