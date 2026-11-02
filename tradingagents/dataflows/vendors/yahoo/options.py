"""yfinance-based option-market data: expirations, chains, and cross-expiry structure.

Option chains exist on Yahoo Finance only as a live, current snapshot with no
historical vintage, so every entry point refuses a run dated in the past —
serving today's chain into that run would leak future positioning data, the
same live-profile rule the fundamentals vendors follow.

Instruments without a Yahoo-listed chain — spot crypto (BTC-USD), the metals
futures an alias like GOLD resolves to (GC=F), and index symbols (^GSPC) — are
answered through a listed proxy underlying (IBIT, GLD, SPY, ...), with the
substitution named in the output so a report can say what it actually quoted.

The analytics (moneyness tagging, greek approximations, put/call ratios, max
pain, gamma exposure) are pure functions of a chain's DataFrames, kept free of
yfinance so the test suite can exercise them without network.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import get_current_date, is_historical
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry

logger = logging.getLogger(__name__)

# Within this fraction of the current spot a strike reads as at-the-money: a
# liquid underlying moves about this much in a day, so the band is where the
# trades that settle the chain live today.
ATM_BAND = 0.01

# Strikes per side shown in a chain table before it is capped to those nearest
# the spot: far expiries on liquid underlyings list hundreds of strikes, and a
# table of all of them costs the reading agent more than it tells it.
MAX_CHAIN_ROWS_PER_SIDE = 200

# The convention where moneyness skew is usually quoted: the ~25-delta wings.
_SKEW_TARGET_DELTA = 0.25
_SKEW_DELTA_WINDOW = 0.12

# Underlyings whose asset class has no Yahoo option feed, -> listed proxies to
# quote instead, most preferred first. Extend by adding a row: call sites
# resolve through here, never by hand.
OPTIONABLE_PROXIES: dict[str, tuple[str, ...]] = {
    # Spot crypto: Yahoo lists no chains; the spot-ETF proxies do.
    "BTC-USD": ("IBIT", "BITB", "COIN", "MSTR"),
    "ETH-USD": ("ETHA", "ETHE", "COIN"),
    # Metals/energy futures: the linked-ETF chains are listed.
    "GC=F": ("GLD", "IAU", "GLDM"),
    "SI=F": ("SLV",),
    "CL=F": ("USO",),
    "NG=F": ("UNG",),
    "HG=F": ("CPER",),
    # Index symbols -> their tradable ETFs.
    "^GSPC": ("SPY", "IVV", "VOO"),
    "^NDX": ("QQQ",),
    "^DJI": ("DIA",),
}


# ---- formatting -----------------------------------------------------------------


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(value)
    except TypeError:
        return False


def _strike_text(strike) -> str:
    """A strike as printed by the exchange: whole, unless quoted with halves."""
    if not _finite(strike):
        return "-"
    return f"{strike:.2f}" if strike != int(strike) else f"{int(strike)}"


def _num_text(value, pattern: str) -> str:
    """A price cell in ``pattern`` form; a value Yahoo left unset prints as a dash."""
    return pattern % value if _finite(value) else "-"


def _int_text(value) -> str:
    return f"{value:,.0f}" if _finite(value) else "-"


def _iv_text(iv) -> str:
    """An implied volatility as a percent; zero stands for unset on Yahoo."""
    return f"{iv * 100:.1f}" if _finite(iv) and iv > 0.0005 else "-"


def _delta_text(delta) -> str:
    return f"{delta:+.2f}" if _finite(delta) else "-"


def _ratio_text(ratio) -> str:
    return f"{ratio:.2f}" if _finite(ratio) and ratio > 0 else "-"


def _gex_text(gex) -> str:
    """A net gamma-exposure figure in $ millions, signed by dealer direction."""
    if not _finite(gex):
        return "-"
    return f"{'+' if gex >= 0 else '-'}${abs(gex) / 1e6:.1f}m"


# ---- analytics on a chain -------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Standard-normal cdf from math.erf: the greek math needs no scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def tag_moneyness(kind: str, strike: float, spot: float, band: float = ATM_BAND) -> str:
    """``ITM`` / ``ATM`` / ``OTM`` for one contract against the spot.

    A strike within ``band`` of the spot reads as ATM whichever side it sits
    on; beyond that band calls are in the money below the spot and puts above
    it. Unusable inputs (never a real chain row) read as ``?``.
    """
    if kind not in ("call", "put") or not (_finite(strike) and _finite(spot)) or spot <= 0:
        return "?"
    mny = (strike - spot) / spot
    if abs(mny) <= band:
        return "ATM"
    itm = strike < spot if kind == "call" else strike > spot
    return "ITM" if itm else "OTM"


def approximate_delta(kind: str, strike: float, spot: float, iv: float,
                      t_years: float) -> float | None:
    """Black-Scholes delta at zero rates, from the chain's own IV.

    An approximation on purpose: the chain carries no rates or dividend
    yield, and the reports say so. None when a value cannot support it.
    """
    if not (_finite(strike) and _finite(spot) and _finite(iv)) or iv <= 0:
        return None
    if spot <= 0 or strike <= 0 or t_years <= 0:
        return None
    sigma = min(iv, 5.0)
    try:
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / (sigma * math.sqrt(t_years))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    call_delta = _normal_cdf(d1)
    return call_delta if kind == "call" else call_delta - 1.0


def approximate_gamma(spot: float, strike: float, iv: float, t_years: float) -> float | None:
    """Black-Scholes gamma at zero rates, from the chain's own IV."""
    if not (_finite(strike) and _finite(spot) and _finite(iv)) or iv <= 0:
        return None
    if spot <= 0 or strike <= 0 or t_years <= 0:
        return None
    sigma = min(iv, 5.0)
    try:
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / (sigma * math.sqrt(t_years))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return _normal_pdf(d1) / (spot * sigma * math.sqrt(t_years))


def max_pain_strike(strikes, call_open_interest: dict, put_open_interest: dict) -> float | None:
    """The strike whose expiry price pays holders of open positions the least.

    At every candidate strike the total settlement payout over all open
    contracts is summed; the payout-minimising strike is max pain, the level
    dealer hedging is said to pin prices toward. None when there is no OI.
    """
    open_strikes = sorted({
        k for k in strikes if call_open_interest.get(k) or put_open_interest.get(k)
    })
    if not open_strikes:
        return None
    best, best_pain = None, None
    for candidate in open_strikes:
        pain = (
            sum(oi * (candidate - k) for k, oi in call_open_interest.items() if candidate > k)
            + sum(oi * (k - candidate) for k, oi in put_open_interest.items() if candidate < k)
        )
        if best_pain is None or pain < best_pain:
            best, best_pain = candidate, pain
    return best


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    """One chain column as coerced floats, zeros when Yahoo left the column out."""
    if name not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def _rows_for_table(df: pd.DataFrame, kind: str, spot: float, t_years: float,
                    money: str) -> list[dict]:
    """One side of the chain as display rows, filtered to a moneyness view.

    Rows beyond the cap keep the ones nearest the spot — the ones an
    OTM/ATM/ITM question is about. Sorted by strike ascending.
    """
    strikes = _numeric(df, "strike")
    volumes = _numeric(df, "volume")
    open_interest = _numeric(df, "openInterest")
    ivs = _numeric(df, "impliedVolatility")
    bids = _numeric(df, "bid")
    asks = _numeric(df, "ask")
    lasts = _numeric(df, "lastPrice")

    rows = []
    for i in df.index:
        strike = float(strikes[i])
        if strike <= 0:
            continue
        iv = float(ivs[i])
        delta = approximate_delta(kind, strike, spot, iv, t_years) if iv > 0 else None
        rows.append({
            "strike": strike,
            "money": tag_moneyness(kind, strike, spot),
            "iv": iv,
            "volume": float(volumes[i]),
            "open_interest": float(open_interest[i]),
            "bid": float(bids[i]),
            "ask": float(asks[i]),
            "last": float(lasts[i]),
            "delta": delta,
        })
    if money != "all":
        rows = [r for r in rows if r["money"].lower() == money]
    if len(rows) > MAX_CHAIN_ROWS_PER_SIDE:
        closest = sorted(rows, key=lambda r: abs(r["strike"] - spot))[:MAX_CHAIN_ROWS_PER_SIDE]
        rows = closest
    return sorted(rows, key=lambda r: r["strike"])


def _iv_nearest_delta(side_rows: list[dict], target: float) -> tuple[float, float] | None:
    """The (IV, strike) of the row nearest ``target`` |delta| in the wing, or None."""
    wing = [r for r in side_rows
            if r["delta"] is not None and abs(abs(r["delta"]) - target) <= _SKEW_DELTA_WINDOW]
    if not wing:
        return None
    best = min(wing, key=lambda r: abs(abs(r["delta"]) - target))
    return best["iv"], best["strike"]


def chain_metrics(spot: float, expiry: str, dte_days: int,
                  calls: pd.DataFrame, puts: pd.DataFrame) -> dict:
    """The per-expiry readings every option report is built from, from one chain."""
    t_years = max(dte_days, 1) / 365.25
    call_rows = _rows_for_table(calls, "call", spot, t_years, "all")
    put_rows = _rows_for_table(puts, "put", spot, t_years, "all")

    strikes = sorted({r["strike"] for r in call_rows + put_rows})
    atm_strike = min(strikes, key=lambda k: abs(k - spot)) if strikes else None

    call_oi = {r["strike"]: r["open_interest"] for r in call_rows if r["open_interest"] > 0}
    put_oi = {r["strike"]: r["open_interest"] for r in put_rows if r["open_interest"] > 0}

    def atm(side_rows: list[dict]) -> dict | None:
        if atm_strike is None:
            return None
        at_strike = [r for r in side_rows if r["strike"] == atm_strike]
        return at_strike[0] if at_strike else None

    call_vol = sum(r["volume"] for r in call_rows)
    put_vol = sum(r["volume"] for r in put_rows)
    call_oi_total = sum(r["open_interest"] for r in call_rows)
    put_oi_total = sum(r["open_interest"] for r in put_rows)

    def signed_gamma(side_rows: list[dict], sign: int) -> float:
        """Dollar gamma with the dealer assumption: short calls, long puts."""
        total = 0.0
        for r in side_rows:
            gamma = approximate_gamma(spot, r["strike"], r["iv"], t_years) if r["iv"] > 0 else None
            if gamma is not None:
                total += sign * gamma * r["open_interest"] * 100 * spot
        return total

    call_skew = _iv_nearest_delta(call_rows, _SKEW_TARGET_DELTA)
    put_skew = _iv_nearest_delta(put_rows, _SKEW_TARGET_DELTA)
    skew_iv_gap = put_skew[0] - call_skew[0] if call_skew and put_skew else None

    def top_oi(rows: list[dict], count: int = 2) -> list[tuple[float, float]]:
        ranked = sorted((r for r in rows if r["open_interest"] > 0),
                        key=lambda r: -r["open_interest"])[:count]
        return [(r["strike"], r["open_interest"]) for r in ranked]

    return {
        "expiry": expiry,
        "dte_days": dte_days,
        "spot": spot,
        "atm_strike": atm_strike,
        "atm_call": atm(call_rows),
        "atm_put": atm(put_rows),
        "call_volume": call_vol,
        "put_volume": put_vol,
        "call_open_interest": call_oi_total,
        "put_open_interest": put_oi_total,
        "put_call_volume": put_vol / call_vol if call_vol > 0 else None,
        "put_call_oi": put_oi_total / call_oi_total if call_oi_total > 0 else None,
        "skew_call": call_skew,
        "skew_put": put_skew,
        "skew_iv_gap": skew_iv_gap,
        "max_pain": max_pain_strike(strikes, call_oi, put_oi),
        "net_gamma": signed_gamma(call_rows, 1) + signed_gamma(put_rows, -1),
        "top_call_oi": top_oi(call_rows),
        "top_put_oi": top_oi(put_rows),
    }


# ---- live Yahoo access -----------------------------------------------------------


def _quote_spot(t, canonical: str) -> float:
    """The underlying's last price: quote feed first, recent closes as fallback."""
    for key in ("lastPrice", "previous_close", "regular_market_previous_close", "open"):
        try:
            value = yf_retry(lambda key=key: t.fast_info.get(key))
        except VendorUnavailableError:
            continue
        if value is not None and math.isfinite(float(value)) and float(value) > 0:
            return float(value)

    history = yf_retry(lambda: t.history(period="5d"))
    if history is not None and len(history):
        return float(history["Close"].iloc[-1])
    raise NoMarketDataError(canonical, canonical, "a current spot price could not be quoted")


def _quote_underlying(canonical: str) -> tuple:
    """Live spot and expiration list of one underlying; chains must exist."""
    import yfinance as yf

    t = yf.Ticker(canonical)
    expiries = tuple(yf_retry(lambda: t.options) or ())
    if not expiries:
        raise_for_empty(canonical, canonical, "listed option chains")
    return t, _quote_spot(t, canonical), expiries


def _override() -> str | None:
    """A caller-forced underlying (the ``option_underlying`` config), when set."""
    return (get_config() or {}).get("option_underlying") or None


def _chain_source(symbol: str) -> tuple:
    """The underlying to quote chains through: the symbol itself, else a proxy.

    The caller-forced ``option_underlying`` is probed first, then the symbol,
    then its class's known proxies (``OPTIONABLE_PROXIES``). Returns
    ``(ticker, spot, expiries, proxy_note_or_None, used_symbol)``; raises
    NoMarketDataError naming everything tried when none lists chains.
    """
    canonical = normalize_symbol(symbol)
    forced = _override()
    candidates = (
        ((forced,) if forced else ())
        + ((canonical,) if not forced else ())
        + (() if forced else OPTIONABLE_PROXIES.get(canonical, ()))
    )

    last_error: NoMarketDataError | None = None
    for candidate in candidates:
        try:
            t, spot, expiries = _quote_underlying(normalize_symbol(candidate))
        except NoMarketDataError as exc:
            last_error = exc
            logger.info("no option chains for %r (%s)", candidate, exc)
            continue
        note = None if candidate == canonical else (
            f"chains are quoted on proxy underlying {candidate}: Yahoo lists no option chains "
            f"for {canonical} — spot crypto, index and futures underlyings are outside "
            "Yahoo's option feed"
        )
        return t, spot, expiries, note, candidate
    raise NoMarketDataError(
        symbol, canonical,
        f"no listed option chains on Yahoo for {', '.join(candidates) or canonical}'s class",
    ) from last_error


def _load_chain(t, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One expiry's call and put frames; Yahoo's silence becomes a typed absence."""
    chain = yf_retry(lambda: t.option_chain(expiry))
    if chain is None:
        raise NoMarketDataError(expiry, expiry, "the chain response came back empty")
    return chain.calls, chain.puts


def _live_only_notice(symbol: str, as_of_date: str | None) -> str | None:
    """A withhold notice for a past-dated run: option chains cannot be time-travelled.

    Yahoo lists no historical chains, so the feed is a snapshot of the market
    now; a run dated earlier would read positioning from after its decision.
    """
    if not is_historical(as_of_date):
        return None
    return (
        f"OPTION DATA WITHHELD for run date {as_of_date}: Yahoo Finance lists option chains "
        f"only as a live, current snapshot with no historical vintage, so today's chain cannot "
        f"be served into a {as_of_date} analysis without leaking future positioning data. "
        f"Re-run with today's date ({get_current_date()}) to analyze {symbol}'s option market."
    )


def _raise_absent(symbol: str, expiry: str, listed: tuple[str, ...]) -> NoMarketDataError:
    nearby = ", ".join(listed[:6]) + (", ..." if len(listed) > 6 else "")
    return NoMarketDataError(
        symbol, expiry, f"no {expiry} chain is listed; the nearest listed expiries are {nearby}",
    )


def _data_note() -> str:
    """The caveats every option report carries, aimed at the reading agent."""
    return (
        "Data notes: live Yahoo Finance snapshot. Volume counts contracts traded today; "
        "open interest is exchange-reported and updates overnight; both are in contracts of "
        "100 shares. IV, delta and gamma are approximated from the chain's own implied "
        "volatilities at a zero interest rate, with no dividend or early-exercise model — "
        "treat them as structural guides, not broker-grade greeks. '-' means Yahoo left "
        "the field unset, not that it is zero."
    )


# ---- public vendor entry points ---------------------------------------------------


def get_option_expirations_yfinance(symbol: str, curr_date: str | None = None) -> str:
    """List the expiration dates with quoted chains for ``symbol`` on Yahoo.

    An instrument without chains is answered through a listed proxy (IBIT for
    BTC-USD, GLD for gold, ...), named in the output. Raises NoMarketDataError
    when neither the symbol nor a proxy has chains.
    """
    if notice := _live_only_notice(symbol, curr_date):
        return notice

    t, spot, expiries, note, used = _chain_source(symbol)
    as_of = curr_date or get_current_date()
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")

    lines = [
        f"## Option expirations for {symbol}"
        + (f" (chains quoted on {used})" if note else ""),
        "",
        f"Spot: {_strike_text(spot)} · live Yahoo snapshot as of {as_of} · "
        f"{len(expiries)} expiration dates listed",
    ]
    if note:
        lines.append(f"Note: {note}")
    shown = expiries[:30]
    if shown:
        lines += ["", "| Expiry | Days to expiry |", "|---|---|"]
        for expiry in shown:
            dte = (datetime.strptime(expiry, "%Y-%m-%d") - as_of_dt).days
            lines.append(f"| {expiry} | {dte} |")
        if len(expiries) > len(shown):
            lines.append(f"…and {len(expiries) - len(shown)} further dates")
    lines += [
        "",
        "Call get_option_term_structure for the cross-expiry summary, then get_option_chain "
        "with one expiry (and its side/moneyness filters) for the strike-level detail. "
        "An expiry within a few days is the front month; a far expiry is slow position, "
        "and an earnings-adjacent expiry carries the IV crush risk (stocks).",
        _data_note(),
    ]
    return "\n".join(lines)


def get_option_term_structure_yfinance(symbol: str, curr_date: str | None = None,
                                       max_expiries: int = 8) -> str:
    """The cross-expiry option market: flow, volatility, and positioning per expiry.

    One row per listed expiration (the nearest ``max_expiries``), each with
    volume and open-interest put/call ratios, at-the-money IV, the 25-delta
    skew, max pain, and an approximate net gamma exposure. This is the broad
    search across the option market that a question usually starts from.
    """
    if notice := _live_only_notice(symbol, curr_date):
        return notice

    t, spot, expiries, note, used = _chain_source(symbol)
    as_of = curr_date or get_current_date()
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    picks = list(expiries[: max(1, max_expiries)])

    rows = []
    for expiry in picks:
        dte = max((datetime.strptime(expiry, "%Y-%m-%d") - as_of_dt).days, 0)
        calls, puts = _load_chain(t, expiry)
        rows.append(chain_metrics(spot, expiry, dte, calls, puts))

    lines = [
        f"## Option term structure for {symbol}"
        + (f" (chains quoted on {used})" if note else ""),
        "",
        f"Spot {_strike_text(spot)} · live Yahoo snapshot as of {as_of} · "
        f"{len(rows)} of {len(expiries)} expiries shown",
    ]
    if note:
        lines.append(f"Note: {note}")

    lines += [
        "",
        "| Expiry | DTE | Call vol | Put vol | P/C vol | P/C OI | ATM IV | 25d skew | "
        "Max pain | Net GEX | Top OI (C / P) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        lines.append(
            f"| {m['expiry']} | {m['dte_days']} | {_int_text(m['call_volume'])} "
            f"| {_int_text(m['put_volume'])} | {_ratio_text(m['put_call_volume'])} "
            f"| {_ratio_text(m['put_call_oi'])} "
            f"| {_iv_text(_atm_iv(m))} | {_skew_text(m['skew_iv_gap'])} "
            f"| {_strike_text(m['max_pain'])} | {_gex_text(m['net_gamma'])} "
            f"| {_oi_label(m['top_call_oi'])} / {_oi_label(m['top_put_oi'])} |"
        )

    known = [iv for m in rows if (iv := _atm_iv(m)) is not None]
    shape = "flat curve"
    if len(known) >= 2:
        shape = ("contango — near-dated IV below far-dated (dormant market, event risk ahead)"
                 if known[0] < known[-1] else
                 "backwardation — near-dated IV above far-dated (stress is now)")
    lines += [
        "",
        f"The ATM IV term curve reads as: {shape}. Reading keys: P/C is the put-to-call ratio "
        "by volume and by open interest (above 1 = puts dominate); 25d skew is OTM put IV minus "
        "OTM call IV in volatility points (positive = put protection is bid); max pain is the "
        "expiry price that would pay open positions least; net GEX is the approximated dealer "
        "gamma exposure in $ millions per 1-point move; Top OI lists the strikes with the "
        "largest open interest per side.",
        "",
        "Call get_option_chain with one expiry (and side/moneyness filters) for the "
        "strike-level table and rows that back this summary.",
        _data_note(),
    ]
    return "\n".join(lines)


def get_option_chain_yfinance(symbol: str, expiry: str, curr_date: str | None = None,
                              side: str = "both", moneyness: str = "all") -> str:
    """One expiration's chain: strike-level quotes tagged ITM / ATM / OTM.

    ``side`` is ``calls`` | ``puts`` | ``both``; ``moneyness`` narrows rows to
    ``itm`` | ``atm`` | ``otm``. Chains carry every listed strike, so filter
    far expiries rather than read them whole. An unknown expiry raises
    NoMarketDataError whose detail lists the nearest listed expiries.
    """
    if notice := _live_only_notice(symbol, curr_date):
        return notice

    side = (side or "both").lower()
    moneyness = (moneyness or "all").lower()
    if side not in ("both", "calls", "puts"):
        raise ValueError(f"side {side!r} is one of 'both', 'calls', 'puts'")
    if moneyness not in ("all", "itm", "atm", "otm"):
        raise ValueError(f"moneyness {moneyness!r} is one of 'all', 'itm', 'atm', 'otm'")

    t, spot, expiries, note, used = _chain_source(symbol)
    if expiry not in expiries:
        raise _raise_absent(symbol, expiry, expiries)

    as_of = curr_date or get_current_date()
    dte = max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.strptime(as_of, "%Y-%m-%d")).days, 0)
    calls_df, puts_df = _load_chain(t, expiry)
    metrics = chain_metrics(spot, expiry, dte, calls_df, puts_df)

    lines = [
        f"## Option chain for {symbol} expiring {expiry} ({dte} DTE)"
        + (f" — chains quoted on {used}" if note else ""),
        "",
        f"Spot {_strike_text(spot)} · live Yahoo snapshot as of {as_of}",
        *_landmarks(metrics),
    ]

    sides = (("call", "Calls", calls_df), ("put", "Puts", puts_df))
    if side == "calls":
        sides = sides[:1]
    elif side == "puts":
        sides = sides[1:]

    for kind, title, df in sides:
        rows = _rows_for_table(df, kind, spot, max(dte, 1) / 365.25, moneyness)
        lines.append("")
        shown_note = ""
        if len(rows) < len(df):
            shown_note = f" of {len(df)} listed"
        lines.append(f"### {title} ({len(rows)} rows{shown_note})"
                     + (f", filtered to {moneyness.upper()}" if moneyness != "all" else ""))
        if not rows:
            lines.append("_no rows match this filter in this chain_")
            continue
        lines += [
            "",
            "| Strike | Money | IV % | Vol | OI | Bid | Ask | Last | Delta |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {_strike_text(r['strike'])} | {r['money']} | {_iv_text(r['iv'])} "
            f"| {_int_text(r['volume'])} | {_int_text(r['open_interest'])} "
            f"| {_num_text(r['bid'], '%.2f')} | {_num_text(r['ask'], '%.2f')} "
            f"| {_num_text(r['last'], '%.2f')} | {_delta_text(r['delta'])} |"
            for r in rows
        ]
    lines += ["", _data_note()]
    return "\n".join(lines)


# ---- report-detail helpers ----------------------------------------------------


def _landmarks(metrics: dict) -> list[str]:
    """The per-expiry headline readings as labelled bullets."""
    atm_call, atm_put = metrics.get("atm_call"), metrics.get("atm_put")
    lines = []
    if atm_call and atm_put:
        lines.append(
            f"- ATM strike {_strike_text(metrics['atm_strike'])}: call IV "
            f"{_iv_text(atm_call['iv'])} (delta {_delta_text(atm_call['delta'])}), put IV "
            f"{_iv_text(atm_put['iv'])} (delta {_delta_text(atm_put['delta'])})"
        )
    else:
        lines.append(
            f"- ATM strike {_strike_text(metrics['atm_strike'])}: no quotes at that strike; "
            "use the nearest quoted rows below"
        )
    lines.append(
        f"- 25-delta skew: OTM put IV {_iv_text(metrics['skew_put'][0] if metrics['skew_put'] else None)} "
        f"vs OTM call IV {_iv_text(metrics['skew_call'][0] if metrics['skew_call'] else None)}"
        f"{'' if metrics['skew_iv_gap'] is None else ' — ' + _skew_gap_text(metrics['skew_iv_gap'])}"
    )
    lines.append(
        f"- Put/Call ratio: {_ratio_text(metrics['put_call_volume'])} by volume, "
        f"{_ratio_text(metrics['put_call_oi'])} by open interest"
    )
    lines.append(
        f"- Flows: calls {_int_text(metrics['call_volume'])} traded / "
        f"{_int_text(metrics['call_open_interest'])} OI; puts "
        f"{_int_text(metrics['put_volume'])} traded / "
        f"{_int_text(metrics['put_open_interest'])} OI ({metrics['dte_days']} DTE)"
    )
    lines.append(
        f"- Max pain: {_strike_text(metrics['max_pain'])} against spot "
        f"{_strike_text(metrics['spot'])}"
    )
    lines.append(
        f"- Net gamma exposure (approx): {_gex_text(metrics['net_gamma'])} per 1-point move"
    )
    for label, key in (("Highest-OI calls", "top_call_oi"), ("Highest-OI puts", "top_put_oi")):
        if metrics[key]:
            detail = "; ".join(
                f"{_strike_text(strike)} (OI {_int_text(oi)})" for strike, oi in metrics[key]
            )
            lines.append(f"- {label}: {detail}")
    return lines


def _atm_iv(metrics: dict) -> float | None:
    """The mid of the ATM call and put IVs, when either side is quoted."""
    ivs = [r["iv"] for r in (metrics.get("atm_call"), metrics.get("atm_put"))
           if r and r.get("iv", 0) > 0]
    return sum(ivs) / len(ivs) if ivs else None


def _skew_text(gap) -> str:
    return f"{gap:+.1f}" if _finite(gap) else "-"


def _skew_gap_text(gap: float) -> str:
    """What a 25-delta skew gap means, in one clause."""
    direction = ("OTM puts carry extra volatility (downside protection is bid)" if gap > 1
                 else "OTM calls carry extra volatility (upside speculation is bid)" if gap < -1
                 else "the wings price near equally")
    return f"a {abs(gap):.1f}-vol-point {'put' if gap > 0 else 'call'} skew: {direction}"


def _oi_label(pairs: list[tuple[float, float]]) -> str:
    """A top-OI strike list as plain strikes, or a dash."""
    if not pairs:
        return "-"
    return ", ".join(_strike_text(strike) for strike, _oi in pairs)
