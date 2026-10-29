"""MCP servers from config as analyst tools, with a real stdio server.

The toy server below is the shape of a caller's custom server: a tool the
model calls like any built-in, one that shadows a built-in name, and one that
errors. The loaders' fail-loud rules (config shape, shadowed names) are pinned
here as they are everywhere else in the project, and the calls go through the
same synchronous `graph.invoke` path a real run takes.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Annotated, TypedDict

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

pytest.importorskip("langchain_mcp_adapters", reason="requires tradingagents[mcp]")

from tradingagents.agents import mcp_tools
from tradingagents.default_config import build_default_config

CLEAN_SERVER = textwrap.dedent("""
    import os
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("toy")


    @mcp.tool()
    def echo_ticker(ticker: str) -> str:
        \"\"\"Echo the ticker back, with a fabricated price number.\"\"\"
        # A run record outside the reply: the caller of this test suite reads
        # what actually arrived at the server, whatever mutated the model.
        path = os.environ.get("MCP_HITS_PATH")
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{ticker}\\n")
        return f"ticker={ticker} price=123.45"


    @mcp.tool()
    def broken_tool(topic: str) -> str:
        \"\"\"Fails like a server gone wrong.\"\"\"
        raise RuntimeError("server exploded")


    if __name__ == "__main__":
        mcp.run()
    """)

# Names a tool `get_stock_data` (a built-in), so the loader must refuse it.
ROGUE_SERVER = textwrap.dedent("""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("toy")


    @mcp.tool()
    def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
        \"\"\"Shadows a built-in analyst tool name on purpose.\"\"\"
        return "must be refused"


    if __name__ == "__main__":
        mcp.run()
    """)

TRADE_DATE = "2026-01-09"


@pytest.fixture
def offline_mini(monkeypatch):
    """The vendor patches a full graph run needs when it must not touch the net.

    A copy of the end-to-end suite's `offline` fixture, trimmed to what this
    module's runs do.
    """
    from tradingagents.agents import context
    from tradingagents.agents.analysts import sentiment_analyst
    from tradingagents.dataflows import router
    from tradingagents.dataflows.vendors.yahoo import market as yahoo_market, snapshot

    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in vendors:
            monkeypatch.setitem(vendors, vendor,
                                lambda *a, _m=method, **k: f"{_m} data")
    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=60),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000,
    })
    monkeypatch.setattr(snapshot, "load_ohlcv", lambda *a, **k: prices.copy())
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages", lambda *a, **k: "no posts")
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts", lambda *a, **k: "no posts")
    monkeypatch.setattr(yahoo_market.yf, "Ticker", lambda s: type("T", (), {"info": {"longName": "NVIDIA"}})())
    context._identity.cache_clear()
    yield


def toy_config(tmp_path, source=CLEAN_SERVER, name="toy", hits=None):
    """The mcp_servers config for a toy server, launched by this interpreter."""
    server = tmp_path / f"{name}_mcp_server.py"
    server.write_text(source, encoding="utf-8")
    spec = {"command": sys.executable, "args": [str(server)]}
    if hits is not None:
        spec["env"] = {"MCP_HITS_PATH": str(hits)}
    return {"mcp_servers": {name: spec}}


def test_no_servers_means_no_tools():
    assert mcp_tools.load_mcp_tools({}) == ()
    assert mcp_tools.load_mcp_tools({"mcp_servers": {}}) == ()


def test_env_override_reads_json(monkeypatch):
    # A structured config value arrives as JSON through one TRADINGAGENTS_* string.
    import tradingagents.default_config as default_config_module

    for name in default_config_module._ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRADINGAGENTS_MCP_SERVERS", '{"mydata": {"command": "python", "args": ["s.py"]}}')
    config = build_default_config()
    assert config["mcp_servers"] == {"mydata": {"command": "python", "args": ["s.py"]}}


def test_malformed_env_json_fails_loudly(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MCP_SERVERS", "{not json")
    with pytest.raises(ValueError, match="expected JSON"):
        build_default_config()


@pytest.mark.parametrize("spec", [
    "python",
    {"args": ["x"]},
    {"command": "python", "args": "s.py"},
    {"command": "python", "env": {"PATH": 3}},
    {"command": "python", "cwd": 3},
])
def test_a_bad_entry_refuses_the_run(spec):
    with pytest.raises(ValueError, match="mcp_servers"):
        mcp_tools.load_mcp_tools({"mcp_servers": {"bad": spec}})


def test_missing_dependency_is_an_instructive_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_adapter(name, *args, **kwargs):
        if name.startswith("langchain_mcp_adapters"):
            raise ImportError("No module named 'langchain_mcp_adapters'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_adapter)
    with pytest.raises(ValueError, match="langchain-mcp-adapters"):
        mcp_tools.load_mcp_tools({"mcp_servers": {"toy": {"command": "python"}}})


def test_tools_load_and_run_synchronously(tmp_path):
    tools = mcp_tools.load_mcp_tools(toy_config(tmp_path))

    assert [t.name for t in tools] == ["echo_ticker", "broken_tool"]
    assert tools[0].invoke({"ticker": "NVDA"}) == "ticker=NVDA price=123.45"


def test_a_shadowed_builtin_name_refuses_the_run(tmp_path):
    """Whatever its server named it, an MCP tool may not shadow a built-in:
    two tools under one name would let a call reach nowhere reliable."""
    with pytest.raises(ValueError, match="shadows a built-in"):
        mcp_tools.load_mcp_tools(toy_config(tmp_path, source=ROGUE_SERVER))


def test_a_server_error_degrades_to_a_error_string(tmp_path):
    """A tool that fails inside the server comes back as text the analyst can
    read and react to, not as an exception out of the tool node."""
    tools = mcp_tools.load_mcp_tools(toy_config(tmp_path))
    broken = next(t for t in tools if t.name == "broken_tool")

    outcome = broken.invoke({"topic": "x"})

    assert isinstance(outcome, str)
    # The adapter renders the server's failure as text; the exact wording is
    # the adapter's, so pin only that it arrives as readable text.
    assert "broken_tool" in outcome and "server exploded" in outcome


def test_a_transport_failure_degrades_to_a_sentinel(tmp_path, monkeypatch):
    """The bridge catches whatever the MCP layer raises, so a broken server
    costs the analyst its data, not the run."""
    tools = mcp_tools.load_mcp_tools(toy_config(tmp_path))

    def exploded(awaitable):
        raise RuntimeError("connection died")

    monkeypatch.setattr(mcp_tools, "_run_await", exploded)

    outcome = tools[0].invoke({"ticker": "NVDA"})

    assert outcome.startswith("MCP_TOOL_ERROR")
    assert "do not estimate or fabricate" in outcome


def test_an_analyst_tool_node_runs_an_mcp_tool(tmp_path):
    """The exact wiring an analyst takes: its sub-graph's tool node under a
    synchronous graph.invoke, the result coming back as a ToolMessage."""

    tools = mcp_tools.load_mcp_tools(toy_config(tmp_path))
    echo = tools[0]

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    turn = {"n": 0}

    def agent(state):
        turn["n"] += 1
        if turn["n"] < 2:
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": echo.name, "args": {"ticker": "TSLA"}, "id": "1", "type": "tool_call"}])]}
        return {"messages": [AIMessage(content="REPORT")]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
    graph.add_edge("tools", "agent")
    app = graph.compile()
    out = app.invoke({"messages": []})

    tool_reports = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_reports and tool_reports[0].content == "ticker=TSLA price=123.45"
    assert out["messages"][-1].content == "REPORT"


def test_the_graph_offers_mcp_tools_to_every_tool_using_analyst(tmp_path, monkeypatch, offline_mini):
    """Through TradingAgentsGraph itself: each tool-using analyst calls the toy
    server's tool through the real tool node, and the run still reaches a
    decision. The analysts' tool messages stay inside their sub-graphs, so the
    server's own record is what shows the calls arrived."""
    from tests.test_graph_end_to_end import ScriptedModel, _graph

    hits = tmp_path / "mcp_hits.log"
    model = ScriptedModel(structured=True)
    graph = _graph(tmp_path, monkeypatch, model, **toy_config(tmp_path, hits=hits))

    state, rating = graph.propagate("NVDA", TRADE_DATE)

    assert rating == "Overweight"
    assert hits.read_text().splitlines() == ["NVDA"] * 3   # market, news, fundamentals
    assert state["market_report"].strip()


def test_an_empty_mcp_config_leaves_the_graph_unchanged(tmp_path, monkeypatch, offline_mini):
    from tests.test_graph_end_to_end import ScriptedModel, _graph

    graph = _graph(tmp_path, monkeypatch, ScriptedModel(structured=True))

    assert graph.graph_setup.extra_analyst_tools == ()
