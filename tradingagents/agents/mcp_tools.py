"""Custom MCP servers as tools for the analysts.

The ``mcp_servers`` config key (and its ``TRADINGAGENTS_MCP_SERVERS`` env
override) maps a nickname to a local stdio server spec::

    "mcp_servers": {
        "mydata": {"command": "python", "args": ["my_server.py"], "env": {...}},
    }

Every tool each server exposes joins every tool-using analyst's toolset, so
the model may call them like any built-in tool. The built-in tools stay in
charge of the core data (prices, fundamentals, news): an MCP tool may not
shadow a built-in name, whatever its server calls itself — the loader refuses
such a run at startup rather than let a call silently reach the wrong vendor.

This graph runs synchronously (``graph.invoke``), and ``langchain-mcp-adapters``
0.3.x only serves async tools, so each MCP tool gets a sync bridge: the call
runs in its own event loop and returns the server's text as the string the
analyst's tool node expects. A failed call is returned as an error sentinel
string the model can read, in the same spirit as ``vendor_unavailable``: it
says nothing about the instrument, and the agent must not fabricate values.

The bridge speaks to each server over stdio per call, so every analyst tool
round spends a subprocess spin-up on tool use; the trade-off is a server
process that can never outlive the call that uses it.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

# The analyst modules own the built-in tool names; a reserved name here means
# the run would offer two tools under one name, and the router would see only
# one of them.
BUILTIN_TOOL_MODULES = ("market_analyst", "news_analyst", "fundamentals_analyst")


def _builtin_tool_names() -> frozenset[str]:
    """Every tool name the analysts already offer, for collision checks."""
    import importlib

    names: set[str] = set()
    for module_name in BUILTIN_TOOL_MODULES:
        module = importlib.import_module(f"tradingagents.agents.analysts.{module_name}")
        names.update(tool.name for tool in module.TOOLS)
    return frozenset(names)


def _as_text(result) -> str:
    """The server's answer as the string an analyst reads.

    A call returns ``(content, artifact)`` — LangChain content blocks, a
    ``ToolMessage``, or plain text. Anything unusual is stringified rather
    than dropped: a report must show what the server said.
    """
    content = result[0] if isinstance(result, tuple) else result
    if isinstance(content, ToolMessage):
        block_text = content.content
        return block_text if isinstance(block_text, str) else json.dumps(block_text, default=str)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        texts = [
            block.get("text")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    # No readable text (images, empty errors, …): hand the value over as-is
    # so the run still records what the server produced.
    return json.dumps(_stringify(content), default=str, ensure_ascii=False)


def _stringify(value):
    """Content blocks hold non-JSON values at times; render each to text."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_stringify(item) for item in value]
    if isinstance(value, dict):
        return {key: _stringify(item) for key, item in value.items()}
    return value


def _stdio_connection(nickname: str, spec) -> dict:
    """Validate one ``mcp_servers`` entry into the client's stdio shape.

    Fail-loud, like every other config in this project: a misspelled key or a
    wrong type must stop the run before it starts, not surface mid-analysis as
    a tool that silently cannot run.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"mcp_servers[{nickname!r}] must be an object with a 'command', got {spec!r}"
        )
    command = spec.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"mcp_servers[{nickname!r}] needs a 'command' string, got {command!r}")

    connection = {"transport": "stdio", "command": command}
    args = spec.get("args")
    if args is not None:
        if not (isinstance(args, list) and all(isinstance(a, str) for a in args)):
            raise ValueError(f"mcp_servers[{nickname!r}]['args'] must be a list of strings, got {args!r}")
        connection["args"] = args
    for key in ("env", "cwd"):
        value = spec.get(key)
        if value is None:
            continue
        if key == "env":
            if not (isinstance(value, dict) and all(
                    isinstance(k, str) and isinstance(v, str) for k, v in value.items())):
                raise ValueError(
                    f"mcp_servers[{nickname!r}]['env'] must map names to string values, got {value!r}"
                )
        elif not isinstance(value, str):
            raise ValueError(f"mcp_servers[{nickname!r}]['cwd'] must be a string, got {value!r}")
        connection[key] = value
    if "/" in command and "cwd" in connection:
        # The child chdirs to ``cwd`` before executing the command, so the
        # executable resolves relative to ``cwd``, not to the process launching
        # the run — a relative path would stop pointing at the binary the
        # config meant. Bind it to that process's directory before then.
        resolved = os.path.abspath(command)
        if not os.path.exists(resolved):
            raise ValueError(
                f"mcp_servers[{nickname!r}] command does not exist: {command!r} "
                f"(resolved as {resolved!r})"
            )
        connection["command"] = resolved
    return connection


def _run_await(awaitable):
    """Drive one awaitable on its own event loop, whatever thread we are on.

    ``asyncio.run`` works from the usual graph thread; when the caller already
    runs a loop (embedding the graph in an asyncio app), the awaitable is sent
    to a fresh loop on a worker thread instead of failing on ``asyncio.run``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_await, awaitable).result()


_MCP_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}

# arg schemas the bridges can't map (nested `$ref`s, raw JSON Schema dicts the
# adapter left in place) pass through unchanged: providers still accept a
# plain schema, and the call-site kwargs come straight from the model.
_MISSING = object()


def _annotation(field_schema):
    """One inputSchema property as a Python type; unknown shapes come back as
    ``Any``, so a model passing anything still reaches the server."""
    declared = field_schema.get("type")
    if isinstance(declared, list):
        variants = tuple(_MCP_TYPE_MAP.get(item) for item in declared if item in _MCP_TYPE_MAP)
        if not variants:
            return Any
        if len(variants) == 1:
            return variants[0]
        from typing import Union
        return Union[variants]  # noqa: UP007 - create_model takes the typing form
    return _MCP_TYPE_MAP.get(declared, Any)


def _args_model(tool_name: str, input_schema) -> type[BaseModel] | None:
    """The tool's inputSchema as a pydantic model for providers and the tool node.

    A plain dict schema survives binding, but some models' tool paths expect
    fields to inspect, and a loose model passes exactly the kwargs through to
    the bridge. Unmappable schemas come back unchanged.
    """
    if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
        return None
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return None
    required = set(input_schema.get("required", []))
    fields = {}
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            return None
        annotation = _annotation(field_schema)
        if field_name in required:
            default = field_schema.get("default", ...)
        else:
            default = field_schema.get("default", None)
        fields[field_name] = (annotation, default)
    try:
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in tool_name)
        return create_model(f"{safe_name}_args", **fields)
    except Exception:  # noqa: BLE001 - fall back to the raw schema
        return None


def _sync_bridge(tool: BaseTool) -> BaseTool:
    """A sync-callable stand-in for an async-only adapter tool.

    The adapter's ``StructuredTool`` raises on a sync call, and LangGraph's
    tool node makes one every analyst turn under ``graph.invoke``. The bridge
    keeps the adapter's argument schema — the model sees the same tool — and
    runs the real call on a fresh loop, turning every failure into a sentinel
    string: an MCP server being down, timed out, or erroring must degrade the
    report's data, not end the run the way a raised exception out of a tool
    node would.
    """

    def run_mcp_tool(**kwargs):
        try:
            result = _run_await(tool.ainvoke(kwargs))
        except Exception as exc:  # noqa: BLE001 - degrade to a sentinel, see above
            logger.warning("MCP tool %s failed: %s", tool.name, exc)
            return (
                f"MCP_TOOL_ERROR: MCP tool {tool.name} could not be served ({type(exc).__name__}: "
                f"{exc}). This says nothing about the instrument; report the data as unavailable "
                "and do not estimate or fabricate values."
            )
        return _as_text(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=_args_model(tool.name, tool.args_schema) or tool.args_schema,
        func=run_mcp_tool,
    )


def load_mcp_tools(config: dict) -> tuple[BaseTool, ...]:
    """Every tool the configured MCP servers expose, sync-callable and
    collision-checked.

    Called once when the ``TradingAgentsGraph`` is built: an unavailable
    server, a bad entry, or a name collision is a startup failure, not a
    mid-run surprise. A config without ``mcp_servers`` returns nothing, so the
    analysts run exactly as before. Requires ``langchain-mcp-adapters`` when
    servers are configured; its absence is a loud, instructive failure.
    """
    servers = config.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        raise ValueError(f"mcp_servers must map nicknames to server specs, got {servers!r}")
    if not servers:
        return ()

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise ValueError(
            "mcp_servers is configured but langchain-mcp-adapters is not installed. "
            'Install it with: pip install "langchain-mcp-adapters"'
        ) from exc

    reserved = _builtin_tool_names()
    connections = {nickname: _stdio_connection(nickname, spec) for nickname, spec in servers.items()}
    adapter_tools = _run_await(MultiServerMCPClient(connections).get_tools())

    tools, names = [], set()
    for adapter_tool in adapter_tools:
        name = adapter_tool.name
        if name in reserved:
            raise ValueError(
                f"MCP server tool {name!r} shadows a built-in analyst tool of the same name; "
                "the server's tool must take a different name."
            )
        if name in names:
            raise ValueError(f"Two configured MCP servers expose a tool named {name!r}.")
        names.add(name)
        tools.append(_sync_bridge(adapter_tool))

    logger.info("Loaded %d MCP tool(s) from %d server(s): %s",
                len(tools), len(servers), ", ".join(sorted(names)))
    return tuple(tools)
