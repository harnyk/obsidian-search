# Design: Migrate ragamuffin MCP server to mcp SDK 2.x

**Date:** 2026-08-18  
**Status:** Approved  
**Repo:** `ragamuffin-mcp`

## Problem

`uvx --from git+https://github.com/harnyk/ragamuffin-mcp muffin mcp` resolves `mcp==2.0.0`. In SDK 2.x, `Server` no longer exposes decorator APIs `@server.list_tools()` / `@server.call_tool()`. Startup fails with:

```text
AttributeError: 'Server' object has no attribute 'list_tools'
```

Cursor then reports MCP `cloud_docs_rag_search` / `backstage_rag_search` as disconnected (0 tools).

## Goals

- `muffin mcp` starts successfully under mcp 2.x.
- Tool names, schemas, and handler behavior unchanged (`muffin_index`, `muffin_search`, `muffin_status`, `muffin_read`).
- Dependency on `mcp` is version-pinned so `uvx` cannot silently pull an incompatible major.

## Non-goals

- Rewrite on FastMCP.
- Dual support for mcp 1.x and 2.x.
- Changes to CLI/web indexing/search logic.
- Cursor `mcp.json` workarounds (`--with mcp<2`).

## Approach

Minimal low-level migration in `src/ragamuffin/mcp_server.py`:

1. Keep `_build_tool_schemas`, `handle_*`, directory/env resolution as-is.
2. Replace decorator registration with constructor handlers:
   - `on_list_tools(ctx, params) -> ListToolsResult`
   - `on_call_tool(ctx, params) -> CallToolResult`
3. Wire: `Server("ragamuffin", on_list_tools=..., on_call_tool=...)`.
4. Map `params.name` / `params.arguments` into existing handler dispatch; wrap `list[TextContent]` in `CallToolResult(content=...)`.
5. Keep `stdio_server` + `server.run(...)` startup path.

## Dependency pin

In `pyproject.toml`:

```toml
"mcp>=2,<3",
```

Regenerate `uv.lock`. Bump package version (patch) per `AGENTS.md`.

## Success criteria

- Local: `RAGAMUFFIN_DIRECTORY=<dir> uv run muffin mcp` does not raise `AttributeError` on start (stdio server stays up until stdin closes / SIGTERM).
- After publish/push: Cursor MCP servers using `uvx --from git+...` list the four muffin tools again.

## Risks

- SDK 2 may change result type shapes further within 2.x — mitigated by `<3` pin and smoke test against resolved lockfile.
- Import paths (`mcp.types` vs `mcp_types`) — use public `mcp.server` / `mcp.types` exports preferred by the package docs.
