# MCP SDK 2.x Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `muffin mcp` work with mcp Python SDK 2.x and pin `mcp>=2,<3`.

**Architecture:** Replace decorator-based tool registration with constructor `on_list_tools` / `on_call_tool` handlers; keep existing tool schemas and `handle_*` logic unchanged.

**Tech Stack:** Python 3.10+, `mcp` 2.x, `uv`, hatchling

## Global Constraints

- Pin dependency exactly as: `mcp>=2,<3`
- Do not change tool names: `muffin_index`, `muffin_search`, `muffin_status`, `muffin_read`
- Do not change `RAGAMUFFIN_DIRECTORY` / `RAGAMUFFIN_DESCRIPTION` behavior
- Bump patch version in `pyproject.toml` per `AGENTS.md`
- Work in `/Users/mark.harnyk.ext/CODE/MY/ragamuffin-mcp`

## File map

| File | Role |
|------|------|
| `src/ragamuffin/mcp_server.py` | Migrate Server wiring to mcp 2.x handlers |
| `pyproject.toml` | Pin `mcp>=2,<3`, bump version |
| `uv.lock` | Lock resolved mcp 2.x |
| `docs/superpowers/specs/2026-08-18-mcp-sdk-v2-migration-design.md` | Approved design (already written) |

---

### Task 1: Upgrade dependency pin and lockfile

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (via `uv lock` / `uv sync`)

- [x] **Step 1:** In `pyproject.toml`, set `version = "0.9.4"` and change `"mcp"` to `"mcp>=2,<3"`.
- [x] **Step 2:** Run `uv lock && uv sync` in the repo root.
- [x] **Step 3:** Confirm resolved mcp major is 2: `uv run python -c "import importlib.metadata as m; print(m.version('mcp'))"` → starts with `2.`.

---

### Task 2: Migrate `create_server` to `on_*` handlers

**Files:**
- Modify: `src/ragamuffin/mcp_server.py`

**Interfaces:**
- Consumes: existing `_build_tool_schemas`, `handle_*`, `_error`
- Produces: `create_server(dir_path) -> Server` that registers tools via mcp 2.x constructor API

- [x] **Step 1:** Update imports to include `ListToolsResult`, `CallToolResult` (and keep `Server`, `stdio_server`, `TextContent`, `Tool`).
- [x] **Step 2:** Rewrite `create_server` to define nested `async def on_list_tools(ctx, params)` returning `ListToolsResult(tools=...)` and `async def on_call_tool(ctx, params)` that dispatches by `params.name` / `(params.arguments or {})` into existing handlers and returns `CallToolResult(content=...)`.
- [x] **Step 3:** Construct `Server("ragamuffin", on_list_tools=on_list_tools, on_call_tool=on_call_tool)`.
- [x] **Step 4:** Leave `run_server` using `stdio_server` + `server.run(...)` unless the 2.x API requires a signature tweak (fix compile/import errors only).

---

### Task 3: Smoke-test MCP startup

**Files:** none (verification only)

- [x] **Step 1:** Run a short-lived process that imports `create_server` and asserts `list_tools` decorator is not required — call `create_server(Path(...))` successfully under mcp 2.
- [x] **Step 2:** Start `uv run muffin mcp` with `RAGAMUFFIN_DIRECTORY` set, feed nothing / close stdin after brief wait, confirm no `AttributeError: list_tools` in stderr.
- [x] **Step 3:** Optional: `uv run pytest` if existing tests are cheap and green.

---

### Task 4: Commit (only if user asked) / handoff

- [ ] **Step 1:** Summarize changes; do **not** commit unless user explicitly requests a commit.
- [ ] **Step 2:** Remind user that Cursor picks up the fix after push to GitHub (because `mcp.json` uses `git+https://github.com/harnyk/ragamuffin-mcp`).
