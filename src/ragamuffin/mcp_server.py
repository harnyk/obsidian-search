"""MCP server for ragamuffin semantic document search."""

import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from .core import (
    DEFAULT_SEARCH_LIMIT,
    EmbeddingModelError,
    IndexError,
    PREVIEW_LENGTH,
    build_obsidian_uri,
    ensure_embedding_model,
    get_index_status,
    is_obsidian_vault,
    search_dir,
)
from .indexer import index_directory

# Global default directory path (set via CLI argument or RAGAMUFFIN_DIRECTORY env var)
_default_dir_path: Path | None = None


# ============================================================================
# Tool Schema Definitions
# ============================================================================


def _build_tool_schemas(with_dir_path: bool, description: str | None = None) -> list[Tool]:
    """Build tool schemas based on whether dir_path parameter is required.

    When a default directory is configured, dir_path becomes optional.
    When description is set, it is prepended to search/index tool descriptions.
    """
    dir_prop = {
        "dir_path": {
            "type": "string",
            "description": "Path to the directory to search",
        }
    }

    def make_schema(properties: dict, required: list[str]) -> dict:
        props = {**properties}
        reqs = required.copy()
        if with_dir_path:
            props.update(dir_prop)
            reqs.append("dir_path")
        return {"type": "object", "properties": props, "required": reqs if reqs else None}

    def clean_schema(schema: dict) -> dict:
        if schema.get("required") is None:
            del schema["required"]
        return schema

    def with_desc(base: str) -> str:
        return f"{description} — {base}" if description else base

    return [
        Tool(
            name="muffin_index",
            description=with_desc("Index documents in a directory for semantic search. Incremental by default (skips unchanged files). Pass force=true to re-index everything."),
            inputSchema=clean_schema(make_schema(
                {
                    "force": {
                        "type": "boolean",
                        "description": "Re-index all files, not just new/modified ones (default: false)",
                        "default": False,
                    }
                },
                [],
            )),
        ),
        Tool(
            name="muffin_search",
            description=with_desc("Semantic search in an indexed directory. Returns documents matching the query."),
            inputSchema=clean_schema(make_schema(
                {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Maximum number of results (default: {DEFAULT_SEARCH_LIMIT})",
                        "default": DEFAULT_SEARCH_LIMIT,
                    },
                },
                ["query"],
            )),
        ),
        Tool(
            name="muffin_status",
            description=with_desc("Get indexing status for a directory."),
            inputSchema=clean_schema(make_schema({}, [])),
        ),
        Tool(
            name="muffin_read",
            description=with_desc("Read a document from the directory by its path."),
            inputSchema=clean_schema(make_schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Path to the document (relative to directory root, as returned by search)",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line offset to start reading from (0-based, default: 0)",
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return (default: all)",
                    },
                },
                ["path"],
            )),
        ),
    ]


# ============================================================================
# Directory Path Resolution
# ============================================================================


def _get_dir_path(dir_path_str: str | None) -> Path:
    """Resolve directory path, defaulting to configured default or current directory."""
    if dir_path_str:
        return Path(dir_path_str).resolve()
    if _default_dir_path:
        return _default_dir_path
    return Path.cwd().resolve()


def _text(content: str) -> list[TextContent]:
    return [TextContent(type="text", text=content)]


def _error(message: str) -> list[TextContent]:
    return _text(f"Error: {message}")


# ============================================================================
# Tool Handlers
# ============================================================================


async def handle_index(dir_path_str: str | None, update_only: bool) -> list[TextContent]:
    """Handle index/update tool calls."""
    dir_path = _get_dir_path(dir_path_str)

    try:
        ensure_embedding_model()
    except EmbeddingModelError as e:
        return _error(str(e))

    try:
        indexed, skipped, deleted = index_directory(dir_path, update_only=update_only)

        action = "Updated" if update_only else "Indexed"
        lines = [f"{action} directory: {dir_path}", f"- Indexed: {indexed} documents"]
        if skipped:
            lines.append(f"- Skipped: {skipped} documents")
        if deleted:
            lines.append(f"- Deleted: {deleted} documents")

        return _text("\n".join(lines))
    except Exception as e:
        return _error(f"Indexing failed: {e}")


async def handle_search(
    query: str, dir_path_str: str | None, limit: int
) -> list[TextContent]:
    """Handle search tool calls."""
    dir_path = _get_dir_path(dir_path_str)
    obsidian = is_obsidian_vault(dir_path)

    try:
        results = search_dir(dir_path, query, limit=limit)
    except IndexError as e:
        return _error(str(e))
    except Exception as e:
        return _error(f"Search failed: {e}")

    if not results:
        return _text("No results found.")

    lines = [f"Search results for: {query}", ""]
    for i, result in enumerate(results, 1):
        preview = result.preview(length=PREVIEW_LENGTH + 100)
        entry = [
            f"{i}. {result.title or '(untitled)'} (score: {result.score:.2f})",
            f"   Path: {result.path}",
            f"   Preview: {preview}",
        ]
        if obsidian:
            entry.append(f"   Obsidian URI: {build_obsidian_uri(dir_path, result.path)}")
        entry.append("")
        lines.extend(entry)

    return _text("\n".join(lines))


async def handle_status(dir_path_str: str | None) -> list[TextContent]:
    """Handle status tool calls."""
    dir_path = _get_dir_path(dir_path_str)
    status = get_index_status(dir_path)

    lines = [
        f"Directory: {status.dir_path}",
        f"Database: {status.db_path}",
    ]

    if status.indexed:
        lines.append(f"Status: Indexed ({status.doc_count} documents, {status.chunk_count} chunks)")
    else:
        lines.append("Status: Not indexed")

    return _text("\n".join(lines))


async def handle_read(
    doc_path: str,
    dir_path_str: str | None,
    offset: int = 0,
    limit: int | None = None,
) -> list[TextContent]:
    """Handle read tool calls."""
    dir_path = _get_dir_path(dir_path_str)
    full_path = dir_path / doc_path

    if not full_path.exists():
        return _error(f"Document not found: {doc_path}")

    if not full_path.is_file():
        return _error(f"Not a file: {doc_path}")

    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        total_lines = len(lines)

        if offset > 0:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]

        header_lines = [f"File: {doc_path}", f"Total lines: {total_lines}"]
        if offset > 0 or limit is not None:
            header_lines.append(f"Showing lines {offset + 1}-{offset + len(lines)} of {total_lines}")
        header_lines.append("---")

        return _text("\n".join(header_lines) + "\n" + "\n".join(lines))
    except Exception as e:
        return _error(f"Read failed: {e}")


# ============================================================================
# Server Setup
# ============================================================================


def create_server(dir_path: Path | None = None) -> Server:
    """Create and configure the MCP server."""
    global _default_dir_path

    if dir_path is None:
        env_dir = os.environ.get("RAGAMUFFIN_DIRECTORY")
        if env_dir:
            dir_path = Path(env_dir).resolve()

    _default_dir_path = dir_path

    description = os.environ.get("RAGAMUFFIN_DESCRIPTION") or None
    tools_without_dir = _build_tool_schemas(with_dir_path=False, description=description)
    tools_with_dir = _build_tool_schemas(with_dir_path=True, description=description)

    async def on_list_tools(ctx, params) -> ListToolsResult:
        tools = tools_without_dir if _default_dir_path else tools_with_dir
        return ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params) -> CallToolResult:
        name = params.name
        arguments = params.arguments or {}
        dir_path_arg = arguments.get("dir_path")

        handlers = {
            "muffin_index": lambda: handle_index(
                dir_path_arg, update_only=not arguments.get("force", False)
            ),
            "muffin_search": lambda: handle_search(
                arguments["query"], dir_path_arg, arguments.get("limit", DEFAULT_SEARCH_LIMIT)
            ),
            "muffin_status": lambda: handle_status(dir_path_arg),
            "muffin_read": lambda: handle_read(
                arguments["path"],
                dir_path_arg,
                arguments.get("offset", 0),
                arguments.get("limit"),
            ),
        }

        handler = handlers.get(name)
        if handler:
            content = await handler()
        else:
            content = _error(f"Unknown tool: {name}")
        return CallToolResult(content=content)

    return Server(
        "ragamuffin",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_server(dir_path: Path | None = None):
    """Run the MCP server."""
    server = create_server(dir_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
