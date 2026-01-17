"""MCP server for Obsidian vault semantic search."""

from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .core import (
    DEFAULT_SEARCH_LIMIT,
    EmbeddingModelError,
    IndexError,
    PREVIEW_LENGTH,
    VaultError,
    ensure_embedding_model,
    get_vault_status,
    search_vault,
    validate_vault,
)
from .indexer import index_vault

# Global default vault path (set via CLI argument)
_default_vault_path: Path | None = None


# ============================================================================
# Tool Schema Definitions
# ============================================================================


def _build_tool_schemas(with_vault_path: bool) -> list[Tool]:
    """Build tool schemas based on whether vault_path parameter is required.

    When a default vault is configured via CLI, vault_path becomes optional.
    """
    vault_prop = {
        "vault_path": {
            "type": "string",
            "description": "Path to the Obsidian vault directory",
        }
    }

    def make_schema(properties: dict, required: list[str]) -> dict:
        props = {**properties}
        reqs = required.copy()
        if with_vault_path:
            props.update(vault_prop)
            reqs.append("vault_path")
        return {"type": "object", "properties": props, "required": reqs if reqs else None}

    # Clean up None required fields
    def clean_schema(schema: dict) -> dict:
        if schema.get("required") is None:
            del schema["required"]
        return schema

    return [
        Tool(
            name="obsidian_index",
            description="Index an Obsidian vault for semantic search. Creates embeddings for all markdown notes.",
            inputSchema=clean_schema(make_schema({}, [])),
        ),
        Tool(
            name="obsidian_update",
            description="Update the index for an Obsidian vault. Only re-indexes new or modified files.",
            inputSchema=clean_schema(make_schema({}, [])),
        ),
        Tool(
            name="obsidian_search",
            description="Semantic search in an indexed Obsidian vault. Returns notes matching the query.",
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
            name="obsidian_status",
            description="Get indexing status for an Obsidian vault.",
            inputSchema=clean_schema(make_schema({}, [])),
        ),
        Tool(
            name="obsidian_read",
            description="Read a note from an Obsidian vault by its path.",
            inputSchema=clean_schema(make_schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Path to the note (relative to vault root, as returned by search)",
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


# Pre-build schemas for both modes
_TOOLS_WITH_VAULT = _build_tool_schemas(with_vault_path=True)
_TOOLS_WITHOUT_VAULT = _build_tool_schemas(with_vault_path=False)


# ============================================================================
# Vault Path Resolution
# ============================================================================


def _get_vault_path(vault_path_str: str | None) -> Path:
    """Resolve vault path, defaulting to configured default or current directory."""
    if vault_path_str:
        return Path(vault_path_str).resolve()
    if _default_vault_path:
        return _default_vault_path
    return Path.cwd().resolve()


def _text(content: str) -> list[TextContent]:
    """Wrap text in a TextContent list for MCP responses."""
    return [TextContent(type="text", text=content)]


def _error(message: str) -> list[TextContent]:
    """Return an error response."""
    return _text(f"Error: {message}")


# ============================================================================
# Tool Handlers
# ============================================================================


async def handle_index(vault_path_str: str | None, update_only: bool) -> list[TextContent]:
    """Handle index/update tool calls."""
    vault_path = _get_vault_path(vault_path_str)

    try:
        validate_vault(vault_path)
    except VaultError as e:
        return _error(str(e))

    try:
        ensure_embedding_model()
    except EmbeddingModelError as e:
        return _error(str(e))

    try:
        indexed, skipped, deleted = index_vault(vault_path, update_only=update_only)

        action = "Updated" if update_only else "Indexed"
        lines = [f"{action} vault: {vault_path}", f"- Indexed: {indexed} notes"]
        if skipped:
            lines.append(f"- Skipped: {skipped} notes")
        if deleted:
            lines.append(f"- Deleted: {deleted} notes")

        return _text("\n".join(lines))
    except Exception as e:
        return _error(f"Indexing failed: {e}")


async def handle_search(
    query: str, vault_path_str: str | None, limit: int
) -> list[TextContent]:
    """Handle search tool calls."""
    vault_path = _get_vault_path(vault_path_str)

    try:
        results = search_vault(vault_path, query, limit=limit)
    except IndexError as e:
        return _error(str(e))
    except Exception as e:
        return _error(f"Search failed: {e}")

    if not results:
        return _text("No results found.")

    lines = [f"Search results for: {query}", ""]
    for i, result in enumerate(results, 1):
        # Use slightly longer preview for MCP (more context for AI)
        preview = result.preview(length=PREVIEW_LENGTH + 100)
        lines.extend([
            f"{i}. {result.title or '(untitled)'} (score: {result.score:.2f})",
            f"   Path: {result.path}",
            f"   Preview: {preview}",
            "",
        ])

    return _text("\n".join(lines))


async def handle_status(vault_path_str: str | None) -> list[TextContent]:
    """Handle status tool calls."""
    vault_path = _get_vault_path(vault_path_str)
    status = get_vault_status(vault_path)

    lines = [
        f"Vault: {status.vault_path}",
        f"Database: {status.db_path}",
    ]

    if status.indexed:
        lines.append(f"Status: Indexed ({status.note_count} notes, {status.chunk_count} chunks)")
    else:
        lines.append("Status: Not indexed")

    return _text("\n".join(lines))


async def handle_read(
    note_path: str,
    vault_path_str: str | None,
    offset: int = 0,
    limit: int | None = None,
) -> list[TextContent]:
    """Handle read tool calls."""
    vault_path = _get_vault_path(vault_path_str)
    full_path = vault_path / note_path

    if not full_path.exists():
        return _error(f"Note not found: {note_path}")

    if not full_path.is_file():
        return _error(f"Not a file: {note_path}")

    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset and limit
        if offset > 0:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]

        header_lines = [f"File: {note_path}", f"Total lines: {total_lines}"]
        if offset > 0 or limit is not None:
            header_lines.append(f"Showing lines {offset + 1}-{offset + len(lines)} of {total_lines}")
        header_lines.append("---")

        return _text("\n".join(header_lines) + "\n" + "\n".join(lines))
    except Exception as e:
        return _error(f"Read failed: {e}")


# ============================================================================
# Server Setup
# ============================================================================


def create_server(vault_path: Path | None = None) -> Server:
    """Create and configure the MCP server."""
    global _default_vault_path
    _default_vault_path = vault_path

    server = Server("obsidian-search")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return available tools based on configuration."""
        return _TOOLS_WITHOUT_VAULT if _default_vault_path else _TOOLS_WITH_VAULT

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Route tool calls to handlers."""
        vault = arguments.get("vault_path")

        handlers = {
            "obsidian_index": lambda: handle_index(vault, update_only=False),
            "obsidian_update": lambda: handle_index(vault, update_only=True),
            "obsidian_search": lambda: handle_search(
                arguments["query"], vault, arguments.get("limit", DEFAULT_SEARCH_LIMIT)
            ),
            "obsidian_status": lambda: handle_status(vault),
            "obsidian_read": lambda: handle_read(
                arguments["path"], vault, arguments.get("offset", 0), arguments.get("limit")
            ),
        }

        handler = handlers.get(name)
        if handler:
            return await handler()
        return _error(f"Unknown tool: {name}")

    return server


async def run_server(vault_path: Path | None = None):
    """Run the MCP server."""
    server = create_server(vault_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
