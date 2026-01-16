"""MCP server for Obsidian vault semantic search."""

from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .database import get_chunk_count, get_db_path, get_note_count, init_db, search_similar
from .embeddings import ensure_model_available, get_embedding
from .indexer import index_vault

# Global default vault path (set via CLI argument)
_default_vault_path: Path | None = None


def create_server(vault_path: Path | None = None) -> Server:
    """Create and configure the MCP server."""
    global _default_vault_path
    _default_vault_path = vault_path

    server = Server("obsidian-search")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        if _default_vault_path:
            # Vault configured via CLI - don't expose vault_path parameter
            return [
                Tool(
                    name="obsidian_index",
                    description="Index the Obsidian vault for semantic search. Creates embeddings for all markdown notes.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="obsidian_update",
                    description="Update the index for the Obsidian vault. Only re-indexes new or modified files.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="obsidian_search",
                    description="Semantic search in the indexed Obsidian vault. Returns notes matching the query.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (natural language)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="obsidian_status",
                    description="Get indexing status for the Obsidian vault.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="obsidian_read",
                    description="Read a note from the Obsidian vault by its path.",
                    inputSchema={
                        "type": "object",
                        "properties": {
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
                        "required": ["path"],
                    },
                ),
            ]
        else:
            # No default vault - require vault_path parameter
            return [
                Tool(
                    name="obsidian_index",
                    description="Index an Obsidian vault for semantic search. Creates embeddings for all markdown notes.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vault_path": {
                                "type": "string",
                                "description": "Path to the Obsidian vault directory",
                            },
                        },
                        "required": ["vault_path"],
                    },
                ),
                Tool(
                    name="obsidian_update",
                    description="Update the index for an Obsidian vault. Only re-indexes new or modified files.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vault_path": {
                                "type": "string",
                                "description": "Path to the Obsidian vault directory",
                            },
                        },
                        "required": ["vault_path"],
                    },
                ),
                Tool(
                    name="obsidian_search",
                    description="Semantic search in an indexed Obsidian vault. Returns notes matching the query.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (natural language)",
                            },
                            "vault_path": {
                                "type": "string",
                                "description": "Path to the Obsidian vault directory",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query", "vault_path"],
                    },
                ),
                Tool(
                    name="obsidian_status",
                    description="Get indexing status for an Obsidian vault.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vault_path": {
                                "type": "string",
                                "description": "Path to the Obsidian vault directory",
                            },
                        },
                        "required": ["vault_path"],
                    },
                ),
                Tool(
                    name="obsidian_read",
                    description="Read a note from an Obsidian vault by its path.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the note (relative to vault root, as returned by search)",
                            },
                            "vault_path": {
                                "type": "string",
                                "description": "Path to the Obsidian vault directory",
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
                        "required": ["path", "vault_path"],
                    },
                ),
            ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        vault = arguments.get("vault_path") or (_default_vault_path and str(_default_vault_path))

        if name == "obsidian_index":
            return await handle_index(vault, update_only=False)
        elif name == "obsidian_update":
            return await handle_index(vault, update_only=True)
        elif name == "obsidian_search":
            return await handle_search(
                arguments["query"],
                vault,
                arguments.get("limit", 10),
            )
        elif name == "obsidian_status":
            return await handle_status(vault)
        elif name == "obsidian_read":
            return await handle_read(
                arguments["path"],
                vault,
                arguments.get("offset", 0),
                arguments.get("limit"),
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


def get_vault_path(vault_path_str: str | None) -> Path:
    """Resolve vault path, defaulting to configured default or current directory."""
    if vault_path_str:
        return Path(vault_path_str).resolve()
    if _default_vault_path:
        return _default_vault_path
    return Path.cwd().resolve()


async def handle_index(vault_path_str: str | None, update_only: bool) -> list[TextContent]:
    """Handle index/update tool calls."""
    vault_path = get_vault_path(vault_path_str)

    if not (vault_path / ".obsidian").exists():
        return [TextContent(
            type="text",
            text=f"Error: {vault_path} does not appear to be an Obsidian vault (missing .obsidian folder)",
        )]

    if not ensure_model_available():
        return [TextContent(
            type="text",
            text="Error: Could not load embedding model. Make sure Ollama is running and bge-m3 is available.",
        )]

    try:
        indexed, skipped, deleted = index_vault(vault_path, update_only=update_only)
        action = "Updated" if update_only else "Indexed"
        result = f"{action} vault: {vault_path}\n"
        result += f"- Indexed: {indexed} notes\n"
        if skipped:
            result += f"- Skipped: {skipped} notes\n"
        if deleted:
            result += f"- Deleted: {deleted} notes\n"
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error indexing vault: {e}")]


async def handle_search(query: str, vault_path_str: str | None, limit: int) -> list[TextContent]:
    """Handle search tool calls."""
    vault_path = get_vault_path(vault_path_str)
    db_path = get_db_path(vault_path)

    if not db_path.exists():
        return [TextContent(
            type="text",
            text=f"Error: No index found for {vault_path}. Run obsidian_index first.",
        )]

    try:
        conn = init_db(db_path)
        query_embedding = get_embedding(query)
        results = search_similar(conn, query_embedding, limit=limit)
        conn.close()

        if not results:
            return [TextContent(type="text", text="No results found.")]

        # Deduplicate by note path, keep best score
        seen_paths: dict[str, tuple] = {}
        for result in results:
            note_id, path, title, note_content, chunk_content, distance = result
            if path not in seen_paths or distance < seen_paths[path][5]:
                seen_paths[path] = result

        deduplicated = sorted(seen_paths.values(), key=lambda x: x[5])

        output = f"Search results for: {query}\n\n"
        for i, (note_id, path, title, note_content, chunk_content, distance) in enumerate(deduplicated, 1):
            score = 1 / (1 + distance)
            preview = chunk_content.replace("\n", " ")[:300]
            if len(chunk_content) > 300:
                preview += "..."
            output += f"{i}. {title or '(untitled)'} (score: {score:.2f})\n"
            output += f"   Path: {path}\n"
            output += f"   Preview: {preview}\n\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching: {e}")]


async def handle_status(vault_path_str: str | None) -> list[TextContent]:
    """Handle status tool calls."""
    vault_path = get_vault_path(vault_path_str)
    db_path = get_db_path(vault_path)

    result = f"Vault: {vault_path}\n"
    result += f"Database: {db_path}\n"

    if not db_path.exists():
        result += "Status: Not indexed"
        return [TextContent(type="text", text=result)]

    try:
        conn = init_db(db_path)
        note_count = get_note_count(conn)
        chunk_count = get_chunk_count(conn)
        conn.close()
        result += f"Status: Indexed ({note_count} notes, {chunk_count} chunks)"
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting status: {e}")]


async def handle_read(
    note_path: str,
    vault_path_str: str | None,
    offset: int = 0,
    limit: int | None = None,
) -> list[TextContent]:
    """Handle read tool calls."""
    vault_path = get_vault_path(vault_path_str)
    full_path = vault_path / note_path

    if not full_path.exists():
        return [TextContent(type="text", text=f"Error: Note not found: {note_path}")]

    if not full_path.is_file():
        return [TextContent(type="text", text=f"Error: Not a file: {note_path}")]

    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset and limit
        if offset > 0:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]

        output = f"File: {note_path}\n"
        output += f"Total lines: {total_lines}\n"
        if offset > 0 or limit is not None:
            output += f"Showing lines {offset + 1}-{offset + len(lines)} of {total_lines}\n"
        output += "---\n"
        output += "\n".join(lines)

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading note: {e}")]


async def run_server(vault_path: Path | None = None):
    """Run the MCP server."""
    server = create_server(vault_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
