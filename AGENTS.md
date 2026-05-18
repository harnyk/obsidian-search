# ragamuffin — Agent Guide

## Project Overview

CLI tool (and MCP server + web app) that indexes a directory of documents into SQLite with `sqlite-vec` for semantic search using Ollama embeddings. Supports `.md`, `.txt`, and `.rst` files. Automatically detects Obsidian vaults and shows `obsidian://` deep links when applicable.

- **Language**: Python ≥ 3.10
- **Package manager**: `uv`
- **Build backend**: `hatchling`
- **Entry points**: `muffin` and `mf` → `ragamuffin.cli:cli`
- **Version**: tracked in `pyproject.toml`

## Module Map

| File | Responsibility |
|------|---------------|
| `cli.py` | Click CLI — commands: `index`, `search`, `status`, `mcp`, `web`; `--directory/-d` option |
| `core.py` | Shared constants, data types (`SearchResult`, `IndexStatus`), exceptions, `is_obsidian_vault()`, `build_obsidian_uri()`, `search_dir()`, `get_index_status()` |
| `database.py` | SQLite + `sqlite-vec` operations; DB lives at `<dir>/.ragamuffin/index.db` |
| `embeddings.py` | Local embedding via `fastembed` (`BAAI/bge-base-en-v1.5`, 768-dim, ONNX); module-level singleton; model cached in `~/.cache/fastembed` |
| `indexer.py` | Directory scanning (skips hidden paths), change detection by `mtime`, `index_directory()` |
| `parser.py` | Per-format parsing: `parse_markdown`, `parse_txt`, `parse_rst`; dispatch via `parse_document()`; overlapping chunking |
| `mcp_server.py` | MCP server (name: `ragamuffin`) exposing `muffin_index/update/search/status/read` tools |
| `web_app.py` | Flask web UI for search and document viewing |
| `templates/` | Jinja2 templates used by the Flask app |

## Obsidian Auto-Detection

`is_obsidian_vault(dir_path)` in `core.py` checks for `<dir>/.obsidian/`. When true:

- CLI `search`: results are rendered as clickable `obsidian://` links
- MCP `muffin_search`: includes `Obsidian URI:` line per result
- Web app: "Open in Obsidian" button is shown in results and document view

When false, plain file paths are used and no Obsidian UI is shown.

## Supported File Types

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.md` | `parse_markdown` | Frontmatter, wiki-links, heading extraction |
| `.txt` | `parse_txt` | Plain text; title from filename |
| `.rst` | `parse_rst` | RST directives/roles stripped; title from first section heading |

All parsers produce a `ParsedDocument` and feed into the shared `chunk_text()` function.

## Key Design Decisions

- **Chunking**: `CHUNK_SIZE = 1500` chars, `CHUNK_OVERLAP = 200` chars; each chunk is prefixed with the document title for embedding context.
- **Deduplication**: Multiple chunks from the same document are collapsed to the best-scoring one (`deduplicate_results` in `core.py`).
- **Unified search**: `search_dir()` in `core.py` is the single implementation used by CLI, web app, and MCP server. Do not duplicate it.
- **MCP dual-mode**: When started via `muffin mcp`, `dir_path` becomes optional for all tools; without a default, callers must pass it explicitly.
- **Error types**: `IndexError`, `EmbeddingModelError` are defined in `core.py` and should be raised (not generic exceptions) for expected failure paths.

## Running & Installing

```bash
# Install/reinstall locally
uv tool install . --force

# Run CLI
muffin --directory /path/to/docs index
muffin --directory /path/to/docs search "query"
muffin --directory /path/to/docs web
muffin --directory /path/to/docs mcp

# Short alias
mf --directory /path/to/docs search "query"
```

## Developer Workflow

After making changes:

1. Bump the patch version in `pyproject.toml`.
2. Reinstall: `uv tool install . --force`
3. Commit and push.

There is a slash command `.claude/commands/sync-all.md` that automates all three steps.

## Conventions

- All shared logic belongs in `core.py`, not duplicated across CLI / web / MCP.
- New CLI commands use the `@pass_dir` decorator and `DirContext`.
- Database connections are always opened via `open_database()` context manager (ensures `sqlite-vec` is loaded and connection is closed).
- Hidden files and folders (names starting with `.`) are excluded from indexing.
- Do not add comments that narrate what the code does. Comments should explain non-obvious intent or constraints only.
