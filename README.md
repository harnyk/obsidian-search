# ragamuffin

Semantic document search for local directories. Indexes `.md`, `.txt`, and `.rst` files into SQLite with local embeddings (no API keys, no cloud).

## Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (for install and `uvx`)

## Install

**From GitHub (no PyPI required):**

```bash
uv tool install git+https://github.com/harnyk/ragamuffin-mcp
```

**From local source:**

```bash
uv tool install .
```

**Run without installing (ephemeral):**

```bash
uvx --from git+https://github.com/harnyk/ragamuffin-mcp muffin --directory ~/notes search "your query"
```

Pin a branch, tag, or commit:

```bash
uvx --from git+https://github.com/harnyk/ragamuffin-mcp@v0.9.0 muffin --directory ~/notes search "your query"
```

On first run, the embedding model (~435 MB) is downloaded and cached in `~/.cache/fastembed`.

## Usage

All commands accept `--directory` / `-d` to point at a folder. If omitted, the current working directory is used.

### Index a directory

```bash
muffin --directory ~/notes index
```

Re-index everything (skip change detection):

```bash
muffin --directory ~/notes index --force
```

### Search

```bash
muffin --directory ~/notes search "how to set up a VPN"
muffin --directory ~/notes search --limit 5 "kubernetes ingress"
```

### Check index status

```bash
muffin --directory ~/notes status
```

### Web UI

```bash
muffin --directory ~/notes web
# Open http://127.0.0.1:8077
```

Custom host/port:

```bash
muffin --directory ~/notes web --host 0.0.0.0 --port 8080
```

### Short alias

```bash
mf --directory ~/notes search "kubernetes ingress"
```

With `uvx`, replace `muffin` with the full `uvx --from git+https://github.com/harnyk/ragamuffin-mcp muffin` prefix (or use the short alias `mf`).

## MCP Server

Expose the search tools to AI assistants (Cursor, Claude Desktop, etc.).

### With a fixed directory (recommended)

```bash
muffin --directory ~/notes mcp
```

Or via environment variables — useful when the assistant launches the server:

```bash
RAGAMUFFIN_DIRECTORY=~/notes muffin mcp
```

With `uvx`:

```bash
RAGAMUFFIN_DIRECTORY=~/notes uvx --from git+https://github.com/harnyk/ragamuffin-mcp muffin mcp
```

### Without a fixed directory

If neither `--directory` nor `RAGAMUFFIN_DIRECTORY` is set, each MCP tool call must include a `dir_path` argument. When a default directory is configured, `dir_path` is omitted from the tool schemas.

### Cursor config

Project-local (`.cursor/mcp.json`) or global (`~/.cursor/mcp.json`):

**Installed via `uv tool install`:**

```json
{
  "mcpServers": {
    "notes": {
      "command": "muffin",
      "args": ["mcp"],
      "env": {
        "RAGAMUFFIN_DIRECTORY": "/Users/you/notes"
      }
    }
  }
}
```

**Without installing, via `uvx` from GitHub:**

```json
{
  "mcpServers": {
    "notes": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/harnyk/ragamuffin-mcp", "muffin", "mcp"],
      "env": {
        "RAGAMUFFIN_DIRECTORY": "/Users/you/notes"
      }
    }
  }
}
```

### Claude Desktop config

Same JSON as above; config file location is platform-specific (see [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user)).

### Multiple instances with descriptions

Run one server per knowledge base and set `RAGAMUFFIN_DESCRIPTION` so the agent knows which is which:

```json
{
  "mcpServers": {
    "devops-docs": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/harnyk/ragamuffin-mcp", "muffin", "mcp"],
      "env": {
        "RAGAMUFFIN_DIRECTORY": "/Users/you/devops-docs",
        "RAGAMUFFIN_DESCRIPTION": "Search Acme Inc DevOps documentation"
      }
    },
    "personal-notes": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/harnyk/ragamuffin-mcp", "muffin", "mcp"],
      "env": {
        "RAGAMUFFIN_DIRECTORY": "/Users/you/notes",
        "RAGAMUFFIN_DESCRIPTION": "Search personal notes and journal"
      }
    }
  }
}
```

The description is prepended to each tool's description so the agent can pick the right server for a given query.

### MCP tools

| Tool | Description |
|------|-------------|
| `muffin_index` | Index (or re-index) the directory |
| `muffin_search` | Semantic search; returns ranked results with previews |
| `muffin_status` | Show index statistics |
| `muffin_read` | Read a document by path (supports offset/limit) |

## Obsidian vaults

ragamuffin auto-detects Obsidian vaults (looks for `.obsidian/`). When detected, search results include `obsidian://` deep links in the CLI, MCP output, and web UI.

## Supported file types

| Extension | Notes |
|-----------|-------|
| `.md` | Frontmatter stripped, wiki-links preserved |
| `.txt` | Title taken from filename |
| `.rst` | Directives and roles stripped |

## Notes

- Embeddings use `BAAI/bge-base-en-v1.5` via `fastembed` (~435 MB, downloaded on first run, cached in `~/.cache/fastembed`)
- Index is stored at `<directory>/.ragamuffin/index.db`
- Hidden files and directories (names starting with `.`) are excluded
