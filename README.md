# ragamuffin

Semantic document search for local directories. Indexes `.md`, `.txt`, and `.rst` files into SQLite with local embeddings (no API keys, no cloud).

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

## Usage

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
```

### Check index status

```bash
muffin --directory ~/notes status
```

### Web UI

```bash
muffin --directory ~/notes web
# Open http://127.0.0.1:5000
```

### Short alias

```bash
mf --directory ~/notes search "kubernetes ingress"
```

## MCP Server

Expose the search tools to AI assistants (Claude Desktop, etc.).

### With a fixed directory (recommended)

```bash
muffin --directory ~/notes mcp
```

Or via environment variables — useful when the assistant launches the server:

```bash
RAGAMUFFIN_DIRECTORY=~/notes muffin mcp
```

### Claude Desktop config

If installed via `uv tool install`:

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

Or without installing, using `uvx` directly from GitHub:

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

### Multiple instances with descriptions

Run one server per knowledge base and set `RAGAMUFFIN_DESCRIPTION` so the agent knows which is which:

```json
{
  "mcpServers": {
    "devops-docs": {
      "command": "muffin",
      "args": ["mcp"],
      "env": {
        "RAGAMUFFIN_DIRECTORY": "/Users/you/devops-docs",
        "RAGAMUFFIN_DESCRIPTION": "Search Acme Inc DevOps documentation"
      }
    },
    "personal-notes": {
      "command": "muffin",
      "args": ["mcp"],
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
