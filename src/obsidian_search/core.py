"""Core types, constants, and shared utilities for obsidian-search."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

# ============================================================================
# Constants
# ============================================================================

DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50
PREVIEW_LENGTH = 200
DEFAULT_WEB_PORT = 8077

# Error messages (centralized for consistency)
ERROR_NOT_OBSIDIAN_VAULT = (
    "{path} does not appear to be an Obsidian vault (missing .obsidian folder)"
)
ERROR_NO_INDEX = "No index found for {path}. Run 'obsidian-search index' first."
ERROR_EMBEDDING_MODEL = (
    "Could not load embedding model. Make sure Ollama is running and bge-m3 is available."
)


# ============================================================================
# Obsidian URI Helpers
# ============================================================================


def get_vault_name(vault_path: Path) -> str:
    """Extract vault name from vault path (last component)."""
    return vault_path.resolve().name


def build_obsidian_uri(vault_path: Path, note_path: str) -> str:
    """Build an Obsidian URI to open a note.

    Args:
        vault_path: Path to the Obsidian vault
        note_path: Relative path to the note within the vault

    Returns:
        Obsidian URI string (e.g., obsidian://open?vault=MyVault&file=folder%2Fnote.md)
    """
    vault_name = get_vault_name(vault_path)
    # URL-encode vault name and note path (safe='' to encode everything including /)
    encoded_vault = quote(vault_name, safe="")
    encoded_path = quote(note_path, safe="")
    return f"obsidian://open?vault={encoded_vault}&file={encoded_path}"


# ============================================================================
# Data Types
# ============================================================================


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with all relevant data."""

    note_id: int
    path: str
    title: str
    note_content: str
    chunk_content: str
    distance: float

    @property
    def score(self) -> float:
        """Relevance score (higher is better, 0-1 range)."""
        return 1 / (1 + self.distance)

    def preview(self, length: int = PREVIEW_LENGTH) -> str:
        """Generate a truncated preview of the chunk content."""
        text = self.chunk_content.replace("\n", " ")[:length]
        if len(self.chunk_content) > length:
            text += "..."
        return text

    @classmethod
    def from_row(cls, row: tuple) -> SearchResult:
        """Create SearchResult from a database row tuple."""
        note_id, path, title, note_content, chunk_content, distance = row
        return cls(
            note_id=note_id,
            path=path,
            title=title,
            note_content=note_content,
            chunk_content=chunk_content,
            distance=distance,
        )


# ============================================================================
# Search Result Processing
# ============================================================================


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate search results by note path, keeping the best score for each.

    When multiple chunks from the same note match, only the best-scoring chunk
    is kept. Results are returned sorted by score (best first).
    """
    seen: dict[str, SearchResult] = {}
    for result in results:
        existing = seen.get(result.path)
        if existing is None or result.distance < existing.distance:
            seen[result.path] = result
    return sorted(seen.values(), key=lambda r: r.distance)


def parse_search_results(raw_results: list[tuple]) -> list[SearchResult]:
    """Convert raw database tuples to SearchResult objects and deduplicate."""
    results = [SearchResult.from_row(row) for row in raw_results]
    return deduplicate_results(results)


# ============================================================================
# Vault Validation
# ============================================================================


class VaultError(Exception):
    """Raised when vault validation fails."""

    pass


class IndexError(Exception):
    """Raised when index is missing or invalid."""

    pass


class EmbeddingModelError(Exception):
    """Raised when embedding model is unavailable."""

    pass


def validate_vault(vault_path: Path) -> None:
    """Validate that a path is an Obsidian vault.

    Raises:
        VaultError: If the path is not a valid Obsidian vault.
    """
    if not (vault_path / ".obsidian").exists():
        raise VaultError(ERROR_NOT_OBSIDIAN_VAULT.format(path=vault_path))


def resolve_vault_path(vault: Path | None) -> Path:
    """Resolve vault path, defaulting to current directory."""
    return (vault or Path.cwd()).resolve()


# ============================================================================
# Database Context Manager
# ============================================================================


@contextmanager
def open_database(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Context manager for database connections.

    Ensures connections are properly closed even if an exception occurs.
    Also loads the sqlite-vec extension.

    Usage:
        with open_database(db_path) as conn:
            results = search_similar(conn, embedding)
    """
    from .database import init_db

    conn = init_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def require_index(db_path: Path, vault_path: Path) -> None:
    """Ensure the database index exists.

    Raises:
        IndexError: If the index doesn't exist.
    """
    if not db_path.exists():
        raise IndexError(ERROR_NO_INDEX.format(path=vault_path))


def ensure_embedding_model() -> None:
    """Ensure the embedding model is available.

    Raises:
        EmbeddingModelError: If the model cannot be loaded.
    """
    from .embeddings import ensure_model_available

    if not ensure_model_available():
        raise EmbeddingModelError(ERROR_EMBEDDING_MODEL)


# ============================================================================
# Search Operation (unified implementation)
# ============================================================================


def search_vault(
    vault_path: Path,
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[SearchResult]:
    """Perform semantic search on an indexed vault.

    This is the unified search implementation used by CLI, web app, and MCP server.

    Args:
        vault_path: Path to the Obsidian vault
        query: Search query text
        limit: Maximum number of results

    Returns:
        List of SearchResult objects, deduplicated and sorted by relevance.

    Raises:
        IndexError: If vault is not indexed
    """
    from .database import get_db_path, search_similar
    from .embeddings import get_embedding

    db_path = get_db_path(vault_path)
    require_index(db_path, vault_path)

    # Clamp limit to valid range
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    with open_database(db_path) as conn:
        query_embedding = get_embedding(query)
        raw_results = search_similar(conn, query_embedding, limit=limit)

    return parse_search_results(raw_results)


# ============================================================================
# Status Operation (unified implementation)
# ============================================================================


@dataclass(frozen=True, slots=True)
class VaultStatus:
    """Status information for an indexed vault."""

    vault_path: Path
    db_path: Path
    indexed: bool
    note_count: int = 0
    chunk_count: int = 0


def get_vault_status(vault_path: Path) -> VaultStatus:
    """Get indexing status for a vault."""
    from .database import get_chunk_count, get_db_path, get_note_count

    db_path = get_db_path(vault_path)

    if not db_path.exists():
        return VaultStatus(
            vault_path=vault_path,
            db_path=db_path,
            indexed=False,
        )

    with open_database(db_path) as conn:
        note_count = get_note_count(conn)
        chunk_count = get_chunk_count(conn)

    return VaultStatus(
        vault_path=vault_path,
        db_path=db_path,
        indexed=True,
        note_count=note_count,
        chunk_count=chunk_count,
    )
