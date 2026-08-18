"""Core types, constants, and shared utilities for ragamuffin."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from urllib.parse import quote

if TYPE_CHECKING:
    from .database import ChunkWindow

# ============================================================================
# Constants
# ============================================================================

DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50
PREVIEW_LENGTH = 200
DEFAULT_WEB_PORT = 8077

ERROR_NO_INDEX = "No index found for {path}. Run 'muffin index' first."
ERROR_EMBEDDING_MODEL = (
    "Could not load embedding model. "
    "Check that fastembed is installed and you have an internet connection for the first download."
)


# ============================================================================
# Obsidian Detection & URI Helpers
# ============================================================================


def is_obsidian_vault(dir_path: Path) -> bool:
    """Return True if the directory is an Obsidian vault (has a .obsidian folder)."""
    return (dir_path / ".obsidian").is_dir()


def get_vault_name(dir_path: Path) -> str:
    """Extract vault name from path (last component)."""
    return dir_path.resolve().name


def build_obsidian_uri(dir_path: Path, note_path: str) -> str:
    """Build an Obsidian URI to open a note.

    Only call this when is_obsidian_vault() is True.

    Returns:
        Obsidian URI string (e.g., obsidian://open?vault=MyVault&file=folder%2Fnote.md)
    """
    vault_name = get_vault_name(dir_path)
    encoded_vault = quote(vault_name, safe="")
    encoded_path = quote(note_path, safe="")
    return f"obsidian://open?vault={encoded_vault}&file={encoded_path}"


# ============================================================================
# Data Types
# ============================================================================


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with all relevant data."""

    chunk_id: int
    note_id: int
    path: str
    title: str
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
        chunk_id, note_id, path, title, chunk_content, distance = row
        return cls(
            chunk_id=chunk_id,
            note_id=note_id,
            path=path,
            title=title,
            chunk_content=chunk_content,
            distance=distance,
        )


# ============================================================================
# Search Result Processing
# ============================================================================


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate search results by path, keeping the best score for each.

    When multiple chunks from the same document match, only the best-scoring
    chunk is kept. Results are returned sorted by score (best first).
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
# Exceptions
# ============================================================================


class IndexError(Exception):
    """Raised when index is missing or invalid."""

    pass


class EmbeddingModelError(Exception):
    """Raised when embedding model is unavailable."""

    pass


def resolve_dir_path(directory: Path | None) -> Path:
    """Resolve directory path, defaulting to current directory."""
    return (directory or Path.cwd()).resolve()


# ============================================================================
# Database Context Manager
# ============================================================================


@contextmanager
def open_database(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Context manager for database connections.

    Ensures connections are properly closed even if an exception occurs.
    Also loads the sqlite-vec extension.
    """
    from .database import init_db

    conn = init_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def require_index(db_path: Path, dir_path: Path) -> None:
    """Ensure the database index exists.

    Raises:
        IndexError: If the index doesn't exist.
    """
    if not db_path.exists():
        raise IndexError(ERROR_NO_INDEX.format(path=dir_path))


def ensure_embedding_model() -> None:
    """Ensure the embedding model is available.

    Raises:
        EmbeddingModelError: If the model cannot be loaded.
    """
    from .embeddings import ensure_model_available

    if not ensure_model_available():
        raise EmbeddingModelError(ERROR_EMBEDDING_MODEL)


# ============================================================================
# Search Operation
# ============================================================================


def search_dir(
    dir_path: Path,
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[SearchResult]:
    """Perform semantic search on an indexed directory.

    Unified search implementation used by CLI, web app, and MCP server.

    Raises:
        IndexError: If the directory is not indexed.
    """
    from .database import get_db_path, search_similar
    from .embeddings import get_embedding

    db_path = get_db_path(dir_path)
    require_index(db_path, dir_path)

    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    with open_database(db_path) as conn:
        query_embedding = get_embedding(query)
        raw_results = search_similar(conn, query_embedding, limit=limit)

    return parse_search_results(raw_results)


def read_chunk_window(
    dir_path: Path,
    chunk_id: int,
    before: int = 0,
    after: int = 0,
) -> ChunkWindow:
    """Read a chunk window from the index by opaque chunk_id.

    Raises:
        IndexError: If the directory is not indexed or chunk_id is unknown.
    """
    from .database import get_chunk_window, get_db_path

    db_path = get_db_path(dir_path)
    require_index(db_path, dir_path)

    with open_database(db_path) as conn:
        window = get_chunk_window(conn, chunk_id, before=before, after=after)

    if window is None:
        raise IndexError(
            f"Chunk {chunk_id} not found. Re-run search — ids are invalid after reindex."
        )
    return window


# ============================================================================
# Status Operation
# ============================================================================


@dataclass(frozen=True, slots=True)
class IndexStatus:
    """Status information for an indexed directory."""

    dir_path: Path
    db_path: Path
    indexed: bool
    doc_count: int = 0
    chunk_count: int = 0


def get_index_status(dir_path: Path) -> IndexStatus:
    """Get indexing status for a directory."""
    from .database import get_chunk_count, get_db_path, get_note_count

    db_path = get_db_path(dir_path)

    if not db_path.exists():
        return IndexStatus(
            dir_path=dir_path,
            db_path=db_path,
            indexed=False,
        )

    with open_database(db_path) as conn:
        doc_count = get_note_count(conn)
        chunk_count = get_chunk_count(conn)

    return IndexStatus(
        dir_path=dir_path,
        db_path=db_path,
        indexed=True,
        doc_count=doc_count,
        chunk_count=chunk_count,
    )
