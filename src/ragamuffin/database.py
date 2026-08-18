"""SQLite + sqlite-vec database operations."""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from .embeddings import EMBEDDING_DIM


@dataclass(frozen=True, slots=True)
class ChunkPiece:
    chunk_id: int
    content: str


@dataclass(frozen=True, slots=True)
class ChunkWindow:
    path: str
    title: str
    focus_id: int
    chunks: tuple[ChunkPiece, ...]
    prev_id: int | None
    next_id: int | None


def serialize_vector(vector: list[float]) -> bytes:
    """Serialize a vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def get_db_path(dir_path: Path) -> Path:
    """Get the database path for a directory (stored in <dir>/.ragamuffin/index.db)."""
    return dir_path / ".ragamuffin" / "index.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database with required tables and sqlite-vec extension."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            title TEXT,
            content TEXT,
            mtime REAL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            note_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE,
            UNIQUE(note_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_notes_path ON notes(path);
        CREATE INDEX IF NOT EXISTS idx_notes_mtime ON notes(mtime);
        CREATE INDEX IF NOT EXISTS idx_chunks_note_id ON chunks(note_id);
    """)

    # Create embeddings virtual table if it doesn't exist
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    )
    if cursor.fetchone() is None:
        conn.execute(f"""
            CREATE VIRTUAL TABLE embeddings USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                vector FLOAT[{EMBEDDING_DIM}]
            )
        """)

    conn.commit()
    return conn


def get_note_by_path(conn: sqlite3.Connection, path: str) -> tuple[int, str, str, str, float] | None:
    """Get a note by its path.

    Returns (id, path, title, content, mtime) or None if not found.
    """
    cursor = conn.execute(
        "SELECT id, path, title, content, mtime FROM notes WHERE path = ?",
        (path,)
    )
    return cursor.fetchone()


def delete_note_chunks(conn: sqlite3.Connection, note_id: int) -> None:
    """Delete all chunks and their embeddings for a note."""
    cursor = conn.execute("SELECT id FROM chunks WHERE note_id = ?", (note_id,))
    chunk_ids = [row[0] for row in cursor.fetchall()]

    for chunk_id in chunk_ids:
        conn.execute("DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,))

    conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))


def upsert_note(
    conn: sqlite3.Connection,
    path: str,
    title: str,
    content: str,
    mtime: float,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """Insert or update a note and its chunk embeddings.

    Returns the note ID.
    """
    existing = get_note_by_path(conn, path)

    if existing:
        note_id = existing[0]
        conn.execute(
            "UPDATE notes SET title = ?, content = ?, mtime = ? WHERE id = ?",
            (title, content, mtime, note_id)
        )
        delete_note_chunks(conn, note_id)
    else:
        cursor = conn.execute(
            "INSERT INTO notes (path, title, content, mtime) VALUES (?, ?, ?, ?)",
            (path, title, content, mtime)
        )
        note_id = cursor.lastrowid

    # Insert chunks and embeddings
    for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
        cursor = conn.execute(
            "INSERT INTO chunks (note_id, chunk_index, content) VALUES (?, ?, ?)",
            (note_id, i, chunk_content)
        )
        chunk_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
            (chunk_id, serialize_vector(embedding))
        )

    conn.commit()
    return note_id


def search_similar(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 10,
) -> list[tuple[int, int, str, str, str, float]]:
    """Search for chunks similar to the query embedding.

    Returns list of (chunk_id, note_id, path, title, chunk_content, distance) tuples,
    ordered by distance (ascending).
    """
    cursor = conn.execute(
        """
        SELECT
            chunks.id,
            notes.id,
            notes.path,
            notes.title,
            chunks.content,
            embeddings.distance
        FROM embeddings
        JOIN chunks ON chunks.id = embeddings.chunk_id
        JOIN notes ON notes.id = chunks.note_id
        WHERE vector MATCH ?
            AND k = ?
        ORDER BY distance
        """,
        (serialize_vector(query_embedding), limit)
    )
    return cursor.fetchall()


def get_chunk_window(
    conn: sqlite3.Connection,
    chunk_id: int,
    before: int = 0,
    after: int = 0,
) -> ChunkWindow | None:
    """Load a chunk and an optional neighborhood in the same document.

    ``prev_id`` / ``next_id`` are the immediate neighbors of the focus chunk
    (not of the window edges). Chunk ids are opaque; do not infer document
    order from numeric id values.
    """
    before = max(0, before)
    after = max(0, after)

    cursor = conn.execute(
        """
        SELECT chunks.id, chunks.note_id, chunks.chunk_index, chunks.content,
               notes.path, notes.title
        FROM chunks
        JOIN notes ON notes.id = chunks.note_id
        WHERE chunks.id = ?
        """,
        (chunk_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None

    focus_id, note_id, focus_index, _focus_content, path, title = row

    cursor = conn.execute(
        """
        SELECT id, chunk_index, content
        FROM chunks
        WHERE note_id = ?
          AND chunk_index BETWEEN ? AND ?
        ORDER BY chunk_index
        """,
        (note_id, focus_index - before, focus_index + after),
    )
    pieces = tuple(
        ChunkPiece(chunk_id=r[0], content=r[2]) for r in cursor.fetchall()
    )

    cursor = conn.execute(
        """
        SELECT id FROM chunks
        WHERE note_id = ? AND chunk_index = ?
        """,
        (note_id, focus_index - 1),
    )
    prev_row = cursor.fetchone()
    cursor = conn.execute(
        """
        SELECT id FROM chunks
        WHERE note_id = ? AND chunk_index = ?
        """,
        (note_id, focus_index + 1),
    )
    next_row = cursor.fetchone()

    return ChunkWindow(
        path=path,
        title=title or "",
        focus_id=focus_id,
        chunks=pieces,
        prev_id=prev_row[0] if prev_row else None,
        next_id=next_row[0] if next_row else None,
    )


def get_all_notes_mtime(conn: sqlite3.Connection) -> dict[str, float]:
    """Get all note paths and their modification times."""
    cursor = conn.execute("SELECT path, mtime FROM notes")
    return {row[0]: row[1] for row in cursor.fetchall()}


def delete_note(conn: sqlite3.Connection, path: str) -> None:
    """Delete a note and its chunks/embeddings by path."""
    note = get_note_by_path(conn, path)
    if note:
        note_id = note[0]
        delete_note_chunks(conn, note_id)
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()


def get_note_count(conn: sqlite3.Connection) -> int:
    """Get the total number of indexed notes."""
    cursor = conn.execute("SELECT COUNT(*) FROM notes")
    return cursor.fetchone()[0]


def get_chunk_count(conn: sqlite3.Connection) -> int:
    """Get the total number of indexed chunks."""
    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
    return cursor.fetchone()[0]
