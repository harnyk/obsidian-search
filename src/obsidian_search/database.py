"""SQLite + sqlite-vec database operations."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from .embeddings import EMBEDDING_DIM


def serialize_vector(vector: list[float]) -> bytes:
    """Serialize a vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def get_db_path(vault_path: Path) -> Path:
    """Get the database path for a vault (stored in vault's .obsidian folder)."""
    return vault_path / ".obsidian" / "obsidian-search.db"


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
) -> list[tuple[int, str, str, str, str, float]]:
    """Search for chunks similar to the query embedding.

    Returns list of (note_id, path, title, note_content, chunk_content, distance) tuples,
    ordered by distance (ascending).
    """
    cursor = conn.execute(
        """
        SELECT
            notes.id,
            notes.path,
            notes.title,
            notes.content,
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
