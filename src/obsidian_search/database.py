"""SQLite + sqlite-vec database operations."""

import sqlite3
import struct
from pathlib import Path
from typing import Optional

import sqlite_vec


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

        CREATE INDEX IF NOT EXISTS idx_notes_path ON notes(path);
        CREATE INDEX IF NOT EXISTS idx_notes_mtime ON notes(mtime);
    """)

    # Check if embeddings virtual table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    )
    if cursor.fetchone() is None:
        conn.execute("""
            CREATE VIRTUAL TABLE embeddings USING vec0(
                note_id INTEGER PRIMARY KEY,
                vector FLOAT[768]
            )
        """)

    conn.commit()
    return conn


def get_note_by_path(conn: sqlite3.Connection, path: str) -> Optional[tuple]:
    """Get a note by its path."""
    cursor = conn.execute(
        "SELECT id, path, title, content, mtime FROM notes WHERE path = ?",
        (path,)
    )
    return cursor.fetchone()


def upsert_note(
    conn: sqlite3.Connection,
    path: str,
    title: str,
    content: str,
    mtime: float,
    embedding: list[float]
) -> int:
    """Insert or update a note and its embedding."""
    existing = get_note_by_path(conn, path)

    if existing:
        note_id = existing[0]
        conn.execute(
            "UPDATE notes SET title = ?, content = ?, mtime = ? WHERE id = ?",
            (title, content, mtime, note_id)
        )
        # Delete old embedding and insert new one
        conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
    else:
        cursor = conn.execute(
            "INSERT INTO notes (path, title, content, mtime) VALUES (?, ?, ?, ?)",
            (path, title, content, mtime)
        )
        note_id = cursor.lastrowid

    # Insert embedding
    conn.execute(
        "INSERT INTO embeddings (note_id, vector) VALUES (?, ?)",
        (note_id, serialize_vector(embedding))
    )

    conn.commit()
    return note_id


def search_similar(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 10
) -> list[tuple[int, str, str, str, float]]:
    """Search for notes similar to the query embedding.

    Returns list of (id, path, title, content, distance) tuples.
    """
    cursor = conn.execute(
        """
        SELECT
            notes.id,
            notes.path,
            notes.title,
            notes.content,
            embeddings.distance
        FROM embeddings
        JOIN notes ON notes.id = embeddings.note_id
        WHERE vector MATCH ?
            AND k = ?
        ORDER BY distance
        """,
        (serialize_vector(query_embedding), limit)
    )
    return cursor.fetchall()


def get_all_notes_mtime(conn: sqlite3.Connection) -> dict[str, float]:
    """Get all notes paths and their modification times."""
    cursor = conn.execute("SELECT path, mtime FROM notes")
    return {row[0]: row[1] for row in cursor.fetchall()}


def delete_note(conn: sqlite3.Connection, path: str) -> None:
    """Delete a note and its embedding by path."""
    note = get_note_by_path(conn, path)
    if note:
        note_id = note[0]
        conn.execute("DELETE FROM embeddings WHERE note_id = ?", (note_id,))
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()


def get_note_count(conn: sqlite3.Connection) -> int:
    """Get the total number of indexed notes."""
    cursor = conn.execute("SELECT COUNT(*) FROM notes")
    return cursor.fetchone()[0]
