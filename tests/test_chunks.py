"""Tests for chunk_id search results and chunk window reading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ragamuffin.core import SearchResult, parse_search_results
from ragamuffin.database import (
    get_chunk_window,
    init_db,
    search_similar,
    serialize_vector,
)


def _vec(seed: float) -> list[float]:
    return [seed] * 768


def _insert_note_with_chunks(conn, path: str, title: str, chunk_texts: list[str]) -> list[int]:
    cursor = conn.execute(
        "INSERT INTO notes (path, title, content, mtime) VALUES (?, ?, ?, ?)",
        (path, title, "\n\n".join(chunk_texts), 1.0),
    )
    note_id = cursor.lastrowid
    chunk_ids = []
    for i, text in enumerate(chunk_texts):
        cursor = conn.execute(
            "INSERT INTO chunks (note_id, chunk_index, content) VALUES (?, ?, ?)",
            (note_id, i, text),
        )
        chunk_id = cursor.lastrowid
        chunk_ids.append(chunk_id)
        conn.execute(
            "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
            (chunk_id, serialize_vector(_vec(float(i + 1)))),
        )
    conn.commit()
    return chunk_ids


class GetChunkWindowTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "index.db"
        self.conn = init_db(self.db_path)
        self.chunk_ids = _insert_note_with_chunks(
            self.conn,
            "book.md",
            "Book",
            ["chunk-zero", "chunk-one", "chunk-two", "chunk-three"],
        )

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    def test_window_around_middle_chunk(self):
        window = get_chunk_window(self.conn, self.chunk_ids[2], before=1, after=1)

        self.assertIsNotNone(window)
        self.assertEqual(window.path, "book.md")
        self.assertEqual(window.title, "Book")
        self.assertEqual(window.focus_id, self.chunk_ids[2])
        self.assertEqual(
            [(c.chunk_id, c.content) for c in window.chunks],
            [
                (self.chunk_ids[1], "chunk-one"),
                (self.chunk_ids[2], "chunk-two"),
                (self.chunk_ids[3], "chunk-three"),
            ],
        )
        self.assertEqual(window.prev_id, self.chunk_ids[1])
        self.assertEqual(window.next_id, self.chunk_ids[3])

    def test_edges_have_null_neighbors(self):
        first = get_chunk_window(self.conn, self.chunk_ids[0], before=0, after=0)
        last = get_chunk_window(self.conn, self.chunk_ids[-1], before=0, after=0)

        self.assertIsNone(first.prev_id)
        self.assertEqual(first.next_id, self.chunk_ids[1])
        self.assertEqual(last.prev_id, self.chunk_ids[-2])
        self.assertIsNone(last.next_id)

    def test_missing_chunk_returns_none(self):
        self.assertIsNone(get_chunk_window(self.conn, 99999, before=0, after=0))


class SearchSimilarChunkIdTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "index.db"
        self.conn = init_db(self.db_path)
        self.chunk_ids = _insert_note_with_chunks(
            self.conn,
            "doc.md",
            "Doc",
            ["alpha content here", "beta content here"],
        )

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    def test_search_returns_chunk_id_not_note_content(self):
        rows = search_similar(self.conn, _vec(1.0), limit=5)
        self.assertTrue(rows)
        # (chunk_id, note_id, path, title, chunk_content, distance)
        chunk_id, note_id, path, title, chunk_content, distance = rows[0]
        self.assertEqual(chunk_id, self.chunk_ids[0])
        self.assertEqual(path, "doc.md")
        self.assertEqual(title, "Doc")
        self.assertEqual(chunk_content, "alpha content here")
        self.assertIsInstance(distance, float)

    def test_parse_search_results_includes_chunk_id(self):
        rows = search_similar(self.conn, _vec(2.0), limit=5)
        results = parse_search_results(rows)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].chunk_id, self.chunk_ids[1])
        self.assertEqual(results[0].chunk_content, "beta content here")
        self.assertFalse(hasattr(results[0], "note_content"))


if __name__ == "__main__":
    unittest.main()
