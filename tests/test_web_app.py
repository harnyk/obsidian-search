import tempfile
import unittest
from pathlib import Path

from ragamuffin.web_app import create_app, highlight_markdown


class HighlightMarkdownTests(unittest.TestCase):
    def test_returns_pre_with_highlight_class(self):
        result = highlight_markdown("# Title\n\n**bold**")
        self.assertIn('class="highlight-md"', result)
        self.assertIn("<pre>", result)

    def test_escapes_html_in_source(self):
        result = highlight_markdown("<script>alert(1)</script>")
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)


class ReadRouteTests(unittest.TestCase):
    def test_read_md_returns_highlighted_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "note.md").write_text("# Hello\n\n**bold**", encoding="utf-8")
            client = create_app(root).test_client()

            response = client.post("/read", data={"path": "note.md"})

            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn('class="highlight-md"', body)
            self.assertIn("note-content--md", body)

    def test_read_txt_returns_plain_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "note.txt").write_text("plain text", encoding="utf-8")
            client = create_app(root).test_client()

            response = client.post("/read", data={"path": "note.txt"})

            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn("plain text", body)
            self.assertNotIn("highlight-md", body)
            self.assertNotIn("note-content--md", body)


class IndexPageTests(unittest.TestCase):
    def test_index_includes_pygments_css(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_app(Path(tmpdir)).test_client()
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn(".highlight-md", body)
            self.assertIn("note-content--md", body)


if __name__ == "__main__":
    unittest.main()
