import unittest

from ragamuffin.web_app import highlight_markdown


class HighlightMarkdownTests(unittest.TestCase):
    def test_returns_pre_with_highlight_class(self):
        result = highlight_markdown("# Title\n\n**bold**")
        self.assertIn('class="highlight-md"', result)
        self.assertIn("<pre>", result)

    def test_escapes_html_in_source(self):
        result = highlight_markdown("<script>alert(1)</script>")
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)


if __name__ == "__main__":
    unittest.main()
