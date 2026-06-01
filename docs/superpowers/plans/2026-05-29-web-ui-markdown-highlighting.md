# Web UI Markdown Syntax Highlighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show syntax-highlighted markdown source (not rendered HTML) when viewing `.md` files in the web UI document panel.

**Architecture:** Highlight markdown server-side in the Flask `/read` route using Pygments `MarkdownLexer` and `HtmlFormatter`. Inject Pygments token CSS into `index.html` via a Flask context processor. `.txt` and `.rst` files keep the existing plain pre-wrapped display.

**Tech Stack:** Python 3.10+, Flask, Pygments, Jinja2 templates, htmx (unchanged)

**Spec:** `docs/superpowers/specs/2026-05-29-web-ui-markdown-highlighting-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `pygments` direct dependency; bump patch version |
| `src/ragamuffin/web_app.py` | Modify | `highlight_markdown()`, Pygments CSS context processor, `/read` branching |
| `src/ragamuffin/templates/_note.html` | Modify | Render highlighted HTML for `.md`, plain text otherwise |
| `src/ragamuffin/templates/index.html` | Modify | Inject `pygments_css`; layout styles for `.note-content--md` |
| `tests/test_web_app.py` | Create | Unit tests for `highlight_markdown()` and `/read` route behavior |

---

### Task 1: Add Pygments dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `pygments` to dependencies**

In `pyproject.toml`, add `"pygments"` to the `dependencies` list (after `"flask"`):

```toml
dependencies = [
    "fastembed",
    "sqlite-vec",
    "click",
    "pyyaml",
    "rich",
    "mcp",
    "flask",
    "pygments",
]
```

- [ ] **Step 2: Sync lockfile**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv lock`

Expected: `uv.lock` updated with `pygments` as a direct dependency of `ragamuffin`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pygments as direct dependency"
```

---

### Task 2: `highlight_markdown()` helper

**Files:**
- Modify: `src/ragamuffin/web_app.py`
- Create: `tests/test_web_app.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty file).

Create `tests/test_web_app.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv run python -m unittest tests.test_web_app -v`

Expected: FAIL with `ImportError: cannot import name 'highlight_markdown'`

- [ ] **Step 3: Write minimal implementation**

Add imports and helper at the top of `src/ragamuffin/web_app.py` (after existing imports):

```python
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import MarkdownLexer

_MARKDOWN_FORMATTER = HtmlFormatter(style="friendly", nowrap=True, cssclass="highlight-md")


def highlight_markdown(content: str) -> str:
    return highlight(content, MarkdownLexer(), _MARKDOWN_FORMATTER)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv run python -m unittest tests.test_web_app.HighlightMarkdownTests -v`

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ragamuffin/web_app.py tests/__init__.py tests/test_web_app.py
git commit -m "feat(web): add highlight_markdown helper"
```

---

### Task 3: `/read` route highlights `.md` files

**Files:**
- Modify: `src/ragamuffin/web_app.py:78-105`
- Modify: `tests/test_web_app.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_app.py`:

```python
import tempfile
from pathlib import Path

from ragamuffin.web_app import create_app


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv run python -m unittest tests.test_web_app.ReadRouteTests -v`

Expected: FAIL — `highlight-md` or `note-content--md` not found in response

- [ ] **Step 3: Update `/read` route and `_note.html`**

In `src/ragamuffin/web_app.py`, replace the successful read block inside `read()`:

```python
        try:
            content = full_path.read_text(encoding="utf-8")
            title = full_path.stem
            obsidian_uri = build_obsidian_uri(dir_path, doc_path) if obsidian else None
            template_kwargs = {
                "path": doc_path,
                "title": title,
                "obsidian_uri": obsidian_uri,
                "error": None,
            }
            if full_path.suffix.lower() == ".md":
                try:
                    template_kwargs["highlighted"] = highlight_markdown(content)
                except Exception:
                    template_kwargs["content"] = content
            else:
                template_kwargs["content"] = content
            return render_template("_note.html", **template_kwargs)
```

Replace `src/ragamuffin/templates/_note.html` line 20 with:

```html
{% if highlighted is defined and highlighted %}
<div class="note-content note-content--md">{{ highlighted | safe }}</div>
{% else %}
<div class="note-content">{{ content }}</div>
{% endif %}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv run python -m unittest tests.test_web_app -v`

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ragamuffin/web_app.py src/ragamuffin/templates/_note.html tests/test_web_app.py
git commit -m "feat(web): highlight markdown source in /read route"
```

---

### Task 4: Pygments CSS and layout styles

**Files:**
- Modify: `src/ragamuffin/web_app.py` (context processor in `create_app`)
- Modify: `src/ragamuffin/templates/index.html`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_app.py`:

```python
class IndexPageTests(unittest.TestCase):
    def test_index_includes_pygments_css(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = create_app(Path(tmpdir)).test_client()
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn(".highlight-md", body)
            self.assertIn("note-content--md", body)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv run python -m unittest tests.test_web_app.IndexPageTests -v`

Expected: FAIL — `.highlight-md` not found in index HTML

- [ ] **Step 3: Add context processor and template CSS**

In `create_app()`, after `obsidian = is_obsidian_vault(dir_path)`, add:

```python
    @app.context_processor
    def inject_pygments_css():
        return {"pygments_css": _MARKDOWN_FORMATTER.get_style_defs(".highlight-md")}
```

In `src/ragamuffin/templates/index.html`, inside the existing `<style>` block before the closing `</style>`, add:

```css
    {{ pygments_css | safe }}

    .note-content--md {
      font-size: 0.9375rem;
      line-height: 1.7;
    }

    .note-content--md pre {
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      white-space: pre;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv run python -m unittest tests.test_web_app -v`

Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ragamuffin/web_app.py src/ragamuffin/templates/index.html tests/test_web_app.py
git commit -m "feat(web): inject Pygments CSS for markdown highlighting"
```

---

### Task 5: Version bump and manual verification

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump patch version**

In `pyproject.toml`, change `version = "0.9.0"` to `version = "0.9.1"`.

- [ ] **Step 2: Reinstall locally**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv tool install . --force`

Expected: Install succeeds with version 0.9.1.

- [ ] **Step 3: Manual smoke test**

Run: `muffin --directory <path-to-docs-with-md-files> web`

Verify in browser at `http://127.0.0.1:8077`:

1. Search and open a `.md` file — headings, emphasis, and code fences show distinct token colors
2. Open a `.txt` file — plain pre-wrapped text, no highlighting
3. Open a `.md` file containing `<script>` — displays as escaped text, not executed

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/mark.harnyk.ext/CODE/ragamuffin-mcp && uv run python -m unittest tests.test_web_app -v`

Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.9.1"
```
