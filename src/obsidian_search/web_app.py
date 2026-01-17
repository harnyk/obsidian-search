"""Minimal web app server for searching and viewing Obsidian notes."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from .core import (
    DEFAULT_SEARCH_LIMIT,
    EmbeddingModelError,
    IndexError,
    MAX_SEARCH_LIMIT,
    ensure_embedding_model,
    get_vault_status,
    search_vault,
)


def _is_safe_path(path: Path, base: Path) -> bool:
    """Check if path is safely within base directory (prevents path traversal)."""
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def create_app(vault_path: Path) -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__, template_folder="templates")
    app.config["VAULT_PATH"] = vault_path

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/status")
    def status():
        status = get_vault_status(vault_path)
        return render_template("_status.html", indexed=status.indexed, note_count=status.note_count)

    @app.post("/search")
    def search():
        query = request.form.get("query", "").strip()
        if not query:
            return render_template("_results.html", results=[], error=None)

        limit = request.form.get("limit", DEFAULT_SEARCH_LIMIT, type=int)
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))

        try:
            ensure_embedding_model()
        except EmbeddingModelError:
            return render_template("_results.html", results=[], error="Embedding model unavailable. Is Ollama running?")

        try:
            results = search_vault(vault_path, query, limit=limit)
            output = [
                {
                    "title": r.title,
                    "path": r.path,
                    "score": round(r.score, 4),
                    "preview": r.preview(),
                }
                for r in results
            ]
            return render_template("_results.html", results=output, error=None)
        except IndexError as e:
            return render_template("_results.html", results=[], error=str(e))
        except Exception as exc:
            return render_template("_results.html", results=[], error=f"Search failed: {exc}")

    @app.post("/read")
    def read():
        note_path = request.form.get("path", "")
        if not note_path:
            return render_template("_note.html", error="Path is required.")

        full_path = (vault_path / note_path).resolve()

        if not _is_safe_path(full_path, vault_path):
            return render_template("_note.html", error="Invalid path.")

        if not full_path.exists() or not full_path.is_file():
            return render_template("_note.html", error="Note not found.")

        try:
            content = full_path.read_text(encoding="utf-8")
            title = full_path.stem
            return render_template("_note.html", path=note_path, title=title, content=content, error=None)
        except Exception as exc:
            return render_template("_note.html", error=f"Read failed: {exc}")

    return app


def run_web_app(vault_path: Path, host: str = "127.0.0.1", port: int = 8077) -> None:
    """Run the web app server."""
    app = create_app(vault_path)
    app.run(host=host, port=port, debug=False)
