"""Minimal web app server for searching and viewing documents."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from .core import (
    DEFAULT_SEARCH_LIMIT,
    EmbeddingModelError,
    IndexError,
    MAX_SEARCH_LIMIT,
    build_obsidian_uri,
    ensure_embedding_model,
    get_index_status,
    is_obsidian_vault,
    search_dir,
)


def _is_safe_path(path: Path, base: Path) -> bool:
    """Check if path is safely within base directory (prevents path traversal)."""
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def create_app(dir_path: Path) -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__, template_folder="templates")
    app.config["DIR_PATH"] = dir_path
    obsidian = is_obsidian_vault(dir_path)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/status")
    def status():
        s = get_index_status(dir_path)
        return render_template("_status.html", indexed=s.indexed, doc_count=s.doc_count)

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
            return render_template("_results.html", results=[], error="Embedding model unavailable. Check fastembed installation.")

        try:
            results = search_dir(dir_path, query, limit=limit)
            output = [
                {
                    "title": r.title,
                    "path": r.path,
                    "score": round(r.score, 4),
                    "preview": r.preview(),
                    "obsidian_uri": build_obsidian_uri(dir_path, r.path) if obsidian else None,
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
        doc_path = request.form.get("path", "")
        if not doc_path:
            return render_template("_note.html", error="Path is required.")

        full_path = (dir_path / doc_path).resolve()

        if not _is_safe_path(full_path, dir_path):
            return render_template("_note.html", error="Invalid path.")

        if not full_path.exists() or not full_path.is_file():
            return render_template("_note.html", error="Document not found.")

        try:
            content = full_path.read_text(encoding="utf-8")
            title = full_path.stem
            obsidian_uri = build_obsidian_uri(dir_path, doc_path) if obsidian else None
            return render_template(
                "_note.html",
                path=doc_path,
                title=title,
                content=content,
                obsidian_uri=obsidian_uri,
                error=None,
            )
        except Exception as exc:
            return render_template("_note.html", error=f"Read failed: {exc}")

    return app


def run_web_app(dir_path: Path, host: str = "127.0.0.1", port: int = 8077) -> None:
    """Run the web app server."""
    app = create_app(dir_path)
    app.run(host=host, port=port, debug=False)
