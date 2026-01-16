"""Minimal web app server for searching and viewing Obsidian notes."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .database import get_db_path, init_db, search_similar
from .embeddings import ensure_model_available, get_embedding

HTML_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Obsidian Search</title>
    <style>
      :root {
        --bg-1: #0b0b0f;
        --bg-2: #1b1f2a;
        --bg-3: #151821;
        --ink: #e8e6e3;
        --muted: #a5adba;
        --accent: #f5b85b;
        --accent-2: #5bd2f5;
        --card: rgba(255, 255, 255, 0.06);
        --card-strong: rgba(255, 255, 255, 0.12);
        --border: rgba(255, 255, 255, 0.16);
        --shadow: 0 22px 50px rgba(0, 0, 0, 0.35);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "Charter", "Iowan Old Style", "Palatino Linotype", "Palatino", serif;
        color: var(--ink);
        background:
          radial-gradient(1200px 500px at 70% -10%, rgba(91, 210, 245, 0.25), transparent 70%),
          radial-gradient(900px 600px at 10% -10%, rgba(245, 184, 91, 0.25), transparent 70%),
          linear-gradient(160deg, var(--bg-1), var(--bg-2));
        min-height: 100vh;
      }

      .app {
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 20px 40px;
        display: flex;
        flex-direction: column;
        gap: 18px;
        animation: rise 0.5s ease;
      }

      header {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      h1 {
        font-size: clamp(1.8rem, 3vw, 2.8rem);
        letter-spacing: 0.02em;
        margin: 0;
      }

      .subtitle {
        color: var(--muted);
        font-size: 0.95rem;
      }

      .search-bar {
        display: flex;
        gap: 12px;
        align-items: center;
        background: var(--card);
        border: 1px solid var(--border);
        padding: 12px;
        border-radius: 16px;
        box-shadow: var(--shadow);
      }

      .search-bar input {
        flex: 1;
        background: transparent;
        border: none;
        color: var(--ink);
        font-size: 1rem;
        outline: none;
      }

      .search-bar button {
        background: linear-gradient(120deg, var(--accent), var(--accent-2));
        border: none;
        color: #111317;
        font-weight: 600;
        padding: 10px 18px;
        border-radius: 12px;
        cursor: pointer;
      }

      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr);
        gap: 18px;
        min-height: 60vh;
      }

      .panel {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 16px;
        box-shadow: var(--shadow);
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .panel h2 {
        margin: 0;
        font-size: 1.1rem;
      }

      .status {
        color: var(--muted);
        font-size: 0.9rem;
      }

      .results {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .result {
        background: var(--card-strong);
        border: 1px solid transparent;
        padding: 12px;
        border-radius: 12px;
        text-align: left;
        cursor: pointer;
        color: var(--ink);
        transition: border 0.2s ease, transform 0.2s ease;
        animation: fadeIn 0.3s ease;
      }

      .result:hover {
        border-color: var(--accent-2);
        transform: translateY(-2px);
      }

      .result .title {
        font-weight: 600;
        margin-bottom: 4px;
      }

      .result .meta {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 6px;
      }

      .note-header {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .note-path {
        font-size: 0.85rem;
        color: var(--muted);
        word-break: break-word;
      }

      .note-content {
        white-space: pre-wrap;
        line-height: 1.5;
        color: #f4f1ed;
      }

      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(14px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @media (max-width: 900px) {
        .layout {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="app">
      <header>
        <h1>Obsidian Search</h1>
        <div class="subtitle">Semantic search and quick view for your vault notes.</div>
      </header>

      <div class="search-bar">
        <input id="query" type="text" placeholder="Search your notes..." />
        <button id="searchBtn">Search</button>
      </div>

      <div class="layout">
        <section class="panel">
          <h2>Results</h2>
          <div id="status" class="status">Ready.</div>
          <div id="results" class="results"></div>
        </section>

        <section class="panel">
          <div class="note-header">
            <h2 id="noteTitle">Note</h2>
            <div id="notePath" class="note-path">Select a result to view the note.</div>
          </div>
          <div id="noteContent" class="note-content"></div>
        </section>
      </div>
    </div>

    <script>
      const statusEl = document.getElementById("status");
      const resultsEl = document.getElementById("results");
      const noteTitleEl = document.getElementById("noteTitle");
      const notePathEl = document.getElementById("notePath");
      const noteContentEl = document.getElementById("noteContent");
      const queryEl = document.getElementById("query");
      const searchBtn = document.getElementById("searchBtn");

      function setStatus(message) {
        statusEl.textContent = message;
      }

      function renderResults(items) {
        resultsEl.innerHTML = "";
        if (!items.length) {
          resultsEl.innerHTML = "<div class=\\"status\\">No results yet.</div>";
          return;
        }
        items.forEach((item) => {
          const button = document.createElement("button");
          button.className = "result";
          button.type = "button";
          const title = document.createElement("div");
          title.className = "title";
          title.textContent = item.title || "(untitled)";
          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent = item.path;
          const preview = document.createElement("div");
          preview.textContent = item.preview;
          button.appendChild(title);
          button.appendChild(meta);
          button.appendChild(preview);
          button.addEventListener("click", () => readNote(item.path, item.title));
          resultsEl.appendChild(button);
        });
      }

      async function apiPost(path, body) {
        const response = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Request failed");
        }
        return payload;
      }

      async function runSearch() {
        const query = queryEl.value.trim();
        if (!query) {
          setStatus("Enter a search query.");
          return;
        }
        setStatus("Searching...");
        try {
          const data = await apiPost("/api/search", { query });
          renderResults(data.results || []);
          setStatus(`Found ${data.results.length} result(s).`);
        } catch (error) {
          setStatus(error.message);
        }
      }

      async function readNote(path, title) {
        setStatus("Loading note...");
        try {
          const data = await apiPost("/api/read", { path });
          noteTitleEl.textContent = title || "Note";
          notePathEl.textContent = data.path;
          noteContentEl.textContent = data.content;
          setStatus("Note loaded.");
        } catch (error) {
          setStatus(error.message);
        }
      }

      async function loadStatus() {
        try {
          const response = await fetch("/api/status");
          const data = await response.json();
          if (data.indexed) {
            setStatus(`Indexed: ${data.note_count} notes.`);
          } else {
            setStatus("Index missing. Run obsidian-search index first.");
          }
        } catch (error) {
          setStatus("Status unavailable.");
        }
      }

      searchBtn.addEventListener("click", runSearch);
      queryEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          runSearch();
        }
      });

      loadStatus();
    </script>
  </body>
</html>
"""


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


class WebAppServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], vault_path: Path):
        self.vault_path = vault_path
        self.db_path = get_db_path(vault_path)
        super().__init__(server_address, WebAppHandler)


class WebAppHandler(BaseHTTPRequestHandler):
    server: WebAppServer

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return None
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html()
            return
        if parsed.path == "/api/status":
            db_exists = self.server.db_path.exists()
            note_count = 0
            if db_exists:
                conn = init_db(self.server.db_path)
                note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                conn.close()
            self._send_json(200, {
                "indexed": db_exists,
                "note_count": note_count,
            })
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/api/search":
            payload = self._read_json() or {}
            query = payload.get("query", "")
            limit = payload.get("limit", 10)
            if not isinstance(query, str) or not query.strip():
                self._send_json(400, {"error": "Query is required."})
                return
            if not self.server.db_path.exists():
                self._send_json(400, {"error": "Index missing. Run obsidian-search index first."})
                return
            if not ensure_model_available():
                self._send_json(500, {"error": "Embedding model unavailable. Is Ollama running?"})
                return
            try:
                if not isinstance(limit, int):
                    limit = 10
                limit = max(1, min(limit, 50))
                conn = init_db(self.server.db_path)
                query_embedding = get_embedding(query.strip())
                results = search_similar(conn, query_embedding, limit=limit)
                conn.close()
                if not results:
                    self._send_json(200, {"results": []})
                    return
                seen_paths: dict[str, tuple] = {}
                for result in results:
                    note_id, path, title, note_content, chunk_content, distance = result
                    if path not in seen_paths or distance < seen_paths[path][5]:
                        seen_paths[path] = result
                deduplicated = sorted(seen_paths.values(), key=lambda x: x[5])
                output = []
                for note_id, path, title, note_content, chunk_content, distance in deduplicated:
                    score = 1 / (1 + distance)
                    preview = chunk_content.replace("\n", " ")[:200]
                    if len(chunk_content) > 200:
                        preview += "..."
                    output.append({
                        "title": title,
                        "path": path,
                        "score": round(score, 4),
                        "preview": preview,
                    })
                self._send_json(200, {"results": output})
                return
            except Exception as exc:
                self._send_json(500, {"error": f"Search failed: {exc}"})
                return

        if self.path == "/api/read":
            payload = self._read_json() or {}
            note_path = payload.get("path", "")
            if not isinstance(note_path, str) or not note_path:
                self._send_json(400, {"error": "Path is required."})
                return
            try:
                full_path = (self.server.vault_path / note_path).resolve()
                if not _is_relative_to(full_path, self.server.vault_path):
                    self._send_json(400, {"error": "Invalid path."})
                    return
                if not full_path.exists() or not full_path.is_file():
                    self._send_json(404, {"error": "Note not found."})
                    return
                content = full_path.read_text(encoding="utf-8")
                self._send_json(200, {
                    "path": note_path,
                    "content": content,
                })
                return
            except Exception as exc:
                self._send_json(500, {"error": f"Read failed: {exc}"})
                return

        self._send_json(404, {"error": "Not found"})


def run_web_app(vault_path: Path, host: str = "127.0.0.1", port: int = 8077) -> None:
    """Run the web app server."""
    server = WebAppServer((host, port), vault_path)
    server.serve_forever()
