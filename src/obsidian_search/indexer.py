"""Vault scanning and indexing logic."""

from pathlib import Path
from typing import Callable, Iterator

from .database import (
    delete_note,
    get_all_notes_mtime,
    get_db_path,
    init_db,
    upsert_note,
)
from .embeddings import get_embeddings_batch
from .parser import parse_note


def scan_vault(vault_path: Path) -> Iterator[Path]:
    """Recursively scan vault for markdown files, skipping hidden files/folders."""
    for item in vault_path.rglob("*.md"):
        # Skip hidden files and files in hidden directories
        parts = item.relative_to(vault_path).parts
        if any(part.startswith(".") for part in parts):
            continue
        yield item


def get_files_to_index(
    vault_path: Path,
    existing_mtimes: dict[str, float],
    update_only: bool = False,
) -> tuple[list[Path], list[str]]:
    """Determine which files need to be indexed and which should be deleted.

    Returns (files_to_index, paths_to_delete).
    """
    current_files: dict[str, Path] = {}

    for file_path in scan_vault(vault_path):
        rel_path = str(file_path.relative_to(vault_path))
        current_files[rel_path] = file_path

    files_to_index: list[Path] = []
    paths_to_delete: list[str] = []

    # Find files to index
    for rel_path, file_path in current_files.items():
        mtime = file_path.stat().st_mtime

        if not update_only:
            # Full re-index: index everything
            files_to_index.append(file_path)
        elif rel_path not in existing_mtimes:
            # New file
            files_to_index.append(file_path)
        elif mtime > existing_mtimes[rel_path]:
            # Modified file
            files_to_index.append(file_path)

    # Find deleted files (only matters for update mode, but always clean up)
    for rel_path in existing_mtimes:
        if rel_path not in current_files:
            paths_to_delete.append(rel_path)

    return files_to_index, paths_to_delete


def index_vault(
    vault_path: Path,
    update_only: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[int, int, int]:
    """Index a vault into the database.

    Args:
        vault_path: Path to the Obsidian vault
        update_only: If True, only index new/modified files
        progress_callback: Optional callback(status, current, total) for progress updates

    Returns:
        Tuple of (indexed_count, skipped_count, deleted_count)
    """
    vault_path = vault_path.resolve()
    db_path = get_db_path(vault_path)
    conn = init_db(db_path)

    existing_mtimes = get_all_notes_mtime(conn) if update_only else {}
    files_to_index, paths_to_delete = get_files_to_index(
        vault_path, existing_mtimes, update_only
    )

    total = len(files_to_index)
    indexed = 0
    skipped = 0

    # Delete removed files
    for rel_path in paths_to_delete:
        delete_note(conn, rel_path)

    # Index files
    for i, file_path in enumerate(files_to_index):
        rel_path = str(file_path.relative_to(vault_path))

        if progress_callback:
            progress_callback(f"Indexing: {rel_path}", i + 1, total)

        try:
            note = parse_note(file_path)
            mtime = file_path.stat().st_mtime

            # Skip notes with no chunks
            if not note.chunks:
                skipped += 1
                continue

            # Generate embeddings for all chunks
            embeddings = get_embeddings_batch(note.chunks)

            # Store in database
            upsert_note(
                conn,
                path=rel_path,
                title=note.title,
                content=note.content,
                mtime=mtime,
                chunks=note.chunks,
                embeddings=embeddings,
            )
            indexed += 1

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error indexing {rel_path}: {e}", i + 1, total)
            skipped += 1

    conn.close()
    return indexed, skipped, len(paths_to_delete)
