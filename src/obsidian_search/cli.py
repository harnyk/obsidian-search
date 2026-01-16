"""CLI commands for obsidian-search."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .database import get_db_path, get_note_count, init_db, search_similar
from .embeddings import ensure_model_available, get_embedding
from .indexer import index_vault

console = Console()


@click.group()
def cli():
    """Obsidian vault semantic search using Ollama embeddings."""
    pass


@cli.command()
@click.argument("vault_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--update", is_flag=True, help="Only index new/modified files")
def index(vault_path: Path, update: bool):
    """Index an Obsidian vault for semantic search."""
    vault_path = vault_path.resolve()

    if not (vault_path / ".obsidian").exists():
        console.print(
            f"[red]Error:[/red] {vault_path} does not appear to be an Obsidian vault "
            "(missing .obsidian folder)"
        )
        raise SystemExit(1)

    console.print(f"[blue]Vault:[/blue] {vault_path}")

    # Ensure embedding model is available
    with console.status("Checking embedding model..."):
        if not ensure_model_available():
            console.print(
                "[red]Error:[/red] Could not load embedding model. "
                "Make sure Ollama is running and nomic-embed-text is available."
            )
            raise SystemExit(1)

    console.print("[green]Embedding model ready[/green]")

    # Index the vault
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting indexing...", total=None)

        def update_progress(status: str, current: int, total: int):
            progress.update(task, description=f"[{current}/{total}] {status}")

        indexed, skipped, deleted = index_vault(
            vault_path,
            update_only=update,
            progress_callback=update_progress,
        )

    console.print()
    console.print(f"[green]Indexed:[/green] {indexed} notes")
    if skipped:
        console.print(f"[yellow]Skipped:[/yellow] {skipped} notes")
    if deleted:
        console.print(f"[red]Deleted:[/red] {deleted} notes (removed from vault)")

    db_path = get_db_path(vault_path)
    console.print(f"[dim]Database:[/dim] {db_path}")


@cli.command()
@click.argument("query")
@click.option(
    "--vault",
    "-v",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to Obsidian vault (default: current directory)",
)
@click.option("--limit", "-n", default=10, help="Number of results to return")
def search(query: str, vault: Path | None, limit: int):
    """Search notes in an indexed vault."""
    vault_path = (vault or Path.cwd()).resolve()
    db_path = get_db_path(vault_path)

    if not db_path.exists():
        console.print(
            f"[red]Error:[/red] No index found for {vault_path}. "
            "Run 'obsidian-search index' first."
        )
        raise SystemExit(1)

    conn = init_db(db_path)

    # Generate query embedding
    with console.status("Generating query embedding..."):
        query_embedding = get_embedding(query)

    # Search
    results = search_similar(conn, query_embedding, limit=limit)
    conn.close()

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Display results
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=8)
    table.add_column("Title", min_width=20)
    table.add_column("Path", style="dim")

    for i, (note_id, path, title, content, distance) in enumerate(results, 1):
        # Convert distance to similarity score (lower distance = higher similarity)
        score = f"{1 / (1 + distance):.2f}"
        table.add_row(str(i), score, title or "(untitled)", path)

    console.print(table)


@cli.command()
@click.option(
    "--vault",
    "-v",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to Obsidian vault (default: current directory)",
)
def status(vault: Path | None):
    """Show indexing status for a vault."""
    vault_path = (vault or Path.cwd()).resolve()
    db_path = get_db_path(vault_path)

    console.print(f"[blue]Vault:[/blue] {vault_path}")
    console.print(f"[blue]Database:[/blue] {db_path}")

    if not db_path.exists():
        console.print("[yellow]Status:[/yellow] Not indexed")
        return

    conn = init_db(db_path)
    count = get_note_count(conn)
    conn.close()

    console.print(f"[green]Status:[/green] Indexed ({count} notes)")


if __name__ == "__main__":
    cli()
