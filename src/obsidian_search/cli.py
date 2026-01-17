"""CLI commands for obsidian-search."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_WEB_PORT,
    EmbeddingModelError,
    IndexError,
    VaultError,
    build_obsidian_uri,
    ensure_embedding_model,
    get_vault_status,
    resolve_vault_path,
    search_vault,
    validate_vault,
)
from .database import get_db_path
from .indexer import index_vault

console = Console()


def _error(message: str) -> None:
    """Print error message and exit."""
    console.print(f"[red]Error:[/red] {message}")
    raise SystemExit(1)


def _warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]Warning:[/yellow] {message}")


class VaultContext:
    """Context object to pass vault path to subcommands."""

    def __init__(self, vault: Path | None):
        self.vault_path = resolve_vault_path(vault)


pass_vault = click.make_pass_decorator(VaultContext)


@click.group()
@click.option(
    "--vault",
    "-v",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to Obsidian vault (default: current directory)",
)
@click.pass_context
def cli(ctx: click.Context, vault: Path | None):
    """Obsidian vault semantic search using Ollama embeddings."""
    ctx.obj = VaultContext(vault)


@cli.command()
@click.option("--update", is_flag=True, help="Only index new/modified files")
@pass_vault
def index(ctx: VaultContext, update: bool):
    """Index the Obsidian vault for semantic search."""
    vault_path = ctx.vault_path

    try:
        validate_vault(vault_path)
    except VaultError as e:
        _error(str(e))

    console.print(f"[blue]Vault:[/blue] {vault_path}")

    with console.status("Checking embedding model..."):
        try:
            ensure_embedding_model()
        except EmbeddingModelError as e:
            _error(str(e))

    console.print("[green]Embedding model ready[/green]")

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
@click.option("--limit", "-n", default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")
@pass_vault
def search(ctx: VaultContext, query: str, limit: int):
    """Search notes in the indexed vault."""
    try:
        results = search_vault(ctx.vault_path, query, limit=limit)
    except IndexError as e:
        _error(str(e))

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, result in enumerate(reversed(results), 1):
        obsidian_uri = build_obsidian_uri(ctx.vault_path, result.path)
        console.print(
            f"\n[bold cyan]{i}.[/bold cyan] "
            f"[link={obsidian_uri}][bold]{result.title or '(untitled)'}[/bold][/link] "
            f"[dim]({result.score:.2f})[/dim]"
        )
        console.print(f"   [dim][link={obsidian_uri}]{result.path}[/link][/dim]")
        console.print(f"   {result.preview()}")


@cli.command()
@pass_vault
def status(ctx: VaultContext):
    """Show indexing status for the vault."""
    vault_status = get_vault_status(ctx.vault_path)

    console.print(f"[blue]Vault:[/blue] {vault_status.vault_path}")
    console.print(f"[blue]Database:[/blue] {vault_status.db_path}")

    if not vault_status.indexed:
        console.print("[yellow]Status:[/yellow] Not indexed")
    else:
        console.print(
            f"[green]Status:[/green] Indexed ({vault_status.note_count} notes, {vault_status.chunk_count} chunks)"
        )


@cli.command()
@pass_vault
def mcp(ctx: VaultContext):
    """Start the MCP server for integration with AI assistants."""
    import asyncio

    from .mcp_server import run_server

    asyncio.run(run_server(ctx.vault_path))


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind")
@click.option("--port", default=DEFAULT_WEB_PORT, show_default=True, help="Port to serve the web app")
@pass_vault
def web(ctx: VaultContext, host: str, port: int):
    """Start a simple web app to search and view notes."""
    from .web_app import run_web_app

    vault_path = ctx.vault_path

    try:
        validate_vault(vault_path)
    except VaultError as e:
        _error(str(e))

    db_path = get_db_path(vault_path)
    if not db_path.exists():
        _warning(f"No index found for {vault_path}. Run 'obsidian-search index' for search results.")

    console.print(f"[green]Web app:[/green] http://{host}:{port}")
    try:
        run_web_app(vault_path, host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[dim]Web app stopped[/dim]")


if __name__ == "__main__":
    cli()
