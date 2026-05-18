"""CLI commands for ragamuffin."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_WEB_PORT,
    EmbeddingModelError,
    IndexError,
    build_obsidian_uri,
    ensure_embedding_model,
    get_index_status,
    is_obsidian_vault,
    resolve_dir_path,
    search_dir,
)
from .database import get_db_path
from .indexer import index_directory

console = Console()


def _error(message: str) -> None:
    """Print error message and exit."""
    console.print(f"[red]Error:[/red] {message}")
    raise SystemExit(1)


def _warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]Warning:[/yellow] {message}")


class DirContext:
    """Context object to pass directory path to subcommands."""

    def __init__(self, directory: Path | None):
        self.dir_path = resolve_dir_path(directory)


pass_dir = click.make_pass_decorator(DirContext)


@click.group()
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the directory to index/search (default: current directory)",
)
@click.pass_context
def cli(ctx: click.Context, directory: Path | None):
    """Semantic document search using local embeddings (fastembed)."""
    ctx.obj = DirContext(directory)


@cli.command()
@click.option("--force", is_flag=True, help="Re-index all files, not just new/modified ones")
@pass_dir
def index(ctx: DirContext, force: bool):
    """Index documents in the directory for semantic search (incremental by default)."""
    dir_path = ctx.dir_path

    console.print(f"[blue]Directory:[/blue] {dir_path}")

    with console.status("Loading embedding model (downloads ~435 MB on first run)..."):
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

        indexed, skipped, deleted = index_directory(
            dir_path,
            update_only=not force,
            progress_callback=update_progress,
        )

    console.print()
    console.print(f"[green]Indexed:[/green] {indexed} documents")
    if skipped:
        console.print(f"[yellow]Skipped:[/yellow] {skipped} documents")
    if deleted:
        console.print(f"[red]Deleted:[/red] {deleted} documents (removed from directory)")

    db_path = get_db_path(dir_path)
    console.print(f"[dim]Database:[/dim] {db_path}")


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")
@pass_dir
def search(ctx: DirContext, query: str, limit: int):
    """Search documents in the indexed directory."""
    dir_path = ctx.dir_path
    obsidian = is_obsidian_vault(dir_path)

    try:
        results = search_dir(dir_path, query, limit=limit)
    except IndexError as e:
        _error(str(e))

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, result in enumerate(reversed(results), 1):
        if obsidian:
            link = build_obsidian_uri(dir_path, result.path)
            title_markup = f"[link={link}][bold]{result.title or '(untitled)'}[/bold][/link]"
            path_markup = f"[dim][link={link}]{result.path}[/link][/dim]"
        else:
            title_markup = f"[bold]{result.title or '(untitled)'}[/bold]"
            path_markup = f"[dim]{result.path}[/dim]"

        console.print(
            f"\n[bold cyan]{i}.[/bold cyan] "
            f"{title_markup} "
            f"[dim]({result.score:.2f})[/dim]"
        )
        console.print(f"   {path_markup}")
        console.print(f"   {result.preview()}")


@cli.command()
@pass_dir
def status(ctx: DirContext):
    """Show indexing status for the directory."""
    index_status = get_index_status(ctx.dir_path)

    console.print(f"[blue]Directory:[/blue] {index_status.dir_path}")
    console.print(f"[blue]Database:[/blue] {index_status.db_path}")

    if not index_status.indexed:
        console.print("[yellow]Status:[/yellow] Not indexed")
    else:
        console.print(
            f"[green]Status:[/green] Indexed "
            f"({index_status.doc_count} documents, {index_status.chunk_count} chunks)"
        )


@cli.command()
@click.pass_context
def mcp(ctx: click.Context):
    """Start the MCP server for integration with AI assistants."""
    import asyncio

    from .mcp_server import run_server

    # Pass explicit --directory if given; otherwise None so run_server checks RAGAMUFFIN_DIRECTORY
    asyncio.run(run_server(ctx.parent.params.get("directory")))


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind")
@click.option("--port", default=DEFAULT_WEB_PORT, show_default=True, help="Port to serve the web app")
@pass_dir
def web(ctx: DirContext, host: str, port: int):
    """Start a simple web app to search and view documents."""
    from .web_app import run_web_app

    dir_path = ctx.dir_path
    db_path = get_db_path(dir_path)

    if not db_path.exists():
        _warning(f"No index found for {dir_path}. Run 'muffin index' for search results.")

    console.print(f"[green]Web app:[/green] http://{host}:{port}")
    try:
        run_web_app(dir_path, host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[dim]Web app stopped[/dim]")


if __name__ == "__main__":
    cli()
