"""CLI interface for IntraMind AI Agent."""

import asyncio
import sys
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from agent import IntraMindAgent
from config import settings
from utils.logging import setup_logging

console = Console()


def print_banner() -> None:
    """Print the IntraMind banner."""
    banner = """
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║          🧠 IntraMind AI Agent            ║
    ║   Intelligent Document Search Platform    ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_config_info() -> None:
    """Print configuration information."""
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row("API Gateway", settings.api_gateway_url)
    table.add_row("Primary LLM", settings.primary_llm_provider)
    table.add_row("Router LLM", settings.router_llm_provider)
    table.add_row("Default Collection", settings.default_collection)
    table.add_row("Log Level", settings.log_level)

    console.print(table)
    console.print()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """IntraMind AI Agent - Intelligent Document Search Platform."""
    setup_logging()


@cli.command()
@click.option(
    "--query",
    "-q",
    help="Search query (if not provided, will enter interactive mode)",
)
@click.option(
    "--collection",
    "-c",
    default=None,
    help="Collection to search (default: intramind_documents)",
)
@click.option(
    "--limit",
    "-l",
    default=10,
    type=int,
    help="Maximum number of results (default: 10)",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream results as they're generated (default: enabled)",
)
def search(query: str | None, collection: str | None, limit: int, stream: bool) -> None:
    """Search for documents using semantic search."""
    print_banner()

    collection_name = collection or settings.default_collection

    # Interactive mode if no query provided
    if not query:
        console.print("[yellow]Interactive Search Mode[/yellow]")
        console.print("Type your query (or 'exit' to quit)\n")

        while True:
            query = Prompt.ask("[bold cyan]Search")

            if query.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            asyncio.run(_execute_search(query, collection_name, limit, stream))
            console.print()
    else:
        # Single query mode
        asyncio.run(_execute_search(query, collection_name, limit, stream))


async def _execute_search(query: str, collection: str, limit: int, stream: bool) -> None:
    """Execute a search query.

    Args:
        query: Search query
        collection: Collection name
        limit: Max results
        stream: Whether to stream results
    """
    agent = IntraMindAgent()

    if stream:
        # Streaming mode - show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching...", total=None)

            response_text = None
            async for update in agent.stream_search(query, collection, limit):
                step = update.get("step", "")
                if step == "classify_query":
                    progress.update(task, description="Analyzing query...")
                elif step in ["simple_search", "complex_search"]:
                    progress.update(task, description="Searching documents...")
                elif step == "synthesize_results":
                    progress.update(task, description="Synthesizing response...")

                if update.get("response"):
                    response_text = update["response"]

                if update.get("complete"):
                    progress.stop()
                    if not update.get("error"):
                        _display_response(response_text or "No response generated", query)
                    else:
                        console.print(f"[red]Error: {update['error']}[/red]")
    else:
        # Non-streaming mode
        with console.status("[bold green]Searching...", spinner="dots"):
            result = await agent.search(query, collection, limit)

        if result.get("success"):
            _display_response(result.get("response", "No response"), query)
            _display_results(result.get("results", []))
        else:
            console.print(f"[red]Error: {result.get('error')}[/red]")


def _display_response(response: str, query: str) -> None:
    """Display the synthesized response.

    Args:
        response: Response text
        query: Original query
    """
    console.print()
    console.print(
        Panel(
            Markdown(response),
            title=f"[bold cyan]Response to: {query[:50]}...[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()


def _display_results(results: list[dict[str, Any]]) -> None:
    """Display search results in a table.

    Args:
        results: List of search results
    """
    if not results:
        return

    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=20)
    table.add_column("Content", style="white", width=60)
    table.add_column("Score", style="green", width=10)

    for result in results[:5]:  # Show top 5
        content = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
        score = f"{result.get('score', 0):.3f}" if result.get("score") else "N/A"
        table.add_row(result.get("id", ""), content, score)

    console.print(table)


@cli.command()
def info() -> None:
    """Display configuration and system information."""
    print_banner()
    print_config_info()


@cli.command()
@click.option(
    "--url",
    default=None,
    help="API Gateway URL (default: from settings)",
)
def health(url: str | None) -> None:
    """Check API Gateway health status."""
    asyncio.run(_health(url))


async def _health(url: str | None) -> None:
    from tools.api_client import APIGatewayClient

    console.print("[bold]Checking API Gateway health...[/bold]\n")

    try:
        async with APIGatewayClient(base_url=url) as client:
            response = await client.health_check()

        console.print(Panel("API Gateway is healthy", border_style="green"))
        console.print(f"Response: {response}")

    except Exception as e:
        console.print(Panel(f"Health check failed: {e}", border_style="red"))
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
