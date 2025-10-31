import sys
from rich.console import Console
from rich.panel import Panel

from ticketbuddy.pipeline import TicketBuddyPipeline


def main():
    console = Console()
    console.print(Panel("TicketBuddy RAG Pipeline", style="bold blue"))
    try:
        pipeline = TicketBuddyPipeline.build()
        console.print("[green]Pipeline built successfully. Running queries...[/green]")
        outputs = pipeline.run_all()
        console.print(f"[green]Done. Generated {len(outputs)} results and wrote to results.json[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
