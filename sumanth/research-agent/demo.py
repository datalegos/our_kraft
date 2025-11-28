"""
Interactive Demo Script for Research Agent

Run this to interactively test the research agent with custom queries.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from research_agent import ResearchAgent

load_dotenv()

console = Console()


def interactive_demo():
    """Run an interactive demo of the research agent."""
    console.print(Panel(
        "[bold cyan]Research & Analysis Agent - Interactive Demo[/bold cyan]\n\n"
        "Enter research queries and watch the agent:\n"
        "1. Break down your query\n"
        "2. Search the web for information\n"
        "3. Synthesize findings\n"
        "4. Generate and save reports\n\n"
        "[dim]Type 'quit' or 'exit' to stop[/dim]",
        title="Agentic AI Demo",
        border_style="cyan"
    ))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY not found[/bold red]")
        console.print("Please create a .env file with your OpenAI API key")
        return
    
    # Initialize agent
    agent = ResearchAgent()
    
    while True:
        console.print("\n" + "="*60)
        query = console.input("\n[bold]Enter your research query:[/bold] ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            console.print("\n[bold]Thanks for trying the Research Agent![/bold]")
            break
        
        if not query.strip():
            console.print("[yellow]Please enter a valid query[/yellow]")
            continue
        
        # Run research
        result = agent.research(query)
        
        if result["status"] == "success":
            console.print("\n[bold green][OK] Research completed successfully![/bold green]")
            if result.get("saved_file"):
                console.print(f"\n[bold]Report saved to:[/bold] {result['saved_file']}")
            else:
                console.print(f"\n[dim]Check the 'reports' folder for saved reports[/dim]")
        else:
            console.print(f"\n[bold red][ERROR] Research failed: {result.get('error', 'Unknown error')}[/bold red]")


if __name__ == "__main__":
    interactive_demo()

