"""
Simple test script to verify the agent works and saves reports.
This version has reduced verbose output for cleaner terminal display.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from research_agent import ResearchAgent

load_dotenv()

console = Console()


def test_agent():
    """Test the agent with a simple query."""
    console.print("[bold cyan]Testing Research Agent...[/bold cyan]\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY not found[/bold red]")
        console.print("Please create a .env file with: OPENAI_API_KEY=your_key")
        return
    
    # Initialize agent
    console.print("[dim]Initializing agent...[/dim]")
    agent = ResearchAgent()
    
    # Test query
    test_query = "What are the latest trends in AI agents in 2024?"
    console.print(f"\n[bold]Test Query:[/bold] {test_query}\n")
    
    # Run research
    result = agent.research(test_query)
    
    # Check results
    if result["status"] == "success":
        console.print("\n[bold green][OK] SUCCESS![/bold green]")
        
        # Check if report was saved
        reports_dir = "reports"
        if os.path.exists(reports_dir):
            reports = [f for f in os.listdir(reports_dir) if f.endswith('.md')]
            if reports:
                latest_report = max(reports, key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)))
                report_path = os.path.join(reports_dir, latest_report)
                console.print(f"\n[bold green][OK] Report found:[/bold green] {report_path}")
                console.print(f"[dim]File size: {os.path.getsize(report_path)} bytes[/dim]")
            else:
                console.print("\n[yellow][WARNING] No reports found in reports/ folder[/yellow]")
        else:
            console.print("\n[yellow][WARNING] Reports folder does not exist[/yellow]")
        
        if result.get("saved_file"):
            console.print(f"\n[bold]Saved file path:[/bold] {result['saved_file']}")
    else:
        console.print(f"\n[bold red][ERROR] FAILED:[/bold red] {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    test_agent()

