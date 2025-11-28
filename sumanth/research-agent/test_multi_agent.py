"""
Quick test for the multi-agent system.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from research_agent import ResearchAgent
from review_agent import ReviewAgent

load_dotenv()

console = Console()


def quick_test():
    """Quick test of the multi-agent system."""
    console.print("[bold cyan]Testing Multi-Agent System...[/bold cyan]\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY not found[/bold red]")
        return
    
    # Initialize agents
    research_agent = ResearchAgent()
    review_agent = ReviewAgent()
    
    # Test query
    query = "What are the key benefits of using AI agents?"
    
    console.print(f"[bold]Test Query:[/bold] {query}\n")
    
    # Step 1: Research
    console.print("[cyan]Step 1: Research Agent working...[/cyan]")
    research_result = research_agent.research(query)
    
    if research_result["status"] != "success":
        console.print(f"[red]Research failed: {research_result.get('error')}[/red]")
        return
    
    saved_file = research_result.get("saved_file")
    console.print(f"[green][OK] Research completed![/green]")
    if saved_file:
        console.print(f"[dim]Report saved: {saved_file}[/dim]")
    
    # Step 2: Review
    console.print("\n[yellow]Step 2: Review Agent evaluating...[/yellow]")
    
    # Get the response - prefer the saved file content if available
    response_text = research_result.get("result", "")
    if not response_text or len(response_text.strip()) < 50:
        console.print("[dim]Response text is short, will use report file content for review[/dim]")
    
    review_result = review_agent.review(
        query=query,
        response=response_text,
        report_path=saved_file
    )
    
    if review_result["status"] != "success":
        console.print(f"[red]Review failed: {review_result.get('error')}[/red]")
        if review_result.get("review_text"):
            console.print(f"\n[dim]Partial review text:[/dim]\n{review_result.get('review_text')[:500]}")
        return
    
    # Summary
    review_data = review_result.get("review_data", {})
    rating = review_data.get("rating", "N/A")
    satisfactory = review_data.get("satisfactory")
    
    console.print("\n[bold green]Multi-Agent Test Complete![/bold green]")
    console.print(f"\n[bold]Rating:[/bold] {rating}/10")
    if satisfactory is not None:
        status = "Satisfactory" if satisfactory else "Needs Improvement"
        console.print(f"[bold]Status:[/bold] {status}")
    
    # Show full review text if available
    if review_result.get("review_text"):
        console.print(f"\n[bold]Full Review:[/bold]")
        console.print(review_result["review_text"])


if __name__ == "__main__":
    quick_test()

