"""
Multi-Agent System Demo

Demonstrates collaboration between:
1. Research Agent - Conducts research and generates reports
2. Review Agent - Reviews and evaluates the research agent's work
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

from research_agent import ResearchAgent
from review_agent import ReviewAgent

load_dotenv()

console = Console()


def multi_agent_workflow(query: str, max_iterations: int = 3):
    """
    Run a complete iterative multi-agent workflow:
    1. Research Agent conducts research
    2. Review Agent reviews the response
    3. If not satisfactory, Research Agent improves based on feedback
    4. Repeat until satisfactory or max iterations reached
    """
    console.print(Panel(
        "[bold cyan]Iterative Multi-Agent System Workflow[/bold cyan]\n\n"
        "Step 1: Research Agent conducts research\n"
        "Step 2: Review Agent evaluates the response\n"
        "Step 3: If needed, Research Agent improves based on feedback\n"
        "Step 4: Repeat until satisfactory",
        title="Agent Collaboration with Feedback Loop",
        border_style="cyan"
    ))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY not found[/bold red]")
        return
    
    # Initialize agents
    console.print("\n[dim]Initializing agents...[/dim]")
    research_agent = ResearchAgent()
    review_agent = ReviewAgent()
    
    iteration = 0
    best_result = None
    best_rating = 0
    
    while iteration < max_iterations:
        iteration += 1
        console.print("\n" + "="*70)
        console.print(f"[bold cyan]ITERATION {iteration}/{max_iterations}[/bold cyan]")
        console.print("="*70 + "\n")
        
        # Step 1: Research Agent
        console.print(f"[bold cyan]Research Agent (Attempt {iteration})[/bold cyan]\n")
        
        if iteration == 1:
            # First iteration - no feedback yet
            research_result = research_agent.research(query)
        else:
            # Subsequent iterations - use feedback
            if best_result is None:
                console.print("[yellow][WARNING] No previous result available, starting fresh...[/yellow]\n")
                research_result = research_agent.research(query)
            else:
                feedback = best_result.get("feedback", "")
                console.print(f"[dim]Using feedback from Review Agent...[/dim]\n")
                research_result = research_agent.research_with_feedback(query, feedback)
        
        if research_result is None or research_result.get("status") != "success":
            console.print(f"[bold red]Research failed: {research_result.get('error') if research_result else 'Unknown error'}[/bold red]")
            if best_result is not None:
                return best_result
            return None
        
        research_output = research_result.get("result", "")
        # Safety check: ensure research_output is not None
        if research_output is None:
            research_output = ""
        saved_file = research_result.get("saved_file")
        
        # Step 2: Review Agent
        console.print(f"\n[bold yellow]Review Agent (Evaluating Attempt {iteration})[/bold yellow]\n")
        
        review_result = review_agent.review(
            query=query,
            response=research_output,
            report_path=saved_file
        )
        
        if review_result is None or review_result.get("status") != "success":
            console.print(f"[bold red]Review failed: {review_result.get('error') if review_result else 'Unknown error'}[/bold red]")
            if best_result is not None:
                return best_result
            return None
        
        # Extract rating and feedback
        review_data = review_result.get("review_data", {})
        rating = review_data.get("rating")
        satisfactory = review_data.get("satisfactory")
        
        # Generate feedback for next iteration
        feedback = review_agent.generate_feedback_for_research_agent(review_data, review_result.get("review_text", ""))
        
        # Store result
        current_result = {
            "research": research_result,
            "review": review_result,
            "rating": rating,
            "satisfactory": satisfactory,
            "iteration": iteration,
            "feedback": feedback,
            "saved_file": saved_file
        }
        
        # Update best result if this is better
        if rating is not None:
            if best_result is None or rating > best_rating:
                best_result = current_result
                best_rating = rating
        elif best_result is None:
            # If no rating but no best result yet, use this one
            best_result = current_result
        
        # Display current results
        console.print(f"\n[bold]Iteration {iteration} Results:[/bold]")
        console.print(f"  Rating: {rating}/10" if rating else "  Rating: N/A")
        if satisfactory is not None:
            status = "Satisfactory" if satisfactory else "Needs Improvement"
            color = "green" if satisfactory else "yellow"
            console.print(f"  Status: [{color}]{status}[/{color}]")
        
        # Check if satisfactory - use rating as fallback
        should_stop = False
        if satisfactory is True:
            should_stop = True
            console.print("\n[bold green]✓ Research is satisfactory! Stopping iterations.[/bold green]")
        elif rating is not None and rating >= 8:
            # High rating but satisfactory not explicitly set - assume satisfactory
            should_stop = True
            console.print(f"\n[bold green]✓ High rating ({rating}/10)! Stopping iterations.[/bold green]")
        elif rating is not None and rating < 4:
            # Very low rating - probably won't improve much, stop to save iterations
            console.print(f"\n[yellow]Low rating ({rating}/10). Continuing with feedback...[/yellow]")
        
        if should_stop:
            break
        
        # If not last iteration, show feedback
        if iteration < max_iterations:
            console.print(f"\n[yellow]Feedback for next iteration:[/yellow]")
            console.print(f"[dim]{feedback[:300]}...[/dim]")
    
    # Final Summary
    console.print("\n" + "="*70)
    console.print("[bold green]WORKFLOW COMPLETE[/bold green]")
    console.print("="*70)
    
    if best_result is not None:
        console.print(f"\n[bold]Best Result (Iteration {best_result.get('iteration', 'N/A')}):[/bold]")
        console.print(f"  Research Report: {best_result.get('saved_file') or 'Not saved'}")
        console.print(f"  Review Rating: {best_result.get('rating', 'N/A')}/10")
        if best_result.get('satisfactory') is not None:
            status = "Satisfactory" if best_result['satisfactory'] else "Needs Improvement"
            color = "green" if best_result['satisfactory'] else "yellow"
            console.print(f"  Status: [{color}]{status}[/{color}]")
    
    return best_result


def interactive_demo():
    """Interactive demo for multi-agent system."""
    console.print(Panel(
        "[bold cyan]Multi-Agent System - Interactive Demo[/bold cyan]\n\n"
        "Watch two agents collaborate:\n"
        "1. Research Agent conducts research\n"
        "2. Review Agent evaluates the quality\n\n"
        "[dim]Type 'quit' or 'exit' to stop[/dim]",
        title="Agent Collaboration Demo",
        border_style="cyan"
    ))
    
    while True:
        console.print("\n" + "="*70)
        query = console.input("\n[bold]Enter your research query:[/bold] ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            console.print("\n[bold]Thanks for trying the Multi-Agent System![/bold]")
            break
        
        if not query.strip():
            console.print("[yellow]Please enter a valid query[/yellow]")
            continue
        
        # Run multi-agent workflow
        result = multi_agent_workflow(query)
        
        if result:
            console.print("\n[bold green][OK] Multi-agent workflow completed![/bold green]")


if __name__ == "__main__":
    interactive_demo()

