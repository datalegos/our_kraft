"""
Review Agent - Evaluates and Reviews Research Agent's Responses

This agent demonstrates:
- Multi-agent collaboration
- Quality assessment and feedback
- Response optimization suggestions
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

console = Console()


class ReviewAgent:
    """
    A review agent that evaluates research agent responses and provides:
    1. Quality assessment
    2. Completeness check
    3. Optimization suggestions
    4. Overall satisfaction rating
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """Initialize the review agent with LLM."""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.console = Console()
    
    def generate_feedback_for_research_agent(self, review_data: Dict[str, Any], review_text: str) -> str:
        """
        Generate actionable feedback for the Research Agent to improve.
        
        Args:
            review_data: Parsed review data
            review_text: Full review text
            
        Returns:
            Actionable feedback string for Research Agent
        """
        feedback_parts = []
        
        # Add rating context
        rating = review_data.get("rating")
        if rating:
            if rating < 7:
                feedback_parts.append(f"Current rating: {rating}/10 - Needs significant improvement")
            else:
                feedback_parts.append(f"Current rating: {rating}/10 - Good, but can be improved")
        
        # Add areas for improvement
        improvements = review_data.get("improvements", [])
        if improvements:
            feedback_parts.append("\nKey areas to improve:")
            for imp in improvements[:3]:
                feedback_parts.append(f"- {imp}")
        
        # Add optimization suggestions
        suggestions = review_data.get("suggestions", [])
        if suggestions:
            feedback_parts.append("\nSpecific actions to take:")
            for sug in suggestions[:3]:
                feedback_parts.append(f"- {sug}")
        
        # Add summary if available
        summary = review_data.get("summary", "")
        if summary:
            feedback_parts.append(f"\nOverall assessment: {summary}")
        
        return "\n".join(feedback_parts) if feedback_parts else "Please improve semantic understanding and provide more balanced, critical analysis."
    
    def review(self, query: str, response: str, report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Review a research agent's response.
        
        Args:
            query: The original research query
            response: The research agent's response/output
            report_path: Optional path to the saved report file
            
        Returns:
            Dictionary with review results and feedback
        """
        self.console.print(Panel(
            f"[bold yellow]Reviewing Research Agent Response[/bold yellow]\n\n"
            f"[dim]Query:[/dim] {query}",
            title="Review Agent",
            border_style="yellow"
        ))
        
        # Read report content if path provided
        report_content = ""
        if report_path and os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    report_content = content if content is not None else ""
            except Exception as e:
                self.console.print(f"[red][ERROR] Could not read report: {e}[/red]")
                report_content = ""
        
        # Create review prompt
        review_prompt = """You are a Quality Review Agent with deep semantic understanding capabilities. Your role is to evaluate research reports by understanding the true intent and meaning behind the user's query, not just surface-level keyword matching.

CRITICAL: You must semantically understand what the user is REALLY asking:
- What is the underlying question or need?
- What information would truly satisfy this query?
- What are the implicit expectations, assumptions, or hypotheses?
- What context or background might be relevant?
- Are there multiple aspects or dimensions to address?
- Does the user have a specific assumption that needs validation or correction?

RATING GUIDELINES - BE STRICT:
- 9-10/10: Excellent semantic understanding, critical evaluation of assumptions, balanced perspective, comprehensive
- 7-8/10: Good content but missing critical evaluation OR one-sided when balance needed OR incomplete semantic understanding
- 5-6/10: Acceptable content but POOR semantic understanding OR blindly confirms assumptions OR misses key aspects
- 3-4/10: Poor understanding, missing critical elements, low quality
- 1-2/10: Completely misses the point, irrelevant, very poor

CRITICAL EVALUATION OF ASSUMPTIONS - BE HARSH:
When a user states an assumption/hypothesis (e.g., "I think it's all about X"):
- BAD (Rate 5-6/10): Response ONLY confirms what user thinks, provides one-sided view, doesn't evaluate alternatives
- GOOD (Rate 7-8/10): Response validates the assumption BUT also provides critical analysis, mentions alternatives
- EXCELLENT (Rate 9-10/10): Response critically evaluates assumption, provides balanced view, addresses alternatives, helps user think more deeply

Example: User asks "What's the trend at end of 2025? I think it's all about quantum computing"
- BAD Response: Only talks about quantum computing, confirms user's assumption → Rate 5-6/10
- GOOD Response: Covers quantum computing thoroughly BUT also mentions other trends, provides balanced view → Rate 7-8/10
- EXCELLENT Response: Evaluates quantum computing as ONE trend among several, provides critical analysis, addresses alternatives → Rate 9-10/10

Evaluate the research response based on these criteria (in order of importance):

1. **Semantic Relevance & Intent Matching** (MOST IMPORTANT - 40% weight): 
   - Does the response address what the user is REALLY asking, not just keywords?
   - Does it understand the underlying intent and context?
   - Does it answer the implicit questions, not just the explicit ones?
   - If the user has an assumption/hypothesis, does it critically evaluate it rather than just confirming it?
   - Would this response satisfy someone who asked this question?
   - DEDUCT 2-3 POINTS if it only confirms assumptions without critical evaluation

2. **Critical Thinking & Balanced Perspective** (VERY IMPORTANT - 30% weight):
   - Does it blindly follow the user's assumption, or does it provide critical analysis?
   - Does it offer alternative perspectives, not just confirmation?
   - Does it validate AND correct/expand on user assumptions when present?
   - Is it comprehensive enough to challenge or refine the user's thinking?
   - DEDUCT 2-3 POINTS if one-sided when balance is needed

3. **Completeness** (IMPORTANT - 15% weight): 
   - Does it cover all aspects of what the user is asking?
   - Are there important dimensions or perspectives missing?
   - Does it address both explicit and implicit parts of the query?
   - If the user focuses on one thing, does it also address related/alternative aspects?
   - DEDUCT 1-2 POINTS if important aspects are missing

4. **Accuracy** (10% weight): 
   - Are the facts and claims reasonable and well-supported?
   - Is the information current and relevant to the query's context?

5. **Structure, Depth, Clarity** (5% weight combined): 
   - Is the report well-organized, detailed enough, and clear?

When reviewing, ask yourself:
- "If I asked this question with an assumption, would this response just confirm my bias or help me think more critically?"
- "Does this answer what I'm really trying to find out, or does it just echo what I already think?"
- "Are there important aspects of the query that were missed or misunderstood?"
- "Does this response show critical thinking, or does it blindly follow the user's lead?"
- "Would I rate this 7/10 if it only confirms assumptions without evaluation? NO - that's 5-6/10!"

RED FLAGS - DEDUCT POINTS:
- Response only confirms user's assumption without critical evaluation → DEDUCT 2-3 points (this is a major flaw)
- Missing alternative perspectives or related topics → DEDUCT 1-2 points
- One-sided view when a balanced perspective is needed → DEDUCT 2 points
- Doesn't address implicit questions (like "is my assumption correct?") → DEDUCT 2 points
- Good content but poor semantic understanding → DEDUCT 2-3 points

You MUST provide your review in this EXACT format (copy the structure exactly):

**Overall Rating**: [MUST be a number 1-10, e.g., "8" or "7"]
**Satisfactory**: [MUST be exactly "Yes" or "No"]
**Strengths**: [1-2 key strengths only, max 15 words each]
**Areas for Improvement**: [1-2 critical issues only, max 15 words each]
**Top Priority Fix**: [Single most important action item, max 25 words]
**Summary**: [One sentence assessment, max 40 words]

MANDATORY REQUIREMENTS:
- **Overall Rating** MUST be included and MUST be a number between 1-10 (write "8" not "eight" or "good")
- **Satisfactory** MUST be included and MUST be exactly "Yes" or "No" (not "maybe", "partially", or anything else)
- Use the EXACT format above with **bold** markers
- TOTAL review should be under 150 words
- Be EXTREMELY BRIEF and ACTIONABLE
- Focus ONLY on the most critical issues
- No repetition. No verbose explanations. Just the essentials.
- BE STRICT: If semantic understanding is poor or assumptions are blindly confirmed, rate 5-6/10, not 7-8/10!"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", review_prompt),
            ("human", """Original Query: {query}

STEP 1: Analyze what the user is REALLY asking:
- What is the underlying intent or need?
- What information would truly satisfy this query?
- What are the implicit questions or expectations?
- Does the user have a specific assumption, hypothesis, or belief stated in the query?
- If yes, what would a good response do? (Validate AND critically evaluate, not just confirm)
- What context is relevant?
- Are there alternative perspectives or related topics that should be addressed?

STEP 2: Evaluate the Research Agent Response:
{response}

{report_content}

STEP 3: Critical Evaluation - BE STRICT:
- Does the response just confirm what the user thinks, or does it provide critical analysis?
- If the user stated an assumption (e.g., "I think it's all about X"), does the response:
  * Blindly follow the assumption (BAD - rate 5-6/10, not 7-8/10)
  * Critically evaluate and provide balanced perspective (GOOD - rate 7-9/10)
- Does it address alternative perspectives or just the user's focus?
- Is it comprehensive enough to help the user think more critically?
- Would this response satisfy someone who asked this question, or does it just echo their assumption?

RATING REMINDER:
- If response only confirms assumption without critical evaluation → Rate 5-6/10 (not 7-8/10)
- If response is one-sided when balance is needed → Rate 5-7/10 (deduct points)
- If semantic understanding is poor → Rate 5-6/10 (deduct points)
- Good content but missing critical evaluation → Rate 6-7/10 (not 8-9/10)

Review this response by checking:
1. Semantic understanding - does it understand what the user REALLY wants? (40% weight)
2. Critical thinking - does it evaluate assumptions rather than just confirming them? (30% weight)
3. Balanced perspective - does it provide comprehensive, nuanced information? (15% weight)
4. Completeness - are all aspects addressed? (15% weight)

Provide detailed feedback focusing on intent understanding, semantic relevance, and critical evaluation of user assumptions. BE STRICT with ratings when semantic understanding or critical thinking is poor.""")
        ])
        
        try:
            # Check if we have valid content to review
            # Safety check: ensure response is not None
            if response is None:
                response = ""
            
            if not response or len(response.strip()) < 20:
                self.console.print("[yellow][WARNING] Research response is too short or empty. Reading from report file...[/yellow]")
                if report_content and len(report_content.strip() if report_content else 0) > 100:
                    response = report_content
                else:
                    self.console.print("[red][ERROR] No valid content to review[/red]")
                    return {
                        "status": "error",
                        "error": "No valid content to review - response was empty or too short",
                        "query": query
                    }
            
            # Use report content if it's more substantial than the response
            if report_content and len(report_content) > len(response):
                content_to_review = report_content
            else:
                content_to_review = response
            
            # Format the prompt
            formatted_prompt = prompt.format_messages(
                query=query,
                response=content_to_review[:3000] if len(content_to_review) > 3000 else content_to_review,  # Increased limit
                report_content=f"\n\nFull Report Content:\n{report_content[:2000]}" if report_content and report_content != content_to_review else ""
            )
            
            self.console.print("[dim]Generating review...[/dim]")
            
            # Get review from LLM
            review_result = self.llm.invoke(formatted_prompt)
            review_text = review_result.content if hasattr(review_result, 'content') else str(review_result)
            
            # Safety check: ensure review_text is not None
            if review_text is None:
                review_text = ""
            
            if not review_text or len(review_text.strip()) < 10:
                self.console.print("[red][ERROR] Review Agent returned empty response[/red]")
                return {
                    "status": "error",
                    "error": "Review Agent returned empty response",
                    "query": query
                }
            
            # Parse the review to extract structured information
            review_data = self._parse_review(review_text, query)
            
            # Fallback parsing if rating/satisfactory not found
            import re
            if review_data.get("rating") is None:
                # Try to find any number that might be a rating
                number_match = re.search(r'\b([1-9]|10)\b', review_text)
                if number_match:
                    potential_rating = int(number_match.group(1))
                    if 1 <= potential_rating <= 10:
                        review_data["rating"] = potential_rating
                        self.console.print(f"[dim]Extracted rating: {potential_rating} (fallback)[/dim]")
            
            if review_data.get("satisfactory") is None:
                # Use rating as fallback
                if review_data.get("rating") is not None:
                    rating = review_data["rating"]
                    if rating >= 7:
                        review_data["satisfactory"] = True
                        self.console.print(f"[dim]Set satisfactory=True based on rating {rating}[/dim]")
                    elif rating < 5:
                        review_data["satisfactory"] = False
                        self.console.print(f"[dim]Set satisfactory=False based on rating {rating}[/dim]")
            
            # Display review
            self._display_review(review_data, review_text)
            
            return {
                "status": "success",
                "review_text": review_text,
                "review_data": review_data,
                "query": query
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.console.print(f"[bold red][ERROR] Review failed: {str(e)}[/bold red]")
            self.console.print(f"[dim]{error_details}[/dim]")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    def _parse_review(self, review_text: str, query: str) -> Dict[str, Any]:
        """Parse the review text to extract structured information."""
        import re
        
        # Safety check: ensure review_text is not None
        if not review_text:
            review_text = ""
        
        review_data = {
            "rating": None,
            "satisfactory": None,
            "strengths": [],
            "improvements": [],
            "suggestions": [],
            "summary": ""
        }
        
        # Extract rating - try multiple patterns
        rating_match = re.search(r'(?:Overall\s+)?Rating[:\s]+(\d+)', review_text, re.IGNORECASE)
        if not rating_match:
            # Try pattern like "Rating: 8" or "8/10"
            rating_match = re.search(r'Rating[:\s]*(\d+)(?:/10)?', review_text, re.IGNORECASE)
        if not rating_match:
            # Try just finding a number after "rating"
            rating_match = re.search(r'rating[:\s]+(\d+)', review_text, re.IGNORECASE)
        if rating_match:
            try:
                review_data["rating"] = int(rating_match.group(1))
            except (ValueError, IndexError):
                pass
        
        # Extract satisfactory - try multiple patterns
        sat_match = re.search(r'Satisfactory[:\s]+(Yes|No|yes|no)', review_text, re.IGNORECASE)
        if not sat_match:
            # Try without colon
            sat_match = re.search(r'Satisfactory\s+(Yes|No|yes|no)', review_text, re.IGNORECASE)
        if not sat_match:
            # Try "Status: Satisfactory" or similar
            sat_match = re.search(r'(?:Status|Satisfactory)[:\s]+(Satisfactory|Not Satisfactory|Yes|No)', review_text, re.IGNORECASE)
        
        if sat_match:
            sat_value = sat_match.group(1).lower()
            review_data["satisfactory"] = sat_value in ['yes', 'satisfactory']
        
        # Fallback: If rating is high (>=8) but satisfactory not set, assume satisfactory
        if review_data["rating"] is not None and review_data["rating"] >= 8 and review_data["satisfactory"] is None:
            review_data["satisfactory"] = True
        # If rating is low (<5) but satisfactory not set, assume not satisfactory
        elif review_data["rating"] is not None and review_data["rating"] < 5 and review_data["satisfactory"] is None:
            review_data["satisfactory"] = False
        
        # Extract strengths - limit to 2
        strengths_match = re.search(r'Strengths[:\s]+(.*?)(?=Areas for Improvement|Needs Work|Priority|Summary|$)', 
                                   review_text, re.IGNORECASE | re.DOTALL)
        if strengths_match and strengths_match.group(1):
            strengths_text = strengths_match.group(1).strip()
            if strengths_text:
                # Split by lines or bullets, limit to 2
                review_data["strengths"] = [s.strip('- •*').strip() for s in strengths_text.split('\n') if s and s.strip()][:2]
        
        # Extract improvements - limit to 2
        improvements_match = re.search(r'Areas for Improvement|Needs Work[:\s]+(.*?)(?=Priority|Optimization|Summary|$)', 
                                     review_text, re.IGNORECASE | re.DOTALL)
        if improvements_match and improvements_match.group(1):
            improvements_text = improvements_match.group(1).strip()
            if improvements_text:
                review_data["improvements"] = [s.strip('- •*').strip() for s in improvements_text.split('\n') if s and s.strip()][:2]
        
        # Extract suggestions - prioritize "Top Priority Fix" or "Priority Fix"
        priority_match = re.search(r'Top Priority Fix|Priority Fix[:\s]+(.*?)(?=Summary|$)', 
                                   review_text, re.IGNORECASE | re.DOTALL)
        if priority_match and priority_match.group(1):
            priority_text = priority_match.group(1).strip()
            if priority_text:
                review_data["suggestions"] = [priority_text.strip('- •*').strip()[:150]]
        else:
            # Fallback to optimization suggestions
            suggestions_match = re.search(r'Optimization Suggestions[:\s]+(.*?)(?=Summary|$)', 
                                         review_text, re.IGNORECASE | re.DOTALL)
            if suggestions_match and suggestions_match.group(1):
                suggestions_text = suggestions_match.group(1).strip()
                if suggestions_text:
                    # Take only first suggestion, limit length
                    suggestions_list = [s.strip('- •*').strip() for s in suggestions_text.split('\n') if s and s.strip()]
                    if suggestions_list:
                        review_data["suggestions"] = [suggestions_list[0][:150]]
        
        # Extract summary - limit length
        summary_match = re.search(r'Summary[:\s]+(.*?)$', review_text, re.IGNORECASE | re.DOTALL)
        if summary_match and summary_match.group(1):
            summary_text = summary_match.group(1).strip()
            if summary_text:
                review_data["summary"] = summary_text[:200]  # Limit to 200 chars
        
        return review_data
    
    def _display_review(self, review_data: Dict[str, Any], review_text: str = ""):
        """Display the review in a concise, formatted way."""
        self.console.print("\n")
        
        # Rating and satisfactory status - compact display
        rating = review_data.get("rating")
        satisfactory = review_data.get("satisfactory")
        
        # Create a compact summary panel
        rating_display = f"{rating}/10" if rating is not None and isinstance(rating, int) else "N/A"
        if rating is not None and isinstance(rating, int):
            color = "green" if rating >= 7 else "yellow" if rating >= 5 else "red"
            rating_display = f"[bold {color}]{rating}/10[/bold {color}]"
        
        status_display = ""
        if satisfactory is not None:
            status_display = "[OK] Satisfactory" if satisfactory else "[WARNING] Needs Improvement"
            status_color = "green" if satisfactory else "yellow"
            status_display = f"[{status_color}]{status_display}[/{status_color}]"
        
        # Compact header
        header_text = f"Rating: {rating_display}"
        if status_display:
            header_text += f" | {status_display}"
        
        self.console.print(f"[bold]{header_text}[/bold]")
        
        # Create a compact table - limit to 2 items per category
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("", style="cyan", width=20)
        table.add_column("", style="white", width=60)
        
        has_content = False
        
        # Strengths - limit to 2, truncate long items
        strengths = review_data.get("strengths", [])
        if strengths:
            strength_text = "\n".join(f"• {s[:60]}{'...' if len(s) > 60 else ''}" for s in strengths[:2])
            table.add_row("[green]Strengths[/green]", strength_text)
            has_content = True
        
        # Improvements - limit to 2, truncate long items
        improvements = review_data.get("improvements", [])
        if improvements:
            imp_text = "\n".join(f"• {i[:60]}{'...' if len(i) > 60 else ''}" for i in improvements[:2])
            table.add_row("[yellow]Needs Work[/yellow]", imp_text)
            has_content = True
        
        # Top priority fix - single most important item
        suggestions = review_data.get("suggestions", [])
        if suggestions:
            top_fix = suggestions[0][:80] + ('...' if len(suggestions[0]) > 80 else '')
            table.add_row("[red]Priority Fix[/red]", top_fix)
            has_content = True
        
        if has_content:
            self.console.print("\n")
            self.console.print(table)
        
        # Summary - concise
        summary = review_data.get("summary", "")
        if summary:
            # Truncate summary if too long
            summary_display = summary[:150] + ('...' if len(summary) > 150 else '')
            self.console.print(f"\n[dim]{summary_display}[/dim]")
        elif not has_content and review_text:
            # Fallback: show first 200 chars of review text
            self.console.print(f"\n[dim]{review_text[:200]}...[/dim]")
        
        self.console.print("\n")


def main():
    """Demo function for the review agent."""
    console.print(Panel(
        "[bold]Review Agent Demo[/bold]\n\n"
        "This agent reviews research agent responses and provides:\n"
        "• Quality assessment\n"
        "• Completeness check\n"
        "• Optimization suggestions",
        title="Review Agent",
        border_style="yellow"
    ))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY not found[/bold red]")
        return
    
    # Example usage
    review_agent = ReviewAgent()
    
    # Example query and response
    example_query = "What are the latest trends in AI agents in 2024?"
    example_response = """
    AI agents in 2024 are seeing significant growth in several areas:
    1. Enhanced personalization capabilities
    2. Better conversational AI
    3. Increased focus on ethical AI development
    """
    
    console.print("\n[dim]Running example review...[/dim]\n")
    result = review_agent.review(example_query, example_response)
    
    if result["status"] == "success":
        console.print("[bold green][OK] Review completed![/bold green]")


if __name__ == "__main__":
    main()

