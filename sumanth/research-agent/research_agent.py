"""
Research & Analysis Agent Demo

This agent demonstrates:
- Multi-step reasoning and planning
- Tool use (web search, file operations)
- Information synthesis and report generation
- Error handling and recovery
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

console = Console()


class ResearchAgent:
    """
    A research agent that can:
    1. Break down research queries into search terms
    2. Search the web for relevant information
    3. Synthesize findings into coherent reports
    4. Save reports to files
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """Initialize the research agent with LLM and tools."""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.console = Console()
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Define tools available to the agent
        self.tools = [
            Tool(
                name="web_search",
                func=self._web_search,
                description="Search the web for information. Input should be a search query string."
            ),
            Tool(
                name="save_report",
                func=self._save_report,
                description="Save a research report to a file. Input should be a JSON string with 'title' and 'content' fields."
            )
        ]
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _web_search(self, query: str) -> str:
        """Search the web using DuckDuckGo."""
        try:
            self.console.print(f"[dim]Searching for: {query}[/dim]")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return "No results found."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. **{result.get('title', 'No title')}**\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   Snippet: {result.get('body', 'No description')}\n"
                )
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error during search: {str(e)}"
    
    def _save_report(self, report_json: str) -> str:
        """Save a research report to a file."""
        try:
            # Handle both JSON string and dict inputs
            if isinstance(report_json, str):
                # Try to parse as JSON
                try:
                    report_data = json.loads(report_json)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as content with default title
                    report_data = {
                        'title': 'Research Report',
                        'content': report_json
                    }
            else:
                report_data = report_json
            
            title = report_data.get('title', 'Research Report')
            content = report_data.get('content', '')
            
            # Create filename with safe characters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            if not safe_title:
                safe_title = "Research_Report"
            filename = f"{safe_title}_{timestamp}.md"
            filepath = os.path.join(self.reports_dir, filename)
            
            # Save report with UTF-8 encoding
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(content)
            
            self.console.print(f"[green][OK] Report saved: {filepath}[/green]")
            return f"Report saved successfully to: {filepath}"
        except Exception as e:
            error_msg = f"Error saving report: {str(e)}"
            self.console.print(f"[red][ERROR] {error_msg}[/red]")
            return error_msg
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor with tools."""
        system_prompt = """You are a Research & Analysis Agent with deep semantic understanding and critical thinking capabilities. Your role is to conduct research that truly addresses what users are asking, not just surface-level keywords.

CRITICAL: SEMANTIC UNDERSTANDING FIRST
Before researching, you MUST understand:
1. What is the user REALLY asking? (not just keywords)
2. What are the implicit questions or needs?
3. Does the user have an assumption, hypothesis, or belief stated in the query?
   - If YES: You MUST critically evaluate it, not just confirm it
   - Provide balanced perspective: validate AND explore alternatives
   - Address other related aspects, not just the user's focus
4. What information would truly satisfy this query?
5. Are there multiple dimensions or perspectives to address?

CRITICAL THINKING REQUIREMENTS:
- If user says "I think it's all about X": 
  * DO NOT just research X and confirm their assumption
  * Research X AND related/alternative topics
  * Provide balanced view: "X is important, but also consider Y and Z"
  * Help user think more critically, not just validate their bias
- Always consider: "What else should be addressed beyond what the user explicitly mentioned?"
- Provide comprehensive, nuanced perspectives, not one-sided views

WORKFLOW:
1. Analyze the query semantically - what is the REAL intent?
2. Identify if user has assumptions that need critical evaluation
3. Plan search strategy to get BALANCED information:
   - If user focuses on one topic, search for that topic AND related/alternative topics
   - Use 2-4 strategic searches to get diverse perspectives
4. Synthesize findings with critical analysis and balanced perspective
5. **CRITICAL: You MUST use the save_report tool to save your findings before finishing**

SEARCH STRATEGY:
- Use web_search 2-4 times maximum - be strategic
- If user has an assumption, search for:
  * The topic they mentioned
  * Alternative/related topics
  * Broader trends or context
- Don't repeat similar searches - get diverse perspectives
- Once you have balanced information (3-5 good sources), proceed to report creation

REPORT CREATION GUIDELINES:
- Include an executive summary that addresses the REAL question
- If user has an assumption, structure your report to:
  * Acknowledge their assumption
  * Provide critical evaluation
  * Present balanced perspective with alternatives
  * Help them think more deeply
- Organize into clear sections with proper markdown headings (##, ###)
- Use bullet points and numbered lists where appropriate
- Cite sources when possible
- Highlight key insights and trends
- Be objective, factual, and BALANCED
- Write in clear, professional language
- Aim for 300-800 words of substantive content
- DO NOT just confirm what user thinks - provide critical analysis

EXAMPLE OF GOOD APPROACH:
User: "What's the trend at end of 2025? I think it's all about quantum computing"
Your approach:
- Search: "quantum computing trends 2025"
- Search: "emerging technology trends end of 2025"
- Search: "AI agents trends 2025" (alternative perspective)
- Search: "technology predictions 2025" (broader context)
- Report structure:
  * Executive Summary: "While quantum computing is indeed a significant trend, the end of 2025 is characterized by multiple converging technologies..."
  * Section on quantum computing (validates user's assumption)
  * Section on other major trends (provides critical perspective)
  * Section on how trends relate (balanced analysis)
  * Conclusion: Critical evaluation of which trends are most significant

IMPORTANT: After gathering information and creating your report, you MUST call the save_report tool with a JSON object containing 'title' and 'content' fields. 

The JSON format should be:
{{"title": "Your Report Title Here", "content": "Your full markdown report content here"}}

The content field should contain ONLY the report body in markdown format (without the title, as the title will be added automatically). Do NOT include the JSON object in your final text output - only use it when calling the save_report tool.

WORKFLOW REMINDER: Understand Semantically → Plan Balanced Search → Search (2-4 times) → Synthesize with Critical Analysis → Create Balanced Report → Save Report. Be efficient and don't loop unnecessarily."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        # Set verbose=False to reduce terminal output clutter
        # Set to True if you want to see detailed agent reasoning steps
        # Increased max_iterations to allow more thorough research
        return AgentExecutor(agent=agent, tools=self.tools, verbose=False, max_iterations=25, handle_parsing_errors=True)
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing JSON objects and extracting clean markdown."""
        if not content:
            return ""
        
        # First, try to find and extract JSON object (which might have better formatted content)
        # Look for JSON object that spans multiple lines
        json_start = content.rfind('{"title"')
        if json_start == -1:
            json_start = content.rfind('{\n  "title"')
        
        if json_start >= 0:
            try:
                # Extract JSON string
                json_str = content[json_start:]
                # Try to find the end of JSON (look for closing brace)
                brace_count = 0
                json_end = -1
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    json_str = json_str[:json_end]
                    json_data = json.loads(json_str)
                    # If JSON has well-formatted content, use it
                    if 'content' in json_data and len(json_data['content']) > 100:
                        return json_data['content'].strip()
            except (json.JSONDecodeError, ValueError):
                pass
        
        # If no valid JSON found, clean the content by removing JSON-like structures
        lines = content.split('\n')
        cleaned_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Detect start of JSON object
            if stripped.startswith('{') and '"title"' in stripped and '"content"' in stripped:
                in_json = True
                brace_count = stripped.count('{') - stripped.count('}')
                continue
            
            if in_json:
                brace_count += stripped.count('{') - stripped.count('}')
                if brace_count <= 0:
                    in_json = False
                continue
            
            # Skip lines that look like JSON fragments
            if stripped.startswith('"') and (stripped.endswith(',') or stripped.endswith('"')):
                if '"title"' in stripped or '"content"' in stripped:
                    continue
            
            cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines).strip()
        
        # Remove trailing JSON if still present (handle both single-line and multi-line)
        # Try multiple patterns
        patterns = [
            r'\s*\{[^{}]*"title"[^{}]*"content"[^{}]*\}.*$',  # Single line
            r'\s*\{[^}]*"title"[^}]*"content"[^}]*\}.*$',   # More flexible
            r'\s*\{.*?"title".*?"content".*?\}.*$',         # Most flexible
        ]
        
        for pattern in patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL)
        
        # Also remove if JSON appears at the end after a newline
        if cleaned_content.rstrip().endswith('}'):
            # Check if it's JSON by looking backwards
            last_brace = cleaned_content.rfind('}')
            if last_brace > 0:
                before_brace = cleaned_content[:last_brace].rstrip()
                if '"title"' in cleaned_content[last_brace-100:last_brace] and '"content"' in cleaned_content[last_brace-100:last_brace]:
                    # Find the matching opening brace
                    brace_pos = before_brace.rfind('{')
                    if brace_pos >= 0:
                        cleaned_content = before_brace[:brace_pos].rstrip()
        
        return cleaned_content.strip()
    
    def _auto_save_report(self, query: str, content: str) -> str:
        """Fallback method to automatically save report if agent didn't call save_report tool."""
        try:
            # Clean the content to remove JSON objects
            cleaned_content = self._clean_content(content)
            
            # Extract title from content or use query
            title = query[:50] + "..." if len(query) > 50 else query
            title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_', '?')).strip()
            
            # Try to extract title from content if it has a proper heading
            if cleaned_content.startswith('#'):
                first_line = cleaned_content.split('\n')[0]
                title_match = re.match(r'^#+\s*(.+)$', first_line)
                if title_match:
                    title = title_match.group(1).strip()
                    # Remove the title from content since we'll add it as header
                    cleaned_content = '\n'.join(cleaned_content.split('\n')[1:]).strip()
            
            if not title:
                title = "Research Report"
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            filename = f"{safe_title}_{timestamp}.md"
            filepath = os.path.join(self.reports_dir, filename)
            
            # Save report with clean content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Query:** {query}\n\n")
                f.write("---\n\n")
                f.write(cleaned_content)
                f.write("\n")
            
            return filepath
        except Exception as e:
            return f"Error auto-saving report: {str(e)}"
    
    def research_with_feedback(self, query: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct research with optional feedback from Review Agent for improvement.
        
        Args:
            query: The research question or topic
            feedback: Optional feedback from Review Agent to improve the research
            
        Returns:
            Dictionary with research results and metadata
        """
        # If feedback is provided, incorporate it into the query
        if feedback and feedback.strip():
            self.console.print("[bold yellow]Incorporating Review Agent feedback to improve research...[/bold yellow]")
            self.console.print(f"[dim]Feedback: {feedback[:200]}...[/dim]\n")
            
            # Extract original query (remove any previous feedback)
            original_query = query.split("\n\nCRITICAL FEEDBACK")[0].split("\n\nIMPORTANT FEEDBACK")[0].split("\n\n════════════════════")[0].strip()
            
            # Create a more structured feedback prompt
            enhanced_query = f"""{original_query}

═══════════════════════════════════════════════════════════════
🚨 CRITICAL FEEDBACK FROM REVIEW AGENT - YOUR PREVIOUS ATTEMPT FAILED
═══════════════════════════════════════════════════════════════

The Review Agent found significant issues with your previous research. You MUST fix these now.

FEEDBACK:
{feedback}

═══════════════════════════════════════════════════════════════
MANDATORY CORRECTIONS - DO NOT REPEAT THESE MISTAKES:
═══════════════════════════════════════════════════════════════

1. SEMANTIC UNDERSTANDING FAILURE:
   - Your previous report did NOT understand what the user REALLY wants
   - You focused on keywords instead of intent
   - You missed implicit questions or assumptions
   - FIX: Analyze the query semantically - what is the REAL underlying question?

2. CRITICAL EVALUATION MISSING:
   - If the user stated an assumption (e.g., "I think it's all about X"), you just confirmed it
   - You provided a one-sided view instead of balanced perspective
   - You didn't evaluate alternatives or related topics
   - FIX: Research the user's topic AND alternatives/related topics
   - FIX: Provide critical analysis, not just confirmation
   - FIX: Structure report to acknowledge assumption, then provide balanced view

3. INCOMPLETE RESEARCH:
   - You only researched what the user mentioned
   - You didn't explore related or alternative perspectives
   - You missed important dimensions of the query
   - FIX: Search for the user's topic AND broader context/alternatives
   - FIX: Get diverse perspectives, not just one angle

4. SPECIFIC ACTIONS BASED ON FEEDBACK:
   - Read the feedback above carefully
   - Identify the specific issues mentioned
   - Change your search strategy to address them
   - If feedback says "only confirms assumption": Research alternatives too
   - If feedback says "missing critical evaluation": Provide balanced analysis
   - If feedback says "poor semantic understanding": Understand the REAL intent

5. REPORT STRUCTURE FOR THIS ITERATION:
   - Start with understanding what the user REALLY wants
   - If user has assumption: Acknowledge it, then critically evaluate
   - Provide balanced perspective with multiple angles
   - Address implicit questions, not just explicit ones
   - Help user think more critically, not just validate their bias

═══════════════════════════════════════════════════════════════
YOUR MISSION: Generate a MUCH BETTER report that:
═══════════════════════════════════════════════════════════════
✓ Understands the semantic intent (not just keywords)
✓ Provides critical evaluation of assumptions
✓ Offers balanced, comprehensive perspective
✓ Addresses ALL aspects of the query
✓ Would satisfy someone asking this question

DO NOT create the same type of report. The Review Agent will check again and will give you a LOW RATING if you repeat the same mistakes!"""
            
            return self.research(enhanced_query)
        else:
            # No feedback, just do normal research
            return self.research(query)
    
    def _analyze_query_semantically(self, query: str) -> str:
        """
        Analyze the query semantically BEFORE researching to understand:
        - What the user REALLY wants (not just keywords)
        - If user has assumptions that need critical evaluation
        - What topics need to be researched for balanced perspective
        
        Returns:
            Enhanced query with semantic analysis and research plan
        """
        analysis_prompt = """Analyze this research query semantically and create a research plan.

Query: {query}

Provide a brief analysis:
1. What is the user REALLY asking? (underlying intent, not just keywords)
2. Does the user state an assumption, hypothesis, or belief? (e.g., "I think it's all about X")
3. If assumption found: What topics should be researched? (the user's topic AND alternatives/related topics)
4. What are the implicit questions that need to be addressed?
5. What search terms would get balanced, comprehensive information?

Format your response as:
INTENT: [What user really wants]
ASSUMPTION DETECTED: [Yes/No - if yes, what is it?]
TOPICS TO RESEARCH: [List topics - include user's focus AND alternatives if assumption found]
SEARCH STRATEGY: [How to get balanced perspective]
IMPLICIT QUESTIONS: [What else needs to be addressed]"""

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a semantic analysis expert. Analyze queries to understand true intent and identify assumptions."),
                ("human", analysis_prompt)
            ])
            
            formatted = prompt.format_messages(query=query)
            analysis_result = self.llm.invoke(formatted)
            analysis = analysis_result.content if hasattr(analysis_result, 'content') else str(analysis_result)
            
            return analysis
        except Exception as e:
            self.console.print(f"[yellow][WARNING] Semantic analysis failed: {e}. Proceeding with enhanced query.[/yellow]")
            return None
    
    def research(self, query: str) -> Dict[str, Any]:
        """
        Conduct research on a given query.
        
        Args:
            query: The research question or topic
            
        Returns:
            Dictionary with research results and metadata
        """
        self.console.print(Panel(
            f"[bold cyan]Research Query:[/bold cyan]\n{query}",
            title="Research Agent",
            border_style="cyan"
        ))
        
        saved_file = None
        
        try:
            # STEP 1: Semantic Analysis BEFORE Researching
            if "CRITICAL FEEDBACK" not in query and "════════════════════" not in query:
                self.console.print("[dim]Analyzing query semantically...[/dim]")
                semantic_analysis = self._analyze_query_semantically(query)
                
                if semantic_analysis:
                    self.console.print(f"[green]✓ Semantic analysis complete[/green]")
                    self.console.print(f"[dim]{semantic_analysis[:200]}...[/dim]\n")
                    
                    # Create enhanced query with semantic understanding
                    enhanced_query = f"""RESEARCH QUERY:
{query}

═══════════════════════════════════════════════════════════════
SEMANTIC ANALYSIS (Do this BEFORE researching):
═══════════════════════════════════════════════════════════════

{semantic_analysis}

═══════════════════════════════════════════════════════════════
YOUR RESEARCH PLAN:
═══════════════════════════════════════════════════════════════

Based on the semantic analysis above:

1. UNDERSTAND THE REAL INTENT: {{
   - What is the user REALLY asking? (not just keywords)
   - What would truly satisfy this query?
   - What are the implicit questions?
}}

2. IDENTIFY ASSUMPTIONS: {{
   - Does the user state a belief/hypothesis?
   - If YES: You MUST critically evaluate it, not just confirm it
   - Research the user's topic AND alternatives/related topics
}}

3. PLAN YOUR SEARCHES: {{
   - Use 2-4 strategic searches
   - If user has assumption: Search for their topic AND alternatives
   - Get diverse perspectives, not just one angle
   - Examples:
     * User's topic: "quantum computing trends 2025"
     * Alternative: "emerging technology trends 2025"
     * Broader: "technology predictions 2025"
     * Related: "AI agents trends 2025"
}}

4. CREATE BALANCED REPORT: {{
   - If user has assumption: Acknowledge it, then provide critical evaluation
   - Present balanced perspective with multiple angles
   - Address implicit questions, not just explicit ones
   - Help user think more critically, not just validate their bias
}}

NOW: Conduct your research following this plan. Use web_search strategically, then synthesize findings into a comprehensive, balanced report."""
                else:
                    # Fallback if analysis fails
                    enhanced_query = f"""RESEARCH QUERY:
{query}

CRITICAL: Before researching, analyze semantically:
1. What is the user REALLY asking? (not just keywords)
2. Does the user state an assumption? (e.g., "I think it's all about X")
3. If assumption found: Research X AND alternatives for balanced perspective
4. Plan searches to get diverse perspectives

Now conduct your research with semantic understanding and critical thinking."""
            else:
                # Query already has feedback, use as-is
                enhanced_query = query
            
            # STEP 2: Execute the agent with enhanced query
            self.console.print("[dim]Starting research with semantic understanding...[/dim]\n")
            result = self.agent.invoke({"input": enhanced_query})
            output = result.get("output", "")
            
            # Check if agent hit max iterations
            if "max iterations" in output.lower() or "stopped due to" in output.lower() or len(output.strip()) < 50:
                self.console.print("[yellow][WARNING] Agent hit iteration limit or produced minimal output. Attempting to save available information...[/yellow]")
                # Try to extract any useful content from intermediate steps
                if hasattr(result, 'intermediate_steps') and result.get('intermediate_steps'):
                    # Extract information from tool calls
                    collected_info = []
                    for step in result.get('intermediate_steps', []):
                        if len(step) > 1 and isinstance(step[1], str):
                            collected_info.append(step[1][:500])  # Get first 500 chars of each tool result
                    
                    if collected_info:
                        combined_content = "\n\n".join(collected_info)
                        basic_report = f"# Research Report\n\n**Query:** {query}\n\n**Note:** Research was limited due to iteration constraints. Below is available information:\n\n{combined_content[:2000]}"
                        saved_file = self._auto_save_report(query, basic_report)
                        self.console.print(f"[green][OK] Partial report saved to: {saved_file}[/green]")
                    else:
                        # Generate a basic report
                        basic_content = f"# Research Report\n\n**Query:** {query}\n\n**Note:** Research was incomplete due to iteration limits. Please try a more specific query or re-run the research."
                        saved_file = self._auto_save_report(query, basic_content)
                        self.console.print(f"[green][OK] Basic report saved to: {saved_file}[/green]")
                else:
                    # Very little content, try to generate a basic report
                    basic_content = f"# Research Report\n\n**Query:** {query}\n\n**Note:** Research was incomplete due to iteration limits. The agent may need more iterations or a more focused query."
                    saved_file = self._auto_save_report(query, basic_content)
                    self.console.print(f"[green][OK] Basic report saved to: {saved_file}[/green]")
            # Check if report was saved by checking if save_report was called
            elif "saved successfully" not in output.lower() and "save_report" not in str(result):
                self.console.print("[yellow][WARNING] Agent didn't explicitly save report. Auto-saving...[/yellow]")
                saved_file = self._auto_save_report(query, output)
                self.console.print(f"[green][OK] Report auto-saved to: {saved_file}[/green]")
            else:
                # Extract filepath from output if available
                if "saved successfully" in output.lower():
                    # Try to extract the filepath from the output
                    lines = output.split('\n')
                    for line in lines:
                        if 'saved successfully' in line.lower() or 'reports/' in line:
                            saved_file = line.split('reports/')[-1] if 'reports/' in line else None
                            break
            
            self.console.print("\n[bold green][OK] Research completed![/bold green]\n")
            
            return {
                "query": query,
                "result": output,
                "status": "success",
                "saved_file": saved_file
            }
        except Exception as e:
            error_msg = str(e)
            self.console.print(f"[bold red][ERROR] Error: {error_msg}[/bold red]")
            
            # If it's an iteration limit error, try to save what we have
            if "max iterations" in error_msg.lower() or "iteration" in error_msg.lower():
                self.console.print("[yellow]Attempting to save partial results...[/yellow]")
                try:
                    saved_file = self._auto_save_report(query, f"# Research Report\n\n**Query:** {query}\n\n**Note:** Research incomplete due to iteration limits.\n\n{error_msg}")
                    return {
                        "query": query,
                        "result": None,
                        "status": "partial",
                        "saved_file": saved_file,
                        "error": error_msg
                    }
                except:
                    pass
            
            return {
                "query": query,
                "result": None,
                "status": "error",
                "error": error_msg
            }


def main():
    """Demo function to showcase the research agent."""
    console.print(Panel(
        "[bold]Research & Analysis Agent Demo[/bold]\n\n"
        "This agent demonstrates:\n"
        "• Multi-step reasoning and planning\n"
        "• Tool use (web search, file operations)\n"
        "• Information synthesis\n"
        "• Automated report generation",
        title="Agentic AI Demo",
        border_style="blue"
    ))
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY not found in environment[/bold red]")
        console.print("Please set your OpenAI API key in a .env file")
        return
    
    # Initialize agent
    agent = ResearchAgent()
    
    # Example queries for demo
    demo_queries = [
        "What are the latest trends in AI agents in 2024?",
        "Compare the top 3 programming languages for AI development",
        "What are the key challenges in deploying LLM agents in production?"
    ]
    
    console.print("\n[bold]Example Research Queries:[/bold]\n")
    for i, query in enumerate(demo_queries, 1):
        console.print(f"{i}. {query}")
    
    console.print("\n[dim]You can modify the queries in the code or add your own.[/dim]\n")
    
    # Run research on first query as demo
    result = agent.research(demo_queries[0])
    
    if result["status"] == "success":
        console.print("\n[bold]Research Output:[/bold]")
        console.print(Markdown(result["result"]))


if __name__ == "__main__":
    main()

