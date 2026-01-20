"""
Simple AI-Powered Agent Router
Uses LLM to intelligently route requests to the most appropriate agent
"""

from langchain_openai import ChatOpenAI
from typing import Dict, Any

class SimpleAgentRouter:
    """
    AI-powered router that selects the best agent for each request
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize LLM for routing decisions
        self.llm = ChatOpenAI(
            model=config['llm']['model'],
            api_key=config['llm']['api_key'],
            temperature=0.0  # Deterministic routing
        )
    
    def route_request(self, user_input: str) -> Dict[str, Any]:
        """
        Use AI to select the best agent for the request
        """
        
        routing_prompt = f"""You are an intelligent agent router. Analyze the user request and select the most appropriate agent.

AVAILABLE AGENTS:

1. QUERY AGENT - Best for:
   - Direct database queries
   - Specific data retrieval
   - Cypher query execution
   - Schema information requests
   - Simple lookups and searches
   - "Show me...", "Find...", "Get...", "How many..."

2. ANALYTICS AGENT - Best for:
   - Data analysis and insights
   - Statistical analysis
   - Trend identification
   - Complex reasoning tasks
   - Multi-step analysis
   - Reports and correlations
   - "Analyze...", "What factors...", "Insights...", "Trends..."

USER REQUEST: "{user_input}"

DECISION PROCESS:
1. Is this a direct data retrieval request? → Query Agent
2. Does this require analysis, insights, or complex reasoning? → Analytics Agent
3. When in doubt, prefer Analytics Agent for comprehensive responses

Respond with JSON format:
{{
    "selected_agent": "query" or "analytics",
    "reasoning": "Specific explanation of why this agent was chosen based on the request type and complexity"
}}

Be specific in your reasoning - explain what about the request made you choose this agent."""
        
        try:
            response = self.llm.invoke(routing_prompt)
            response_text = response.content.strip()
            
            # Try to parse JSON response
            try:
                import json
                routing_decision = json.loads(response_text)
                selected_agent = routing_decision.get('selected_agent', 'analytics').lower()
                reasoning = routing_decision.get('reasoning', 'AI routing decision')
            except json.JSONDecodeError:
                # Fallback parsing if JSON fails
                response_lower = response_text.lower()
                if 'query' in response_lower:
                    selected_agent = 'query'
                    reasoning = "Detected direct query request requiring specific data retrieval"
                elif 'analytics' in response_lower:
                    selected_agent = 'analytics'
                    reasoning = "Detected complex request requiring analysis and insights"
                else:
                    selected_agent = 'analytics'
                    reasoning = "Defaulting to analytics agent for comprehensive handling"
            
            # Validate response
            if selected_agent in ['query', 'analytics']:
                return {
                    'status': 'success',
                    'selected_agent': selected_agent,
                    'reasoning': reasoning
                }
            else:
                # Fallback to analytics for complex requests
                return {
                    'status': 'fallback',
                    'selected_agent': 'analytics',
                    'reasoning': 'Fallback to analytics agent for comprehensive handling'
                }
                
        except Exception as e:
            # Fallback on error
            return {
                'status': 'error',
                'selected_agent': 'analytics',
                'reasoning': f'Router error, using analytics agent: {str(e)}'
            }