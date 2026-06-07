"""
Analytics Agent - Specialized for data analysis, insights, and complex reasoning
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from typing import Dict, Any, List
import sys
import os
import sys
import os

# Add tools to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.neo4j_tools import create_neo4j_tools

class AnalyticsAgent:
    """
    Specialized agent for data analysis, insights, and complex reasoning
    """
    
    def __init__(self, config: Dict[str, Any], query_service, schema_service, data_loader):
        self.config = config
        self.query_service = query_service
        self.schema_service = schema_service
        self.data_loader = data_loader
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config['llm']['model'],
            api_key=config['llm']['api_key'],
            temperature=0.2  # Slightly more creative for analysis
        )
        
        # Simple conversation history (we'll manage this manually)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Create all tools (analytics needs full access)
        self.tools = create_neo4j_tools(query_service, schema_service, data_loader)
        
        # Create agent
        self._create_agent()
    
    def _create_agent(self):
        """Create the analytics-focused agent"""
        
        system_prompt = """You are an Analytics Agent specialized in data analysis, insights, and complex reasoning.

SPECIALIZATION:
• Perform statistical analysis and correlations
• Generate comprehensive insights and reports
• Identify trends and patterns in data
• Load and analyze new datasets
• Multi-step analytical reasoning

AVAILABLE TOOLS:
• neo4j_query: Execute complex analytical queries
• neo4j_schema: Understand data structure for analysis
• neo4j_data_loader: Load and process new datasets
• neo4j_analytics: Perform advanced statistical analysis

BEHAVIOR:
• Think analytically and provide deep insights
• Use multiple tools to build comprehensive understanding
• Explain correlations and patterns clearly
• Provide actionable recommendations
• Handle complex multi-step analysis tasks

DATABASE CONTEXT:
• Student placement database (~45,000 records)
• Focus on placement success factors, trends, and insights
• Analyze relationships between skills, grades, and outcomes

Be analytical, insightful, and thorough in your responses."""

        # Create the agent using the newer API
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            checkpointer=None  # We'll handle memory differently
        )
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process analytics-focused requests"""
        
        try:
            print(f"\n📊 Analytics Agent Processing: {user_input}")
            
            # Add to conversation history
            self.conversation_history.append({"role": "human", "content": user_input})
            
            # Use the newer agent API
            result = self.agent.invoke({"messages": [("human", user_input)]})
            
            # Extract the clean response from the result
            if hasattr(result, 'messages') and result.messages:
                # Find the last AI message that contains actual content (not tool calls)
                response = None
                for message in reversed(result.messages):
                    if (hasattr(message, 'content') and 
                        message.content and 
                        not hasattr(message, 'tool_calls') and
                        message.content.strip()):
                        response = message.content.strip()
                        break
                
                if not response:
                    # If no clean response found, extract from the last message
                    last_message = result.messages[-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        response = last_message.content.strip()
                    else:
                        response = "Analysis completed successfully."
            else:
                response = "Analysis completed successfully."
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep only last 12 messages (6 exchanges)
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]
            
            return {
                'status': 'success',
                'response': response,
                'agent_type': 'analytics_agent',
                'tools_used': [tool.name for tool in self.tools],
                'specialization': 'Data analysis and insights'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response': f"Analytics Agent error: {str(e)}"
            }
    
    def clear_memory(self):
        """Clear agent memory"""
        self.conversation_history = []