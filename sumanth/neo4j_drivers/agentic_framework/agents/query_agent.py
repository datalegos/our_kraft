"""
Query Agent - Specialized for direct database queries and data retrieval
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

class QueryAgent:
    """
    Specialized agent for direct database queries and data retrieval
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
            temperature=0.0  # Very focused for queries
        )
        
        # Simple conversation history (we'll manage this manually)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Create tools (focused on query and schema)
        all_tools = create_neo4j_tools(query_service, schema_service, data_loader)
        self.tools = [tool for tool in all_tools if tool.name in ['neo4j_query', 'neo4j_schema']]
        
        # Create agent
        self._create_agent()
    
    def _create_agent(self):
        """Create the query-focused agent"""
        
        system_prompt = """You are a Query Agent specialized in direct database operations and data retrieval.

SPECIALIZATION:
• Execute Cypher queries efficiently
• Convert natural language to optimized Cypher
• Retrieve specific data quickly
• Analyze database schema and structure

AVAILABLE TOOLS:
• neo4j_query: Execute Cypher queries or convert natural language to queries
• neo4j_schema: Get database structure information

BEHAVIOR:
• Focus on direct, efficient data retrieval
• Provide precise, factual responses
• Optimize queries for performance
• Handle schema-related questions expertly

DATABASE CONTEXT:
• Student placement database (~45,000 records)
• Key properties: student_id, cgpa, communication_skills, placement_status
• Focus on accurate, fast data access

Be direct, efficient, and precise in your responses."""

        # Create the agent using the newer API
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            checkpointer=None  # We'll handle memory differently
        )
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process query-focused requests"""
        
        try:
            print(f"\n🔍 Query Agent Processing: {user_input}")
            
            # Add to conversation history
            self.conversation_history.append({"role": "human", "content": user_input})
            
            # Use the newer agent API
            result = self.agent.invoke({"messages": [("human", user_input)]})
            
            # Extract the clean response from the result
            response = "Query completed successfully."  # Default fallback
            
            # Handle different result formats
            if isinstance(result, dict):
                # New format: result is a dictionary
                if 'messages' in result:
                    messages = result['messages']
                    
                    # Look for the final AI response
                    for message in reversed(messages):
                        if hasattr(message, 'content') and message.content:
                            content = message.content.strip()
                            if content and 'tool_calls' not in str(type(message)).lower():
                                response = content
                                break
                
                # Check for direct response in dict
                elif 'output' in result:
                    response = result['output']
                elif 'content' in result:
                    response = result['content']
            
            elif hasattr(result, 'messages') and result.messages:
                # Old format: result has .messages attribute
                for message in reversed(result.messages):
                    if hasattr(message, 'content') and message.content:
                        content = message.content.strip()
                        if content:
                            response = content
                            break
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep only last 8 messages (4 exchanges)
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            return {
                'status': 'success',
                'response': response,
                'agent_type': 'query_agent',
                'tools_used': [tool.name for tool in self.tools],
                'specialization': 'Direct queries and data retrieval'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response': f"Query Agent error: {str(e)}"
            }
    
    def clear_memory(self):
        """Clear agent memory"""
        self.conversation_history = []