#!/usr/bin/env python3
"""
Simple Two-Agent LangChain Agentic Neo4j System
Clean demonstration of multi-agent agentic behavior with AI-powered routing
"""

import json
import sys
import os
from typing import Dict, Any

# Add the simple_ai_system to path to import services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system'))

from agents.query_agent import QueryAgent
from agents.analytics_agent import AnalyticsAgent
from router import SimpleAgentRouter
from services.ai_query_service import AIQueryService
from services.schema_service import SchemaService
from services.agentic_data_loader import AgenticDataLoader

class TwoAgentNeo4jSystem:
    """
    Simple two-agent system demonstrating AI-powered routing and specialized agents
    """
    
    def __init__(self, config_file: str = "../simple_ai_system/ai_agent_config.json"):
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize Neo4j services
        self.query_service = AIQueryService(self.config)
        self.schema_service = SchemaService(self.config)
        self.data_loader = AgenticDataLoader(self.config)
        
        # Initialize AI-powered router
        self.router = SimpleAgentRouter(self.config)
        
        # Initialize specialized agents
        self.query_agent = QueryAgent(
            self.config, 
            self.query_service, 
            self.schema_service, 
            self.data_loader
        )
        
        self.analytics_agent = AnalyticsAgent(
            self.config,
            self.query_service,
            self.schema_service,
            self.data_loader
        )
        
        print("🤖 Two-Agent LangChain Neo4j System Ready")
        print(f"🧠 Model: {self.config['llm']['model']}")
        print(f"🎯 Agents: Query Agent + Analytics Agent")
        print(f"🔀 Router: AI-powered (no keyword matching)")
        print(f"💡 Features: Specialized agents, autonomous routing")
    
    def process_request(self, user_input: str, agent_type: str = None) -> Dict[str, Any]:
        """Process user request with AI-powered agent routing"""
        
        # Use AI router if no specific agent requested
        if not agent_type:
            routing_result = self.router.route_request(user_input)
            agent_type = routing_result['selected_agent']
            
            print(f"\n🧠 AI Router Decision:")
            print(f"   Selected: {agent_type.title()} Agent")
            print(f"   Reasoning: {routing_result['reasoning']}")
            print()
        
        try:
            if agent_type == 'query':
                return self.query_agent.process_request(user_input)
            elif agent_type == 'analytics':
                return self.analytics_agent.process_request(user_input)
            else:
                return {
                    'status': 'error',
                    'error': f'Unknown agent type: {agent_type}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response': f"System error: {str(e)}"
            }
    
    def interactive_mode(self):
        """Simple interactive mode with two specialized agents"""
        
        print("\n🚀 Welcome to Two-Agent Agentic Neo4j System!")
        print("💡 I have two specialized agents that work together:")
        print("   🔍 Query Agent: Direct database queries and data retrieval")
        print("   📊 Analytics Agent: Data analysis, insights, and complex reasoning")
        print("🧠 AI Router automatically selects the best agent for each request")
        print("\n📝 Try asking:")
        print("   • 'Show me students with CGPA > 8.5' (→ Query Agent)")
        print("   • 'Analyze placement success factors' (→ Analytics Agent)")
        print("   • 'How many students are placed?' (→ Query Agent)")
        print("   • 'What trends do you see in the data?' (→ Analytics Agent)")
        print("\n💬 Commands: 'help', 'agents', 'status', 'clear', 'exit'")
        print("=" * 70)
        
        while True:
            try:
                user_input = input(f"\n🎤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using the Two-Agent System!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'agents':
                    self._show_agents()
                    continue
                
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.query_agent.clear_memory()
                    self.analytics_agent.clear_memory()
                    print("🧹 All agent memory cleared!")
                    continue
                
                # Process with AI-powered routing
                result = self.process_request(user_input)
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ System error: {e}")
    
    def _display_result(self, result: Dict[str, Any]):
        """Display results with clean formatting"""
        
        if result['status'] == 'success':
            print(f"\n{result['response']}")
        else:
            print(f"\n❌ Error: {result['response']}")
    
    def _show_help(self):
        """Show help for two-agent system"""
        
        print("\n📚 Two-Agent System Help")
        print("=" * 40)
        print("🎯 How it works:")
        print("   • AI Router analyzes your request")
        print("   • Selects the most appropriate specialized agent")
        print("   • No keyword matching - pure AI reasoning")
        print("\n🔍 Query Agent - Best for:")
        print("   • 'Show me students with high CGPA'")
        print("   • 'How many students are placed?'")
        print("   • 'Find students in Computer Science'")
        print("   • 'Get database schema information'")
        print("\n📊 Analytics Agent - Best for:")
        print("   • 'Analyze placement success factors'")
        print("   • 'What trends do you see?'")
        print("   • 'Generate insights about the data'")
        print("   • 'Load and analyze new CSV data'")
        print("\n💡 The AI router automatically chooses the right agent!")
    
    def _show_agents(self):
        """Show information about both agents"""
        
        print("\n🤖 Available Agents")
        print("=" * 40)
        print("🔍 Query Agent:")
        print("   • Specialization: Direct queries and data retrieval")
        print("   • Tools: neo4j_query, neo4j_schema")
        print("   • Focus: Fast, precise data access")
        print("   • Memory: 8 messages")
        print("\n📊 Analytics Agent:")
        print("   • Specialization: Data analysis and insights")
        print("   • Tools: All 4 tools (query, schema, loader, analytics)")
        print("   • Focus: Complex reasoning and analysis")
        print("   • Memory: 12 messages")
        print("\n🔀 AI Router:")
        print("   • Uses OpenAI LLM for intelligent routing")
        print("   • No hardcoded rules or keyword matching")
        print("   • Selects optimal agent based on request analysis")
    
    def _show_status(self):
        """Show system status"""
        
        print("\n📊 Two-Agent System Status")
        print("=" * 40)
        print(f"🧠 Model: {self.config['llm']['model']}")
        print(f"🎯 Architecture: Two specialized agents")
        print(f"🔀 Routing: AI-powered (no keywords)")
        print(f"🔧 Total Tools: 4 (distributed across agents)")
        print(f"💭 Memory: Query(8) + Analytics(12) messages")
        print(f"🎨 Features: Specialization, autonomous routing, error recovery")

def main():
    """Main function"""
    
    try:
        # Initialize two-agent system
        system = TwoAgentNeo4jSystem()
        
        # Run interactive mode
        system.interactive_mode()
        
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print("Please ensure ai_agent_config.json exists in the simple_ai_system directory.")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install langchain langchain-openai")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    main()