#!/usr/bin/env python3
"""
Simple Autonomous Neo4j Agent System
A simplified autonomous system for Neo4j database interactions - Demo Version
"""

import json
from typing import Dict, Any
from services.autonomous_orchestrator_agent import AutonomousOrchestratorAgent
from services.autonomous_decision_agent import AutonomousDecisionAgent
from services.ai_query_service import AIQueryService
from services.schema_service import SchemaService
from services.agentic_data_loader import AgenticDataLoader

class FullyAutonomousNeo4jSystem:
    """
    Simple autonomous system for Neo4j interactions
    """
    
    def __init__(self, config_file: str = "ai_agent_config.json"):
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize agents
        self.orchestrator = AutonomousOrchestratorAgent(self.config)
        self.decision_agent = AutonomousDecisionAgent(self.config)
        
        # Initialize services
        self.query_service = AIQueryService(self.config)
        self.schema_service = SchemaService(self.config)
        self.data_loader = AgenticDataLoader(self.config)
        
        # Register services with orchestrator
        self.orchestrator.register_service("ai_query_service", self.query_service)
        self.orchestrator.register_service("schema_service", self.schema_service)
        self.orchestrator.register_service("agentic_data_loader", self.data_loader)
        
        print("🤖 Simple Autonomous Neo4j Agent System Initialized")
        print(f"🎯 Agents: Orchestrator, Decision Agent")
        print(f"⚙️ Services: Query, Schema, Data Loader")
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """
        Process user request using simple autonomous logic
        """
        
        print(f"\n🎯 Processing: {user_input}")
        print("=" * 50)
        
        try:
            # Let the orchestrator handle the request
            result = self.orchestrator.orchestrate_request(user_input)
            
            # Display results
            self._display_results(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error': str(e),
                'response': f"Error processing request: {str(e)}"
            }
            
            print(f"❌ Error: {str(e)}")
            return error_result
    
    def _display_results(self, result: Dict[str, Any]):
        """Display results simply"""
        
        status = result.get('status', 'unknown')
        response = result.get('response', 'No response')
        
        if status == 'success':
            print(f"✅ Success")
            print(f"🤖 {response}")
            
            if 'goal' in result:
                print(f"🎯 Goal: {result['goal']}")
            
            if 'action_taken' in result:
                print(f"⚡ Action: {result['action_taken']}")
        
        else:
            print(f"❌ {status.title()}")
            print(f"🤖 {response}")
    
    def interactive_mode(self):
        """Simple interactive mode"""
        
        print("\n🚀 Welcome to Simple Autonomous Neo4j Agent!")
        print("💡 I can handle Neo4j requests autonomously.")
        print("📝 Examples:")
        print("   • 'Show me all students'")
        print("   • 'Load data from CSV'")
        print("   • 'What's the database schema?'")
        print("\n💬 Type 'help' for more info or 'exit' to quit.")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n🎤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Process the request
                result = self.process_request(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Show simple help"""
        
        print("\n📚 Simple Autonomous Neo4j Agent - Help")
        print("=" * 45)
        print("🎯 I can handle these requests autonomously:")
        print("\n📊 Data Queries:")
        print("   • 'Show me all students'")
        print("   • 'How many students are placed?'")
        print("   • 'Find students with high CGPA'")
        print("\n📁 Data Loading:")
        print("   • 'Load data from CSV'")
        print("   • 'Import new dataset'")
        print("\n🏗️ Schema:")
        print("   • 'Show database schema'")
        print("   • 'What data is available?'")
        print("\n🔧 Commands:")
        print("   • 'help' - Show this help")
        print("   • 'exit' - Quit")

def main():
    """Main function"""
    
    try:
        # Initialize the system
        system = FullyAutonomousNeo4jSystem()
        
        # Run interactive mode
        system.interactive_mode()
        
    except FileNotFoundError:
        print("❌ Configuration file 'ai_agent_config.json' not found!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    main()