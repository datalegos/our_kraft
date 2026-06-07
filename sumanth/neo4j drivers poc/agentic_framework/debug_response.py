#!/usr/bin/env python3
"""
Debug Response Extraction
Test what the agent is actually returning
"""

import json
import sys
import os

# Add the simple_ai_system to path to import services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system'))

def debug_response():
    """Debug the response extraction issue"""
    
    print("🔍 Debugging Response Extraction")
    print("=" * 50)
    
    try:
        from agents.query_agent import QueryAgent
        from services.ai_query_service import AIQueryService
        from services.schema_service import SchemaService
        from services.agentic_data_loader import AgenticDataLoader
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system', 'ai_agent_config.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize services
        print("🔧 Initializing services...")
        query_service = AIQueryService(config)
        schema_service = SchemaService(config)
        data_loader = AgenticDataLoader(config)
        
        # Initialize Query Agent
        print("🤖 Initializing Query Agent...")
        query_agent = QueryAgent(config, query_service, schema_service, data_loader)
        
        # Test a simple query
        print("📊 Testing query: 'How many students are there?'")
        result = query_agent.process_request("How many students are there?")
        
        print("\n📋 Final Result:")
        print(f"Status: {result['status']}")
        print(f"Response: '{result['response']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_response()