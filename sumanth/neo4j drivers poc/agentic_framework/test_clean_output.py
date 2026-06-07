#!/usr/bin/env python3
"""
Test Clean Output - Verify the system produces clean responses
"""

import json
import sys
import os

# Add the simple_ai_system to path to import services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system'))

from agents.query_agent import QueryAgent
from services.ai_query_service import AIQueryService
from services.schema_service import SchemaService
from services.agentic_data_loader import AgenticDataLoader

def test_clean_output():
    """Test that the system produces clean, readable output"""
    
    print("🧪 Testing Clean Output Format")
    print("=" * 50)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system', 'ai_agent_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Configuration file not found!")
        return False
    
    try:
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
        
        print("\n📋 Result:")
        print(f"Status: {result['status']}")
        print(f"Agent: {result['agent_type']}")
        print(f"Response: {result['response']}")
        
        # Check if response is clean
        response = result['response']
        if isinstance(response, str) and len(response) < 200 and 'messages' not in response.lower():
            print("\n✅ Output format is clean and readable!")
            return True
        else:
            print("\n❌ Output format needs improvement")
            print(f"Response length: {len(response)} characters")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    
    success = test_clean_output()
    
    if success:
        print("\n🎉 Clean output test passed!")
    else:
        print("\n❌ Clean output test failed.")

if __name__ == "__main__":
    main()