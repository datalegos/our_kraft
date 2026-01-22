#!/usr/bin/env python3
"""
Main entry point for Agentic ETL System
"""

import asyncio
import sys
import os
import yaml
from agents.orchestrator import SimpleOrchestrator


async def run_etl_system():
    """Main execution flow for ETL system"""
    print("🚀 Starting Agentic ETL System...")
    
    # Check if config exists
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ {config_path} not found!")
        return
    
    try:
        # Initialize orchestrator
        orchestrator = SimpleOrchestrator(config_path)
        print("✅ System initialized successfully")
        
        # Load config to check for file path
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        configured_file = config.get('file', {}).get('input_path', '').strip()
        auto_process = config.get('file', {}).get('auto_process', True)
        
        # Check if we have command line arguments
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            print(f"📁 Processing file: {file_path}")
            await process_single_file(orchestrator, file_path)
            
        elif configured_file and os.path.exists(configured_file) and auto_process:
            print(f"📁 Found configured file: {configured_file}")
            await process_single_file(orchestrator, configured_file)
            
        else:
            await interactive_mode(orchestrator)
            
    except Exception as e:
        print(f"❌ System error: {e}")


async def process_single_file(orchestrator, file_path: str):
    """Process a single file and show results"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print("🚀 Processing with 3-Agent Pipeline (ETL → Metrics → Neo4j)...")
    result = await orchestrator.process_file(file_path)
    
    print(f"📊 Status: {result['status']}")
    
    if result['status'] == 'success':
        # ETL Results
        etl_result = result.get('etl_result', {})
        print(f"✅ Agent 1 (ETL): {etl_result.get('agent_output', 'ETL completed')[:100]}...")
        
        # Metrics Results
        metrics_result = result.get('metrics_result', {})
        print(f"📈 Agent 2 (Metrics): {metrics_result.get('agent_output', 'Metrics completed')[:100]}...")
        
        # Neo4j Results
        neo4j_result = result.get('neo4j_result', {})
        print(f"🗄️ Agent 3 (Neo4j): {neo4j_result.get('agent_output', 'Neo4j completed')[:100]}...")
        
        # Data shape
        if result.get('data_shape'):
            print(f"📋 Final data shape: {result['data_shape']}")
        
        # Saved files
        if result.get('saved_files'):
            print(f"💾 Saved files: {', '.join(result['saved_files'])}")
    else:
        print(f"❌ Error: {result.get('error')}")


async def interactive_mode(orchestrator):
    """Simple interactive mode"""
    print("\n🎯 No file configured. Enter file path manually:")
    
    while True:
        print("\n1. 📁 Upload file")
        print("2. 🚪 Exit")
        
        choice = input("\n👉 Choice (1-2): ").strip()
        
        if choice == "1":
            file_path = input("📁 Paste file path: ").strip()
            
            if not file_path:
                print("❌ File path required!")
                continue
                
            await process_single_file(orchestrator, file_path)
                
        elif choice == "2":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice! Please enter 1 or 2")


def main():
    """Entry point"""
    try:
        asyncio.run(run_etl_system())
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()