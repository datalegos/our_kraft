#!/usr/bin/env python3
"""
Neo4j Rate Limit Reset Helper
Helps resolve Neo4j authentication rate limit issues
"""

import time
import json
import os
from neo4j import GraphDatabase

def reset_rate_limit():
    """Help reset Neo4j rate limit by waiting and testing connection"""
    
    print("🔄 Neo4j Rate Limit Reset Helper")
    print("=" * 50)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system', 'ai_agent_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Configuration file not found!")
        return False
    
    # Get database config
    db_config = config.get('neo4j', config.get('database', {}))
    uri = db_config.get('uri', 'bolt://localhost:7687')
    username = db_config.get('username', 'neo4j')
    password = db_config.get('password', '')
    
    print(f"📍 Target: {uri}")
    print(f"👤 User: {username}")
    print()
    
    print("⏳ Waiting for rate limit to reset...")
    print("   (This may take 5-10 minutes)")
    
    # Wait and test connection periodically
    wait_times = [30, 60, 120, 300, 600]  # 30s, 1m, 2m, 5m, 10m
    
    for i, wait_time in enumerate(wait_times):
        print(f"\n🕐 Waiting {wait_time} seconds... (attempt {i+1}/{len(wait_times)})")
        
        for remaining in range(wait_time, 0, -10):
            print(f"   ⏱️  {remaining} seconds remaining...", end='\r')
            time.sleep(10)
        
        print("\n🧪 Testing connection...")
        
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
                if test_value == 1:
                    print("✅ Connection successful! Rate limit has been reset.")
                    driver.close()
                    return True
            
            driver.close()
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "authenticationratelimit" in error_str:
                print("❌ Rate limit still active, continuing to wait...")
                continue
            else:
                print(f"❌ Different error occurred: {e}")
                return False
    
    print("\n❌ Rate limit still active after maximum wait time.")
    print("\n💡 Manual Solutions:")
    print("   1. Restart your Neo4j database service")
    print("   2. Wait longer (rate limits can last up to 30 minutes)")
    print("   3. Check Neo4j logs for more information")
    print("   4. Consider using Neo4j Desktop to reset the database")
    
    return False

def main():
    """Main function"""
    
    success = reset_rate_limit()
    
    if success:
        print("\n🎉 Rate limit resolved! You can now use the agentic system.")
    else:
        print("\n❌ Could not resolve rate limit automatically.")
        print("   Please try the manual solutions listed above.")

if __name__ == "__main__":
    main()