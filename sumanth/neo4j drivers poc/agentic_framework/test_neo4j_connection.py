#!/usr/bin/env python3
"""
Neo4j Connection Test Script
Test your Neo4j database connection and credentials
"""

import json
import sys
import os
from neo4j import GraphDatabase

def test_neo4j_connection():
    """Test Neo4j connection with current configuration"""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system', 'ai_agent_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Configuration file not found!")
        print(f"Expected location: {config_path}")
        return False
    
    # Get database config (handle both 'neo4j' and 'database' keys)
    db_config = config.get('neo4j', config.get('database', {}))
    
    if not db_config:
        print("❌ No database configuration found!")
        print("Please add 'neo4j' or 'database' section to your config file.")
        return False
    
    # Extract connection details
    uri = db_config.get('uri', 'bolt://localhost:7687')
    username = db_config.get('username', 'neo4j')
    password = db_config.get('password', '')
    
    print("🔧 Testing Neo4j Connection...")
    print(f"   URI: {uri}")
    print(f"   Username: {username}")
    print(f"   Password: {'*' * len(password) if password else 'NOT SET'}")
    print()
    
    if not password:
        print("❌ Password is not set in configuration!")
        return False
    
    # Test connection
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Test basic connectivity
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
            if test_value == 1:
                print("✅ Connection successful!")
                
                # Get database info
                try:
                    db_info = session.run("CALL dbms.components() YIELD name, versions, edition")
                    for record in db_info:
                        print(f"   Database: {record['name']} {record['versions'][0]} ({record['edition']})")
                except:
                    print("   Database info not available")
                
                # Test if we can query data
                try:
                    count_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                    total_nodes = count_result.single()["total_nodes"]
                    print(f"   Total nodes in database: {total_nodes}")
                    
                    if total_nodes > 0:
                        # Get node labels
                        labels_result = session.run("CALL db.labels()")
                        labels = [record["label"] for record in labels_result]
                        if labels:
                            print(f"   Node labels: {', '.join(labels)}")
                    else:
                        print("   Database is empty (no nodes found)")
                        
                except Exception as e:
                    print(f"   Could not query database: {e}")
                
                driver.close()
                return True
            else:
                print("❌ Connection test failed!")
                driver.close()
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
        # Provide helpful error messages
        error_str = str(e).lower()
        if "authentication" in error_str:
            print("\n💡 Authentication Error Solutions:")
            print("   1. Check your username and password")
            print("   2. Reset Neo4j password: neo4j-admin set-initial-password <new-password>")
            print("   3. Or use Neo4j Desktop to reset password")
        elif "connection refused" in error_str or "failed to establish connection" in error_str:
            print("\n💡 Connection Error Solutions:")
            print("   1. Make sure Neo4j is running")
            print("   2. Check if Neo4j is listening on the correct port (7687)")
            print("   3. Verify the URI is correct")
        elif "rate limit" in error_str:
            print("\n💡 Rate Limit Error Solutions:")
            print("   1. Wait a few minutes before trying again")
            print("   2. Restart Neo4j database")
            print("   3. Check Neo4j logs for more details")
        
        return False

def main():
    """Main function"""
    
    print("🧪 Neo4j Connection Test")
    print("=" * 50)
    
    success = test_neo4j_connection()
    
    print()
    if success:
        print("🎉 Connection test passed! Your agentic system should work.")
    else:
        print("❌ Connection test failed. Please fix the issues above.")
        print("\n📝 Quick Setup Reminder:")
        print("   1. Start Neo4j database")
        print("   2. Set correct credentials in ai_agent_config.json")
        print("   3. Make sure Neo4j is accessible on bolt://localhost:7687")

if __name__ == "__main__":
    main()