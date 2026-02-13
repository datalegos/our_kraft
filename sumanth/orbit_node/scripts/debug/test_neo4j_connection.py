"""
Simple Neo4j Connection Test Script
Tests connection to Neo4j with current configuration.
"""

import yaml
from neo4j import GraphDatabase
import sys

def test_connection():
    """Test Neo4j connection"""
    try:
        # Load config
        with open('neo4j_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        neo4j_config = config.get('neo4j', {})
        uri = neo4j_config.get('uri', 'bolt://localhost:7687')
        username = neo4j_config.get('username', 'neo4j')
        password = neo4j_config.get('password', '')
        database = neo4j_config.get('database', 'neo4j')
        
        print("=" * 60)
        print("Neo4j Connection Test")
        print("=" * 60)
        print(f"URI: {uri}")
        print(f"Username: {username}")
        print(f"Password: {'*' * len(password) if password else 'NOT SET'} (length: {len(password)})")
        print(f"Database: {database}")
        print("=" * 60)
        
        # Debug: Check for special characters or whitespace
        if password:
            print(f"\nPassword details:")
            print(f"  Length: {len(password)}")
            print(f"  Has leading/trailing spaces: {password != password.strip()}")
            print(f"  Contains special chars: {any(c in password for c in '@#$%^&*()')}")
            print(f"  First char: '{password[0] if password else 'N/A'}'")
            print(f"  Last char: '{password[-1] if password else 'N/A'}'")
            # Show password with repr to see hidden characters
            print(f"  Password repr: {repr(password)}")
        
        if not password:
            print("ERROR: Password is not set in neo4j_config.yaml")
            return False
        
        # Try to connect
        print("\nAttempting to connect...")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record:
                print("[SUCCESS] Connection successful!")
                print(f"  Test result: {record['test']}")
                
                # Get Neo4j version
                version_result = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] as version, edition")
                version_record = version_result.single()
                if version_record:
                    print(f"  Neo4j {version_record['name']}: {version_record['version']} ({version_record['edition']})")
                
                return True
            else:
                print("[ERROR] Connection test returned no result")
                return False
        
    except FileNotFoundError:
        print("ERROR: neo4j_config.yaml not found")
        return False
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse neo4j_config.yaml: {e}")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] Connection failed!")
        print(f"  Error: {error_msg}")
        
        if "authentication failure" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print("\nTroubleshooting:")
            print("  1. Verify Neo4j is running:")
            print("     - Check if Neo4j service/process is running")
            print("     - Try accessing Neo4j Browser at http://localhost:7474")
            print("  2. Verify credentials:")
            print("     - Check username and password in neo4j_config.yaml")
            print("     - Try logging into Neo4j Browser with the same credentials")
            print("  3. If this is a fresh install:")
            print("     - Default username is 'neo4j'")
            print("     - You may need to change the default password on first login")
            print("     - Update neo4j_config.yaml with the new password")
        elif "could not be resolved" in error_msg.lower() or "connection refused" in error_msg.lower():
            print("\nTroubleshooting:")
            print("  1. Verify Neo4j is running")
            print("  2. Check if the URI is correct (default: bolt://localhost:7687)")
            print("  3. Verify the port is not blocked by firewall")
        
        return False
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

