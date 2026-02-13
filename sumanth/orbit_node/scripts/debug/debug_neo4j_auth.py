"""
Debug Neo4j Authentication
Tests different authentication methods to identify the issue.
"""

import yaml
from neo4j import GraphDatabase
import sys

def test_different_auth_methods():
    """Test different authentication methods"""
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
        print("Neo4j Authentication Debug")
        print("=" * 60)
        print(f"URI: {uri}")
        print(f"Username: {username}")
        print(f"Password: {repr(password)}")
        print(f"Database: {database}")
        print("=" * 60)
        
        # Test 1: Basic connection with tuple auth
        print("\n[Test 1] Basic connection with tuple auth...")
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    print("  [SUCCESS] Test 1 passed!")
                    driver.close()
                    return True
        except Exception as e:
            print(f"  [FAILED] {e}")
            if 'driver' in locals():
                driver.close()
        
        # Test 2: Try with different URI formats
        print("\n[Test 2] Trying neo4j:// URI (routing)...")
        try:
            neo4j_uri = uri.replace('bolt://', 'neo4j://')
            driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    print("  [SUCCESS] Test 2 passed with neo4j:// URI!")
                    driver.close()
                    return True
        except Exception as e:
            print(f"  [FAILED] {e}")
            if 'driver' in locals():
                driver.close()
        
        # Test 3: Try with BasicAuth explicitly
        print("\n[Test 3] Trying with BasicAuth class...")
        try:
            from neo4j import basic_auth
            driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    print("  [SUCCESS] Test 3 passed with BasicAuth!")
                    driver.close()
                    return True
        except Exception as e:
            print(f"  [FAILED] {e}")
            if 'driver' in locals():
                driver.close()
        
        # Test 4: Check if password needs URL encoding
        print("\n[Test 4] Checking password encoding...")
        import urllib.parse
        encoded_password = urllib.parse.quote(password, safe='')
        print(f"  Original: {repr(password)}")
        print(f"  URL encoded: {repr(encoded_password)}")
        if password != encoded_password:
            print("  Note: Password contains special characters that would be URL encoded")
        
        # Test 5: Try without database parameter (use default)
        print("\n[Test 5] Trying without explicit database parameter...")
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session() as session:  # No database parameter
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    print("  [SUCCESS] Test 5 passed without database parameter!")
                    driver.close()
                    return True
        except Exception as e:
            print(f"  [FAILED] {e}")
            if 'driver' in locals():
                driver.close()
        
        # Test 6: Try with different connection parameters
        print("\n[Test 6] Trying with minimal connection parameters...")
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=False)
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    print("  [SUCCESS] Test 6 passed with minimal params!")
                    driver.close()
                    return True
        except Exception as e:
            print(f"  [FAILED] {e}")
            if 'driver' in locals():
                driver.close()
        
        print("\n" + "=" * 60)
        print("All tests failed. Possible issues:")
        print("  1. Password might be different for Bolt vs HTTP (Browser)")
        print("  2. Neo4j might require password change on first Bolt connection")
        print("  3. There might be a firewall or network issue")
        print("  4. The Bolt protocol might be disabled")
        print("=" * 60)
        
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_different_auth_methods()
    sys.exit(0 if success else 1)

