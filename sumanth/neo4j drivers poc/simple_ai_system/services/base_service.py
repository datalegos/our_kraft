"""
Base Service Class
Provides common functionality for all Neo4j services
"""

from neo4j import GraphDatabase
from typing import Dict, Any, List, Optional
import logging

class BaseService:
    """Base class for all Neo4j services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Handle both 'neo4j' and 'database' keys for backward compatibility
        self.neo4j_config = config.get('neo4j', config.get('database', {}))
        
        # Initialize Neo4j driver with connection pooling settings
        self.driver = GraphDatabase.driver(
            self.neo4j_config.get('uri', 'bolt://localhost:7687'),
            auth=(
                self.neo4j_config.get('username', 'neo4j'),
                self.neo4j_config.get('password', 'password')
            ),
            # Add connection pooling and timeout settings
            max_connection_lifetime=30 * 60,  # 30 minutes
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,  # 60 seconds
            connection_timeout=30,  # 30 seconds
            keep_alive=True
        )
        
        # Get the database name from config
        self.database_name = self.neo4j_config.get('database', 'neo4j')
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        print(f"🔗 Connecting to Neo4j database: '{self.database_name}'")
        
        # Test connection on initialization
        if not self.test_connection():
            print("⚠️  Warning: Neo4j connection test failed during initialization")
            print("   The system may not work properly until connection is fixed")
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        
        if parameters is None:
            parameters = {}
        
        try:
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle specific Neo4j errors
            if "authenticationratelimit" in error_str or "rate limit" in error_str:
                self.logger.error("❌ Neo4j authentication rate limit exceeded")
                print("❌ Neo4j Rate Limit Error!")
                print("💡 Solutions:")
                print("   1. Wait 5-10 minutes before trying again")
                print("   2. Restart your Neo4j database")
                print("   3. Check if multiple applications are connecting")
                raise Exception("Neo4j rate limit exceeded. Please wait and try again.")
            
            elif "authentication" in error_str:
                self.logger.error("❌ Neo4j authentication failed")
                print("❌ Neo4j Authentication Error!")
                print("💡 Check your username and password in ai_agent_config.json")
                raise Exception("Neo4j authentication failed. Check credentials.")
            
            else:
                self.logger.error(f"Query execution failed: {e}")
                raise
    
    def execute_write_query(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a write query and return summary"""
        
        if parameters is None:
            parameters = {}
        
        try:
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query, parameters)
                summary = result.consume()
                return {
                    'nodes_created': summary.counters.nodes_created,
                    'relationships_created': summary.counters.relationships_created,
                    'properties_set': summary.counters.properties_set,
                    'labels_added': summary.counters.labels_added,
                    'constraints_added': summary.counters.constraints_added,
                    'indexes_added': summary.counters.indexes_added
                }
        except Exception as e:
            self.logger.error(f"Write query execution failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test Neo4j database connection"""
        
        try:
            with self.driver.session(database=self.database_name) as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()["test"] == 1
                if test_result:
                    self.logger.info(f"✅ Neo4j connection successful to database '{self.database_name}'")
                return test_result
        except Exception as e:
            self.logger.error(f"❌ Neo4j connection failed: {e}")
            print(f"❌ Neo4j Connection Error: {e}")
            print(f"🔧 Check your credentials:")
            print(f"   URI: {self.neo4j_config.get('uri', 'bolt://localhost:7687')}")
            print(f"   Username: {self.neo4j_config.get('username', 'neo4j')}")
            print(f"   Database: {self.database_name}")
            print(f"   Password: {'*' * len(str(self.neo4j_config.get('password', 'password')))}")
            return False
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()