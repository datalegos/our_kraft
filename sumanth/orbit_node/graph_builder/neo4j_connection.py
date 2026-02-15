"""
Neo4j Connection Handler
Manages connection to Neo4j database using configuration file.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Handles Neo4j database connection and session management"""
    
    def __init__(self, config_path: str = "config/neo4j_config.yaml"):
        """Initialize Neo4j connection from config file"""
        self.config = self._load_config(config_path)
        self.driver = None
        self._connect()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Neo4j configuration from YAML file"""
        try:
            # Handle both absolute and relative paths
            config_file = Path(config_path)
            if not config_file.is_absolute():
                if not config_file.exists():
                    project_root = Path(__file__).parent.parent
                    config_file = project_root / config_path
                    if not config_file.exists():
                        config_file = project_root / 'config' / Path(config_path).name
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded Neo4j configuration from {config_file}")
            return config.get('neo4j', {})
        except FileNotFoundError:
            logger.error(f"Neo4j config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing Neo4j config YAML: {e}")
            raise
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            uri = self.config.get('uri', 'bolt://localhost:7687')
            username = self.config.get('username', 'neo4j')
            password = self.config.get('password', '')
            database = self.config.get('database', 'neo4j')
            encrypted = self.config.get('encrypted', False)
            
            # Validate required fields
            if not password:
                raise ValueError("Neo4j password is required but not provided in config")
            
            # Debug: Log password details (without exposing actual password)
            password_length = len(password)
            has_whitespace = password != password.strip()
            logger.info(f"Connecting to Neo4j at {uri} with username: {username}")
            logger.debug(f"Database: {database}, Encrypted: {encrypted}")
            logger.debug(f"Password length: {password_length}, Has whitespace: {has_whitespace}")
            
            # Strip password to remove any accidental whitespace
            password = password.strip()
            
            # Note: If password contains special characters and authentication fails,
            # it might be a Neo4j configuration issue where Bolt protocol requires
            # password to be set separately from HTTP/Browser authentication
            
            # Build connection parameters
            connection_params = {
                'uri': uri,
                'auth': (username, password),
                'max_connection_lifetime': self.config.get('max_connection_lifetime', 3600),
                'max_connection_pool_size': self.config.get('max_connection_pool_size', 50),
                'connection_acquisition_timeout': self.config.get('connection_acquisition_timeout', 60),
            }
            
            # Check if URI already indicates encryption (neo4j+s://, bolt+s://, etc.)
            # URIs with +s or +ssc already have encryption built-in
            uri_has_encryption = '+s' in uri or '+ssc' in uri
            
            # Handle encrypted connection and trust settings
            if encrypted and not uri_has_encryption:
                # For bolt:// and neo4j:// schemes, set encrypted/trust explicitly
                # Import trust constants (handle version differences)
                from neo4j import TRUST_ALL_CERTIFICATES, TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
                try:
                    from neo4j import TRUST_CUSTOM_CA_SIGNED_CERTIFICATES
                except ImportError:
                    # TRUST_CUSTOM_CA_SIGNED_CERTIFICATES not available in this driver version
                    TRUST_CUSTOM_CA_SIGNED_CERTIFICATES = None
                
                trust = self.config.get('trust', 'TRUST_ALL_CERTIFICATES')
                connection_params['encrypted'] = True
                
                if trust == 'TRUST_ALL_CERTIFICATES':
                    connection_params['trust'] = TRUST_ALL_CERTIFICATES
                elif trust == 'TRUST_SYSTEM_CA_SIGNED_CERTIFICATES':
                    connection_params['trust'] = TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
                elif trust == 'TRUST_CUSTOM_CA_SIGNED_CERTIFICATES':
                    if TRUST_CUSTOM_CA_SIGNED_CERTIFICATES is not None:
                        connection_params['trust'] = TRUST_CUSTOM_CA_SIGNED_CERTIFICATES
                    else:
                        logger.warning("TRUST_CUSTOM_CA_SIGNED_CERTIFICATES not available in this driver version, using TRUST_ALL_CERTIFICATES")
                        connection_params['trust'] = TRUST_ALL_CERTIFICATES
                else:
                    connection_params['trust'] = TRUST_ALL_CERTIFICATES
            elif uri_has_encryption:
                # URI already has encryption built-in (neo4j+s://, bolt+s://, etc.)
                # For these schemes, we cannot set encrypted/trust/ssl_context parameters
                # The driver handles SSL automatically, but certificate verification depends on:
                # 1. System CA certificate store
                # 2. Network/firewall settings
                # 3. Aura instance availability
                logger.info("URI already indicates encryption (bolt+s:// or neo4j+s://)")
                logger.info("SSL certificate verification handled automatically by driver")
                logger.info("If connection fails, check: network, firewall, or Aura instance status")
            
            self.driver = GraphDatabase.driver(**connection_params)
            self.database = database
            
            # Test connection
            self.test_connection()
            logger.info(f"Successfully connected to Neo4j at {uri}")
            
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            if "authentication failure" in error_msg.lower() or "unauthorized" in error_msg.lower():
                logger.error(f"Authentication failed. Please check:")
                logger.error(f"  - Username: {self.config.get('username', 'NOT SET')}")
                logger.error(f"  - Password: {'SET' if self.config.get('password') else 'NOT SET'}")
                logger.error(f"  - URI: {self.config.get('uri', 'NOT SET')}")
                logger.error(f"  - Database: {self.config.get('database', 'NOT SET')}")
                logger.error(f"Make sure Neo4j is running and credentials are correct.")
            else:
                logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    logger.info("Neo4j connection test successful")
                    return True
                else:
                    logger.warning("Neo4j connection test returned no result")
                    return False
        except Exception as e:
            error_msg = str(e)
            if "authentication failure" in error_msg.lower() or "unauthorized" in error_msg.lower():
                logger.error("Neo4j authentication failed. Please verify:")
                logger.error("  1. Neo4j server is running")
                logger.error("  2. Username and password in neo4j_config.yaml are correct")
                logger.error("  3. The password matches your Neo4j instance")
                logger.error("  4. If this is a fresh install, you may need to change the default password")
            else:
                logger.error(f"Neo4j connection test failed: {e}")
            raise
    
    def get_session(self):
        """Get a Neo4j session"""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self.driver.session(database=self.database)
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a Cypher query and return results"""
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return result.data()
    
    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a write transaction"""
        with self.get_session() as session:
            result = session.write_transaction(
                lambda tx: tx.run(query, parameters or {}).data()
            )
            return result
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

