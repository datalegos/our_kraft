"""
Constraints Manager
Creates and manages Neo4j constraints and indexes based on graph configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from .neo4j_connection import Neo4jConnection
import logging

logger = logging.getLogger(__name__)


class ConstraintsManager:
    """Manages Neo4j constraints and indexes"""
    
    def __init__(self, neo4j_conn: Neo4jConnection, graph_config_path: str = "config/graph_config.yaml"):
        """Initialize constraints manager"""
        self.neo4j_conn = neo4j_conn
        self.graph_config = self._load_graph_config(graph_config_path)
    
    def _load_graph_config(self, config_path: str) -> Dict[str, Any]:
        """Load graph configuration from YAML file"""
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
            logger.info(f"Loaded graph configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Graph config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing graph config YAML: {e}")
            raise
    
    def _create_unique_constraint(self, label: str, properties: List[str], constraint_name: str, composite: bool = False) -> bool:
        """Create a UNIQUE constraint on node label and properties"""
        try:
            if composite and len(properties) > 1:
                # For composite constraints, use tuple syntax: (n.prop1, n.prop2)
                props_tuple = ", ".join([f"n.{prop}" for prop in properties])
                props_str = f"({props_tuple})"
            else:
                # For single property constraints
                props_str = ", ".join([f"n.{prop}" for prop in properties])
            
            # Create constraint query - correct Neo4j syntax
            query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{label})
            REQUIRE {props_str} IS UNIQUE
            """
            
            self.neo4j_conn.execute_write(query)
            if composite:
                logger.info(f"Created composite UNIQUE constraint '{constraint_name}' on {label}({', '.join(properties)})")
            else:
                logger.info(f"Created UNIQUE constraint '{constraint_name}' on {label}({', '.join(properties)})")
            return True
            
        except Exception as e:
            # Check if constraint already exists
            if "already exists" in str(e).lower() or "equivalent constraint" in str(e).lower():
                logger.info(f"Constraint '{constraint_name}' already exists")
                return True
            logger.error(f"Failed to create UNIQUE constraint '{constraint_name}': {e}")
            return False
    
    def _create_index(self, label: str, properties: List[str], index_name: str) -> bool:
        """Create an INDEX on node label and properties"""
        try:
            # Format properties for Cypher
            props_str = ", ".join([f"n.{prop}" for prop in properties])
            
            # Create index query
            query = f"""
            CREATE INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON ({props_str})
            """
            
            self.neo4j_conn.execute_write(query)
            logger.info(f"Created INDEX '{index_name}' on {label}({', '.join(properties)})")
            return True
            
        except Exception as e:
            # Check if index already exists
            if "already exists" in str(e).lower() or "equivalent index" in str(e).lower():
                logger.info(f"Index '{index_name}' already exists")
                return True
            logger.error(f"Failed to create INDEX '{index_name}': {e}")
            return False
    
    def create_constraints_for_node(self, node_type: str) -> Dict[str, bool]:
        """Create all constraints and indexes for a node type"""
        nodes_config = self.graph_config.get('nodes', {})
        node_config = nodes_config.get(node_type)
        
        if not node_config:
            logger.warning(f"Node type '{node_type}' not found in graph configuration")
            return {}
        
        label = node_config.get('label', node_type)
        constraints_config = node_config.get('constraints', {})
        
        results = {}
        
        for constraint_name, constraint_config in constraints_config.items():
            if not constraint_config.get('enabled', True):
                logger.info(f"Constraint '{constraint_name}' is disabled, skipping")
                continue
            
            constraint_type = constraint_config.get('type', '').upper()
            properties = constraint_config.get('properties', [])
            composite = constraint_config.get('composite', False)
            
            if not properties:
                logger.warning(f"Constraint '{constraint_name}' has no properties, skipping")
                continue
            
            if constraint_type == 'UNIQUE':
                results[constraint_name] = self._create_unique_constraint(
                    label, properties, constraint_name, composite=composite
                )
            elif constraint_type == 'INDEX':
                results[constraint_name] = self._create_index(
                    label, properties, constraint_name
                )
            else:
                logger.warning(f"Unknown constraint type '{constraint_type}' for '{constraint_name}'")
                results[constraint_name] = False
        
        return results
    
    def create_all_constraints(self) -> Dict[str, Dict[str, bool]]:
        """Create constraints for all node types in configuration"""
        nodes_config = self.graph_config.get('nodes', {})
        all_results = {}
        
        for node_type in nodes_config.keys():
            logger.info(f"Creating constraints for node type: {node_type}")
            results = self.create_constraints_for_node(node_type)
            all_results[node_type] = results
        
        return all_results
    
    def list_existing_constraints(self) -> List[Dict[str, Any]]:
        """List all existing constraints and indexes in Neo4j"""
        try:
            # Query for constraints
            constraints_query = """
            SHOW CONSTRAINTS
            """
            
            # Query for indexes
            indexes_query = """
            SHOW INDEXES
            """
            
            constraints = self.neo4j_conn.execute_query(constraints_query)
            indexes = self.neo4j_conn.execute_query(indexes_query)
            
            return {
                'constraints': constraints,
                'indexes': indexes
            }
        except Exception as e:
            logger.error(f"Failed to list constraints: {e}")
            return {'constraints': [], 'indexes': []}

