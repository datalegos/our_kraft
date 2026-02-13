"""
Node Inserter
Inserts nodes into Neo4j graph database based on extracted data and configuration.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .neo4j_connection import Neo4jConnection
import logging

logger = logging.getLogger(__name__)


class NodeInserter:
    """Handles insertion of nodes into Neo4j"""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        """Initialize node inserter"""
        self.neo4j_conn = neo4j_conn
    
    def _convert_datetime(self, value: Any) -> Optional[str]:
        """Convert datetime value to ISO format string for Neo4j"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            # Try to parse and return as ISO string
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.isoformat()
            except ValueError:
                return value
        return value
    
    def _prepare_node_properties(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare node properties for Neo4j insertion"""
        properties = {}
        
        for key, value in node_data.items():
            # Convert datetime objects to ISO strings
            if isinstance(value, datetime) or (isinstance(value, str) and 'T' in value):
                properties[key] = self._convert_datetime(value)
            else:
                properties[key] = value
        
        return properties
    
    def insert_node(self, label: str, properties: Dict[str, Any], merge_key: Optional[str] = None, composite_keys: Optional[List[str]] = None) -> bool:
        """
        Insert or merge a node in Neo4j
        
        Args:
            label: Node label (e.g., "Asset")
            properties: Node properties dictionary
            merge_key: Property name to use for MERGE (e.g., "asset_id")
            composite_keys: List of property names for composite key MERGE (e.g., ["agent_id", "os_name"])
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare properties
            props = self._prepare_node_properties(properties)
            
            # Build Cypher query
            if composite_keys and len(composite_keys) > 1:
                # Use composite key for MERGE
                merge_clauses = ", ".join([f"{k}: ${k}" for k in composite_keys if k in props])
                set_clauses = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
                query = f"""
                MERGE (n:{label} {{{merge_clauses}}})
                SET {set_clauses}
                RETURN n
                """
                parameters = props
            elif merge_key and merge_key in props:
                # Use single key MERGE to avoid duplicates
                merge_value = props[merge_key]
                # Build SET clause for all properties
                set_clauses = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
                query = f"""
                MERGE (n:{label} {{{merge_key}: $merge_value}})
                SET {set_clauses}
                RETURN n
                """
                # Include merge_value in parameters
                parameters = {**props, 'merge_value': merge_value}
            else:
                # Use CREATE (will fail if duplicate exists with unique constraint)
                props_str = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
                query = f"""
                CREATE (n:{label})
                SET {props_str}
                RETURN n
                """
                parameters = props
            
            result = self.neo4j_conn.execute_write(query, parameters)
            
            if result:
                logger.debug(f"Successfully inserted/merged {label} node with {merge_key or 'properties'}")
                return True
            else:
                logger.warning(f"No result returned when inserting {label} node")
                return False
                
        except Exception as e:
            logger.error(f"Failed to insert {label} node: {e}")
            return False
    
    def insert_nodes_batch(self, label: str, nodes: List[Dict[str, Any]], merge_key: Optional[str] = None, composite_keys: Optional[List[str]] = None, batch_size: int = 100) -> Dict[str, int]:
        """
        Insert multiple nodes in batches
        
        Args:
            label: Node label
            nodes: List of node property dictionaries
            merge_key: Property name to use for MERGE
            composite_keys: List of property names for composite key MERGE
            batch_size: Number of nodes to insert per batch
        
        Returns:
            Dictionary with success and failure counts
        """
        results = {'success': 0, 'failed': 0, 'total': len(nodes)}
        
        logger.info(f"Inserting {len(nodes)} {label} nodes in batches of {batch_size}")
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(nodes) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} nodes)")
            
            for node in batch:
                if self.insert_node(label, node, merge_key, composite_keys):
                    results['success'] += 1
                else:
                    results['failed'] += 1
        
        logger.info(f"Batch insertion complete: {results['success']} succeeded, {results['failed']} failed")
        return results
    
    def load_nodes_from_json(self, json_file: str) -> List[Dict[str, Any]]:
        """Load nodes from JSON file"""
        json_path = Path(json_file)
        
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_file}")
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        logger.info(f"Loading nodes from: {json_file}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        
        if not isinstance(nodes, list):
            logger.warning("JSON file does not contain a list, wrapping in list")
            nodes = [nodes]
        
        logger.info(f"Loaded {len(nodes)} nodes from JSON file")
        return nodes
    
    def count_nodes(self, label: str) -> int:
        """Count nodes with given label"""
        try:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = self.neo4j_conn.execute_query(query)
            if result:
                return result[0].get('count', 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to count {label} nodes: {e}")
            return 0

