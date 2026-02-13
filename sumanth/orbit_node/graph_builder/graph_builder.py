"""
Graph Builder
Main module for building Neo4j knowledge graph from extracted data.
Orchestrates constraint creation and node insertion.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from .neo4j_connection import Neo4jConnection
from .constraints_manager import ConstraintsManager
from .node_inserter import NodeInserter
from .relationship_manager import RelationshipManager
import logging

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Main class for building Neo4j knowledge graph"""
    
    def __init__(
        self,
        neo4j_config_path: str = "config/neo4j_config.yaml",
        graph_config_path: str = "config/graph_config.yaml"
    ):
        """Initialize graph builder with configuration files"""
        self.neo4j_conn = Neo4jConnection(neo4j_config_path)
        self.constraints_manager = ConstraintsManager(self.neo4j_conn, graph_config_path)
        self.node_inserter = NodeInserter(self.neo4j_conn)
        self.relationship_manager = RelationshipManager(
            self.neo4j_conn, 
            graph_config_path,
            paths_config_path="config/paths_config.yaml"
        )
        self.graph_config = self.constraints_manager.graph_config
    
    def setup_constraints(self, node_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Setup constraints and indexes for node types
        
        Args:
            node_type: Specific node type to setup, or None for all types
        
        Returns:
            Dictionary with constraint creation results
        """
        logger.info("=" * 60)
        logger.info("Setting up Neo4j constraints and indexes")
        logger.info("=" * 60)
        
        if node_type:
            results = self.constraints_manager.create_constraints_for_node(node_type)
        else:
            results = self.constraints_manager.create_all_constraints()
        
        # Log summary
        for node_type_key, constraints in results.items():
            if isinstance(constraints, dict):
                success_count = sum(1 for v in constraints.values() if v)
                total_count = len(constraints)
                logger.info(f"{node_type_key}: {success_count}/{total_count} constraints created successfully")
        
        return results
    
    def insert_nodes(self, node_type: str, json_file: str, merge_key: Optional[str] = None, composite_keys: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Insert nodes from extracted JSON file
        
        Args:
            node_type: Node type (e.g., "asset", "host")
            json_file: Path to JSON file containing nodes
            merge_key: Property to use for MERGE operation (single key)
            composite_keys: List of properties for composite key MERGE
        
        Returns:
            Dictionary with insertion results
        """
        logger.info("=" * 60)
        logger.info(f"Inserting {node_type.upper()} nodes into Neo4j")
        logger.info("=" * 60)
        
        # Load nodes from JSON
        nodes = self.node_inserter.load_nodes_from_json(json_file)
        
        if not nodes:
            logger.warning("No nodes found in JSON file")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        # Get node label from config
        node_config = self.graph_config.get('nodes', {}).get(node_type, {})
        label = node_config.get('label', node_type.capitalize())
        
        # Count existing nodes
        existing_count = self.node_inserter.count_nodes(label)
        logger.info(f"Existing {label} nodes in database: {existing_count}")
        
        # Insert nodes
        results = self.node_inserter.insert_nodes_batch(
            label=label,
            nodes=nodes,
            merge_key=merge_key,
            composite_keys=composite_keys,
            batch_size=100
        )
        
        # Count nodes after insertion
        final_count = self.node_inserter.count_nodes(label)
        logger.info(f"Total {label} nodes in database after insertion: {final_count}")
        
        return results
    
    def insert_asset_nodes(self, json_file: str, merge_key: str = "asset_id") -> Dict[str, int]:
        """Insert asset nodes (convenience method)"""
        return self.insert_nodes("asset", json_file, merge_key=merge_key)
    
    def insert_host_nodes(self, json_file: str, composite_keys: List[str] = ["agent_id", "os_name"]) -> Dict[str, int]:
        """Insert host nodes with composite key (convenience method)"""
        return self.insert_nodes("host", json_file, composite_keys=composite_keys)
    
    def create_relationships(self, relationship_name: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """
        Create relationships based on configuration
        
        Args:
            relationship_name: Specific relationship to create, or None for all
        
        Returns:
            Dictionary with relationship creation results
        """
        logger.info("=" * 60)
        logger.info("Creating Relationships")
        logger.info("=" * 60)
        
        results = self.relationship_manager.create_relationships_from_config(relationship_name)
        
        # Log summary
        for rel_name, rel_results in results.items():
            success = rel_results.get('success', 0)
            total = rel_results.get('total', 0)
            logger.info(f"{rel_name}: {success} relationships created")
        
        return results
    
    def build_graph(self, asset_json_file: str, setup_constraints: bool = True) -> Dict[str, Any]:
        """
        Complete graph building process
        
        Args:
            asset_json_file: Path to extracted asset nodes JSON file
            setup_constraints: Whether to setup constraints before insertion
        
        Returns:
            Dictionary with build results
        """
        logger.info("=" * 60)
        logger.info("Starting Graph Building Process")
        logger.info("=" * 60)
        
        results = {
            'constraints': {},
            'nodes': {}
        }
        
        try:
            # Setup constraints
            if setup_constraints:
                logger.info("\nStep 1: Setting up constraints...")
                results['constraints'] = self.setup_constraints()
            
            # Insert nodes
            logger.info("\nStep 2: Inserting nodes...")
            results['nodes'] = self.insert_asset_nodes(asset_json_file)
            
            logger.info("\n" + "=" * 60)
            logger.info("Graph Building Completed Successfully!")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during graph building: {e}", exc_info=True)
            raise
    
    def close(self):
        """Close Neo4j connection"""
        self.neo4j_conn.close()

