"""
Relationship Manager
Creates relationships between nodes based on graph configuration.
Supports both property-based matching and source file-based matching.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .neo4j_connection import Neo4jConnection
import logging

logger = logging.getLogger(__name__)


class RelationshipManager:
    """Manages relationship creation between nodes"""
    
    def __init__(self, neo4j_conn: Neo4jConnection, graph_config_path: str = "config/graph_config.yaml", paths_config_path: str = "config/paths_config.yaml"):
        """Initialize relationship manager"""
        self.neo4j_conn = neo4j_conn
        self.graph_config = self._load_graph_config(graph_config_path)
        self.paths_config = self._load_paths_config(paths_config_path)
        self.base_dir = self._get_base_directory()
    
    def _load_graph_config(self, config_path: str) -> Dict[str, Any]:
        """Load graph configuration from YAML file"""
        try:
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
    
    def _load_paths_config(self, config_path: str) -> Dict[str, Any]:
        """Load paths configuration from YAML file"""
        try:
            config_file = Path(config_path)
            if not config_file.is_absolute():
                if not config_file.exists():
                    project_root = Path(__file__).parent.parent
                    config_file = project_root / config_path
                    if not config_file.exists():
                        config_file = project_root / 'config' / Path(config_path).name
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded paths configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Paths config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing paths config YAML: {e}")
            raise
    
    def _get_base_directory(self) -> Path:
        """Get the base directory for collected data"""
        paths = self.paths_config.get('paths', {})
        use_latest = paths.get('use_latest', False)
        base_dir_str = paths.get('base_directory')
        
        if use_latest:
            collected_data_parent = Path("collected_data")
            if collected_data_parent.exists():
                import re
                latest_dir = max(
                    (d for d in collected_data_parent.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)),
                    key=lambda d: d.name,
                    default=None
                )
                if latest_dir:
                    logger.info(f"Using latest collected data directory: {latest_dir.name}")
                    return latest_dir
            logger.warning("Could not find latest collected data directory. Falling back to configured base_directory.")
        
        if base_dir_str:
            return Path(base_dir_str)
        
        raise ValueError("No base directory configured in paths_config.yaml")
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation path"""
        if not path:
            return data
        
        keys = path.split('.')
        value = data
        
        for i, key in enumerate(keys):
            if key.endswith('[]'):
                # Handle array access - return the list
                array_key = key[:-2]
                if isinstance(value, dict) and array_key in value:
                    value = value[array_key]
                    # If this is the last key and it's an array, return it
                    if i == len(keys) - 1:
                        return value if isinstance(value, list) else [value] if value else []
                elif isinstance(value, dict):
                    return []
            elif isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list):
                # If we have a list, process each item
                if key.isdigit():
                    idx = int(key)
                    value = value[idx] if idx < len(value) else None
                else:
                    # Apply key to each item in list - but we want to keep the list structure
                    result = []
                    for item in value:
                        if isinstance(item, dict) and key in item:
                            result.append(item[key])
                    return result if result else None
            else:
                return None
            
            if value is None:
                return None
        
        return value
    
    def _load_source_data(
        self,
        source_file: str,
        source_directory: str,
        source_path: Optional[str] = None,
        agent_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load source data from file (integrated from source_data_loader)
        
        Args:
            source_file: Name of the source file
            source_directory: Directory key from paths_config
            source_path: Optional path to extract specific data (e.g., "data.affected_items[]")
            agent_ids: Optional list of agent IDs for per-agent files
        
        Returns:
            List of data items
        """
        data_sources = self.paths_config.get('paths', {}).get('data_sources', {})
        directory_config = data_sources.get(source_directory)
        
        if not directory_config:
            logger.error(f"Directory '{source_directory}' not found in paths_config.yaml")
            return []
        
        directory = directory_config.get('directory')
        data_dir = self.base_dir / directory
        
        all_data = []
        
        # Check if file is per-agent or single file
        if '{agent_id}' in source_file and agent_ids:
            # Per-agent files
            for agent_id in agent_ids:
                agent_file = source_file.replace('{agent_id}', agent_id)
                file_path = data_dir / f"agent_{agent_id}" / agent_file
                
                if not file_path.exists():
                    # Try without agent subdirectory
                    file_path = data_dir / agent_file
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if source_path:
                            # Extract data using source_path
                            extracted = self._get_nested_value(data, source_path)
                            if isinstance(extracted, list):
                                # Add agent_id to each item for matching
                                for item in extracted:
                                    if isinstance(item, dict):
                                        item['_agent_id'] = agent_id
                                all_data.extend(extracted)
                            elif extracted:
                                if isinstance(extracted, dict):
                                    extracted['_agent_id'] = agent_id
                                all_data.append(extracted)
                        else:
                            # Return full data structure
                            if isinstance(data, dict):
                                data['_agent_id'] = agent_id
                            all_data.append(data)
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        else:
            # Single file
            file_path = data_dir / source_file
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if source_path:
                        # Extract data using source_path
                        extracted = self._get_nested_value(data, source_path)
                        if isinstance(extracted, list):
                            all_data.extend(extracted)
                        elif extracted:
                            all_data.append(extracted)
                    else:
                        # Handle different data structures
                        if 'data' in data and 'affected_items' in data['data']:
                            items = data['data']['affected_items']
                            all_data.extend(items)
                        elif isinstance(data, dict):
                            # Handle per-agent structure (e.g., {"000": {...}, "001": {...}})
                            for agent_id, agent_data in data.items():
                                if 'data' in agent_data and 'affected_items' in agent_data['data']:
                                    items = agent_data['data']['affected_items']
                                    for item in items:
                                        if isinstance(item, dict):
                                            item['_agent_id'] = agent_id
                                    all_data.extend(items)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return all_data
    
    def create_relationship(
        self,
        from_label: str,
        to_label: str,
        relationship_type: str,
        from_property: str,
        to_property: str,
        from_value: Any = None,
        to_value: Any = None
    ) -> bool:
        """
        Create a relationship between two nodes
        
        Args:
            from_label: Label of source node
            to_label: Label of target node
            relationship_type: Type of relationship
            from_property: Property name to match on source node
            to_property: Property name to match on target node
            from_value: Optional specific value to match (if None, matches all)
            to_value: Optional specific value to match (if None, matches all)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if from_value and to_value:
                # Match specific nodes
                query = f"""
                MATCH (a:{from_label} {{{from_property}: $from_value}})
                MATCH (b:{to_label} {{{to_property}: $to_value}})
                MERGE (a)-[r:{relationship_type}]->(b)
                RETURN r
                """
                parameters = {'from_value': from_value, 'to_value': to_value}
            else:
                # Match all nodes based on property equality
                query = f"""
                MATCH (a:{from_label})
                MATCH (b:{to_label})
                WHERE a.{from_property} = b.{to_property}
                MERGE (a)-[r:{relationship_type}]->(b)
                RETURN r
                """
                parameters = {}
            
            result = self.neo4j_conn.execute_write(query, parameters)
            
            if result:
                logger.debug(f"Created relationship {relationship_type} from {from_label} to {to_label}")
                return True
            else:
                logger.warning(f"No relationship created between {from_label} and {to_label}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create relationship {relationship_type}: {e}")
            return False
    
    def create_relationships_batch(
        self,
        from_label: str,
        to_label: str,
        relationship_type: str,
        from_property: str,
        to_property: str,
        additional_matches: Optional[List[Dict[str, str]]] = None,
        array_match: bool = False
    ) -> Dict[str, int]:
        """
        Create relationships between all matching nodes
        
        Args:
            from_label: Label of source nodes
            to_label: Label of target nodes
            relationship_type: Type of relationship
            from_property: Property name to match on source nodes
            to_property: Property name to match on target nodes
            additional_matches: Optional list of additional match criteria
                Each dict should have 'from_property' and 'to_property' keys
            array_match: If True, from_property is an array and we check if to_property is in it
        
        Returns:
            Dictionary with success and failure counts
        """
        try:
            # Build WHERE clause with primary match
            if array_match:
                # Handle array matching: check if to_property value is in from_property array
                where_clauses = [f"b.{to_property} IN a.{from_property}"]
            else:
                where_clauses = [f"a.{from_property} = b.{to_property}"]
            
            # Add additional match criteria if provided
            if additional_matches:
                for match in additional_matches:
                    if 'from_property' in match and 'to_property' in match:
                        match_array = match.get('array_match', False)
                        if match_array:
                            where_clauses.append(f"b.{match['to_property']} IN a.{match['from_property']}")
                        else:
                            where_clauses.append(f"a.{match['from_property']} = b.{match['to_property']}")
            
            where_clause = " AND ".join(where_clauses)
            
            query = f"""
            MATCH (a:{from_label})
            MATCH (b:{to_label})
            WHERE {where_clause}
            MERGE (a)-[r:{relationship_type}]->(b)
            RETURN count(r) as count
            """
            
            result = self.neo4j_conn.execute_write(query)
            
            if result and len(result) > 0:
                count = result[0].get('count', 0)
                logger.info(f"Created {count} {relationship_type} relationships from {from_label} to {to_label}")
                return {'success': count, 'failed': 0, 'total': count}
            else:
                logger.warning(f"No relationships created")
                return {'success': 0, 'failed': 0, 'total': 0}
                
        except Exception as e:
            logger.error(f"Failed to create relationships {relationship_type}: {e}")
            return {'success': 0, 'failed': 1, 'total': 0}
    
    def create_relationships_from_source(
        self,
        from_label: str,
        to_label: str,
        relationship_type: str,
        source_config: Dict[str, Any],
        to_property: str,
        from_node_key_property: str = "asset_id"
    ) -> Dict[str, int]:
        """
        Create relationships by loading data from source files
        
        Args:
            from_label: Label of source nodes
            to_label: Label of target nodes
            relationship_type: Type of relationship
            source_config: Configuration for source data (source_field, source_path, source_file, source_directory)
            to_property: Property name on target nodes to match against
            from_node_key_property: Property on from_node to match with agent_id from source
        
        Returns:
            Dictionary with success and failure counts
        """
        try:
            source_field = source_config.get('source_field')
            source_path = source_config.get('source_path')
            source_file = source_config.get('source_file')
            source_directory = source_config.get('source_directory')
            array_match = source_config.get('array_match', False)
            
            if not all([source_field, source_file, source_directory]):
                logger.error("Source configuration missing required fields")
                return {'success': 0, 'failed': 1, 'total': 0}
            
            # Get node keys from existing from_nodes (these are the values we'll match against)
            # For example, if from_node is Asset and from_node_key_property is asset_id,
            # we get all asset_ids from Asset nodes
            node_keys_query = f"MATCH (n:{from_label}) RETURN DISTINCT n.{from_node_key_property} as node_key"
            node_keys_result = self.neo4j_conn.execute_query(node_keys_query)
            node_keys = [row['node_key'] for row in node_keys_result if row.get('node_key')]
            
            if not node_keys:
                logger.warning(f"No {from_label} nodes found to create relationships")
                return {'success': 0, 'failed': 0, 'total': 0}
            
            logger.debug(f"Found {len(node_keys)} {from_node_key_property} values from {from_label} nodes: {node_keys[:5]}...")
            
            # Determine if we need agent_ids for per-agent files
            # If source_file contains {agent_id}, we need to load per-agent files
            # Otherwise, load from single file
            needs_agent_ids = '{agent_id}' in source_file
            load_all_agents = source_config.get('load_all_agents', False)
            
            # For per-agent files, we need to get agent_ids
            if needs_agent_ids:
                if load_all_agents:
                    # Load all agent files - get agent_ids from Asset nodes
                    asset_query = "MATCH (n:NJS_Asset) RETURN DISTINCT n.asset_id as agent_id"
                    asset_result = self.neo4j_conn.execute_query(asset_query)
                    agent_ids_to_load = [row['agent_id'] for row in asset_result if row.get('agent_id')]
                    logger.info(f"Loading all agent files for {len(agent_ids_to_load)} agents")
                else:
                    # Extract agent_ids from node_keys (these are the IDs we'll use to load files)
                    # For most cases, node_keys are agent_ids, but we need to handle different cases
                    agent_ids_to_load = node_keys
                
                source_data_list = self._load_source_data(
                    source_file=source_file,
                    source_directory=source_directory,
                    source_path=source_path,
                    agent_ids=agent_ids_to_load
                )
            else:
                # Single file - load all data
                source_data_list = self._load_source_data(
                    source_file=source_file,
                    source_directory=source_directory,
                    source_path=source_path,
                    agent_ids=None
                )
            
            logger.debug(f"Loaded {len(source_data_list)} items from source data")
            if source_data_list and len(source_data_list) > 0:
                sample = source_data_list[0]
                logger.debug(f"Sample source item keys: {list(sample.keys())[:10] if isinstance(sample, dict) else type(sample)}")
            
            if not source_data_list:
                logger.warning(f"No source data loaded for relationship creation")
                return {'success': 0, 'failed': 0, 'total': 0}
            
            # Extract matching values from source data
            relationships_to_create = []
            node_keys_str = [str(key) for key in node_keys]
            
            # For load_all_agents, we match differently - we don't filter by source_key_field
            # Instead, we extract the matching field and match it with from_node keys
            if load_all_agents:
                # Load all data and match by the source_field value
                for source_item in source_data_list:
                    if not isinstance(source_item, dict):
                        continue
                    
                    # Extract the field value for relationship matching
                    if '.' in source_field:
                        field_value = self._get_nested_value(source_item, source_field)
                    else:
                        field_value = source_item.get(source_field)
                    
                    if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                        continue
                    
                    field_value_str = str(field_value).strip()
                    
                    # Differentiate between software_to_vulnerability and asset_to_vulnerability
                    # software_to_vulnerability: source_field is "package_name", from_node_key_property is "name"
                    # asset_to_vulnerability: source_field is "cve", from_node_key_property is "asset_id"
                    if source_field == "package_name" and from_node_key_property == "name":
                        # For software_to_vulnerability: Match Software.name with Vulnerability.package_name
                        # We match field_value (package_name) with from_node keys (Software.name)
                        if field_value_str in node_keys_str:
                            # Get cve_id from source for the to_node
                            cve_id = source_item.get('cve') or source_item.get('cve_id')
                            if cve_id:
                                relationships_to_create.append({
                                    'from_key': field_value_str,  # Software.name
                                    'to_value': str(cve_id).strip()  # Vulnerability.cve_id
                                })
                    elif source_field == "cve" and from_node_key_property == "asset_id":
                        # For asset_to_vulnerability: Match Asset.asset_id with Vulnerability.cve_id via agent_id
                        # We need to match from_node keys (Asset.asset_id) with source agent_id
                        agent_id = source_item.get('agent_id') or source_item.get('_agent_id')
                        if agent_id and str(agent_id) in node_keys_str:
                            # Get cve_id from source (source_field is "cve")
                            cve_id = source_item.get('cve') or source_item.get('cve_id')
                            if cve_id and str(cve_id).strip():
                                relationships_to_create.append({
                                    'from_key': str(agent_id),  # Asset.asset_id
                                    'to_value': str(cve_id).strip()  # Vulnerability.cve_id
                                })
                    elif to_property == "cve_id":
                        # Fallback for other cases with cve_id
                        if field_value_str in node_keys_str:
                            cve_id = source_item.get('cve') or source_item.get('cve_id')
                            if cve_id:
                                relationships_to_create.append({
                                    'from_key': field_value_str,
                                    'to_value': str(cve_id).strip()
                                })
            else:
                # Normal matching - filter by source_key_field first
                for source_item in source_data_list:
                    if not isinstance(source_item, dict):
                        continue
                        
                    # Get the matching key from source item (usually 'id' for agents)
                    # This should match the from_node_key_property value
                    source_key_field = source_config.get('source_key_field', 'id')  # Default to 'id'
                    source_key = source_item.get(source_key_field) or source_item.get('id') or source_item.get('agent_id')
                    
                    # Only process items that match our from_node keys
                    if not source_key or str(source_key) not in node_keys_str:
                        continue
                    
                    # Extract the field value for relationship matching
                    if '.' in source_field:
                        field_value = self._get_nested_value(source_item, source_field)
                    else:
                        field_value = source_item.get(source_field)
                    
                    if field_value is None:
                        continue
                    
                    if isinstance(field_value, list) and len(field_value) == 0:
                        continue
                    
                    if array_match and isinstance(field_value, list):
                        # For array matches, create relationship for each value in array
                        for value in field_value:
                            if value and value != '' and value != ' ':  # Skip empty values
                                relationships_to_create.append({
                                    'from_key': str(source_key),  # This matches from_node_key_property
                                    'to_value': str(value).strip() if value else None
                                })
                    elif not array_match and field_value:
                        # For single value matches
                        relationships_to_create.append({
                            'from_key': str(source_key),  # This matches from_node_key_property
                            'to_value': str(field_value).strip() if field_value else None
                        })
            
            logger.info(f"Prepared {len(relationships_to_create)} relationships to create from source data")
            
            if not relationships_to_create:
                logger.warning(f"No matching values found in source data")
                logger.debug(f"Source data had {len(source_data_list)} items, but no valid relationships extracted")
                return {'success': 0, 'failed': 0, 'total': 0}
            
            logger.info(f"Creating {len(relationships_to_create)} relationships using source data")
            logger.debug(f"Sample relationships: {relationships_to_create[:3]}")
            
            # Create relationships using Cypher
            # Use UNWIND to process all relationships efficiently
            query = f"""
            UNWIND $relationships AS rel
            MATCH (a:{from_label} {{{from_node_key_property}: rel.from_key}})
            MATCH (b:{to_label} {{{to_property}: rel.to_value}})
            MERGE (a)-[r:{relationship_type}]->(b)
            RETURN count(r) as count
            """
            
            result = self.neo4j_conn.execute_write(query, {'relationships': relationships_to_create})
            
            if result and len(result) > 0:
                count = result[0].get('count', 0)
                logger.info(f"Created {count} {relationship_type} relationships from {from_label} to {to_label} using source data")
                return {'success': count, 'failed': 0, 'total': count}
            else:
                logger.warning(f"No relationships created")
                return {'success': 0, 'failed': 0, 'total': 0}
                
        except Exception as e:
            logger.error(f"Failed to create relationships from source {relationship_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': 0, 'failed': 1, 'total': 0}
    
    def create_relationships_from_source_multi_field(
        self,
        from_label: str,
        to_label: str,
        relationship_type: str,
        source_config: Dict[str, Any],
        to_properties: List[str],
        from_node_key_properties: List[str]
    ) -> Dict[str, int]:
        """
        Create relationships by loading data from source files with multi-field matching
        (e.g., Software to Vulnerability matching on agent_id, name, version)
        
        Args:
            from_label: Label of source nodes
            to_label: Label of target nodes
            relationship_type: Type of relationship
            source_config: Configuration for source data
            to_properties: List of property names on target nodes to match against
            from_node_key_properties: List of property names on from_node to match with source
        
        Returns:
            Dictionary with success and failure counts
        """
        try:
            source_fields = source_config.get('source_fields', [])
            source_path = source_config.get('source_path')
            source_file = source_config.get('source_file')
            source_directory = source_config.get('source_directory')
            
            if not all([source_fields, source_file, source_directory]) or len(source_fields) != len(to_properties):
                logger.error("Source configuration missing required fields or field count mismatch")
                return {'success': 0, 'failed': 1, 'total': 0}
            
            # Get agent IDs from existing from_nodes
            # Use first key property to get agent IDs
            first_key_prop = from_node_key_properties[0]
            agent_ids_query = f"MATCH (n:{from_label}) RETURN DISTINCT n.{first_key_prop} as agent_id"
            agent_ids_result = self.neo4j_conn.execute_query(agent_ids_query)
            agent_ids = [row['agent_id'] for row in agent_ids_result if row.get('agent_id')]
            
            if not agent_ids:
                logger.warning(f"No {from_label} nodes found to create relationships")
                return {'success': 0, 'failed': 0, 'total': 0}
            
            # Load source data
            source_data_list = self._load_source_data(
                source_file=source_file,
                source_directory=source_directory,
                source_path=source_path,
                agent_ids=agent_ids
            )
            
            if not source_data_list:
                logger.warning(f"No source data loaded for relationship creation")
                return {'success': 0, 'failed': 0, 'total': 0}
            
            # Extract matching values from source data
            relationships_to_create = []
            for source_item in source_data_list:
                if not isinstance(source_item, dict):
                    continue
                
                # Extract all required fields
                field_values = {}
                for field in source_fields:
                    if '.' in field:
                        field_values[field] = self._get_nested_value(source_item, field)
                    else:
                        field_values[field] = source_item.get(field)
                
                # Check if all fields have values
                if any(v is None or (isinstance(v, str) and not v.strip()) for v in field_values.values()):
                    continue
                
                # Create relationship entry with all matching fields
                rel_entry = {}
                for i, key_prop in enumerate(from_node_key_properties):
                    rel_entry[f'from_{key_prop}'] = str(field_values[source_fields[i]])
                for i, to_prop in enumerate(to_properties):
                    rel_entry[f'to_{to_prop}'] = str(field_values[source_fields[i]])
                
                relationships_to_create.append(rel_entry)
            
            if not relationships_to_create:
                logger.warning(f"No matching values found in source data")
                return {'success': 0, 'failed': 0, 'total': 0}
            
            # Create relationships using Cypher with multi-field matching
            # Build WHERE clause for all matching fields
            where_clauses = []
            for i, key_prop in enumerate(from_node_key_properties):
                where_clauses.append(f"a.{key_prop} = rel.from_{key_prop}")
            for i, to_prop in enumerate(to_properties):
                where_clauses.append(f"b.{to_prop} = rel.to_{to_prop}")
            
            where_clause = " AND ".join(where_clauses)
            
            query = f"""
            UNWIND $relationships AS rel
            MATCH (a:{from_label})
            MATCH (b:{to_label})
            WHERE {where_clause}
            MERGE (a)-[r:{relationship_type}]->(b)
            RETURN count(r) as count
            """
            
            result = self.neo4j_conn.execute_write(query, {'relationships': relationships_to_create})
            
            if result and len(result) > 0:
                count = result[0].get('count', 0)
                logger.info(f"Created {count} {relationship_type} relationships from {from_label} to {to_label} using multi-field source data")
                return {'success': count, 'failed': 0, 'total': count}
            else:
                logger.warning(f"No relationships created")
                return {'success': 0, 'failed': 0, 'total': 0}
                
        except Exception as e:
            logger.error(f"Failed to create relationships from source (multi-field) {relationship_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': 0, 'failed': 1, 'total': 0}
    
    def create_relationships_from_config(self, relationship_name: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """
        Create all relationships defined in configuration
        Supports both property-based and source-based matching
        
        Args:
            relationship_name: Specific relationship to create, or None for all
        
        Returns:
            Dictionary with relationship creation results
        """
        relationships_config = self.graph_config.get('relationships', {})
        results = {}
        
        for rel_name, rel_config in relationships_config.items():
            if relationship_name and rel_name != relationship_name:
                continue
            
            if not rel_config.get('enabled', True):
                logger.info(f"Relationship '{rel_name}' is disabled, skipping")
                continue
            
            from_node = rel_config.get('from_node')
            to_node = rel_config.get('to_node')
            rel_type = rel_config.get('type')
            match_criteria = rel_config.get('match_criteria', {})
            
            # Get node labels from config
            from_node_config = self.graph_config.get('nodes', {}).get(from_node, {})
            to_node_config = self.graph_config.get('nodes', {}).get(to_node, {})
            
            from_label = from_node_config.get('label', from_node.capitalize())
            to_label = to_node_config.get('label', to_node.capitalize())
            
            logger.info(f"Creating relationship '{rel_name}': {from_label}-[:{rel_type}]->{to_label}")
            
            # Check if using source-based matching
            from_source = match_criteria.get('from_source')
            to_property = match_criteria.get('to_property')
            to_properties = match_criteria.get('to_properties')  # For multi-field matching
            
            if from_source:
                # Use source-based matching
                # Get from_node_key_property from source config or match_criteria
                from_node_key_property = from_source.get('from_node_key_property') or match_criteria.get('from_node_key_property', 'asset_id')
                from_node_key_properties = from_source.get('from_node_key_properties')  # For multi-field matching
                
                # Check if this is multi-field matching (software_to_vulnerability)
                if from_node_key_properties and to_properties:
                    result = self.create_relationships_from_source_multi_field(
                        from_label=from_label,
                        to_label=to_label,
                        relationship_type=rel_type,
                        source_config=from_source,
                        to_properties=to_properties,
                        from_node_key_properties=from_node_key_properties
                    )
                else:
                    result = self.create_relationships_from_source(
                        from_label=from_label,
                        to_label=to_label,
                        relationship_type=rel_type,
                        source_config=from_source,
                        to_property=to_property,
                        from_node_key_property=from_node_key_property
                    )
            else:
                # Use property-based matching (existing logic)
                from_property = match_criteria.get('from_property')
                
                if not all([from_node, to_node, rel_type, from_property, to_property]):
                    logger.warning(f"Relationship '{rel_name}' missing required configuration, skipping")
                    continue
                
                # Get additional match criteria if any
                additional_matches = []
                for key in match_criteria.keys():
                    if key.startswith('additional_match'):
                        additional_match = match_criteria.get(key)
                        if isinstance(additional_match, dict) and 'from_property' in additional_match and 'to_property' in additional_match:
                            additional_matches.append(additional_match)
                
                # Check if this is an array-based match
                array_match = match_criteria.get('array_match', False)
                
                result = self.create_relationships_batch(
                    from_label=from_label,
                    to_label=to_label,
                    relationship_type=rel_type,
                    from_property=from_property,
                    to_property=to_property,
                    additional_matches=additional_matches if additional_matches else None,
                    array_match=array_match
                )
            
            results[rel_name] = result
        
        return results

