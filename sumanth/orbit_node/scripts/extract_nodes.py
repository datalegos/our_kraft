"""
Generic Node Extractor
Extracts nodes from collected Wazuh data based on configuration files.
Supports multiple node types (Asset, Host, etc.) with configurable properties.
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/node_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration files"""
    
    def __init__(self):
        self.graph_config = None
        self.paths_config = None
        self.neo4j_config = None
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files"""
        try:
            # Get config directory relative to script location
            script_dir = Path(__file__).parent
            config_dir = script_dir.parent / 'config'
            
            with open(config_dir / 'graph_config.yaml', 'r') as f:
                self.graph_config = yaml.safe_load(f)
            logger.info("Loaded graph_config.yaml")
            
            with open(config_dir / 'paths_config.yaml', 'r') as f:
                self.paths_config = yaml.safe_load(f)
            logger.info("Loaded paths_config.yaml")
            
            with open(config_dir / 'neo4j_config.yaml', 'r') as f:
                self.neo4j_config = yaml.safe_load(f)
            logger.info("Loaded neo4j_config.yaml")
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise


class NodeExtractor:
    """Generic extractor for any node type based on configuration"""
    
    def __init__(self, config_loader: ConfigLoader, node_type: str):
        self.config_loader = config_loader
        self.graph_config = config_loader.graph_config
        self.paths_config = config_loader.paths_config
        self.node_type = node_type
        self.node_config = self._get_node_config()
        self.base_dir = self._get_base_directory()
        
    def _get_node_config(self) -> Dict[str, Any]:
        """Get configuration for the specified node type"""
        nodes_config = self.graph_config.get('nodes', {})
        node_config = nodes_config.get(self.node_type)
        
        if not node_config:
            raise ValueError(f"Node type '{self.node_type}' not found in graph_config.yaml")
        
        return node_config
    
    def _get_base_directory(self) -> Path:
        """Get the base directory for collected data"""
        base_path = self.paths_config['paths']['base_directory']
        
        if self.paths_config['paths'].get('use_latest', False):
            # Find latest timestamped folder
            collected_data_dir = Path('collected_data')
            if collected_data_dir.exists():
                folders = [f for f in collected_data_dir.iterdir() if f.is_dir()]
                if folders:
                    latest = max(folders, key=lambda x: x.stat().st_mtime)
                    base_path = str(latest)
                    logger.info(f"Using latest data folder: {base_path}")
        
        return Path(base_path)
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation path"""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)] if int(key) < len(value) else None
            else:
                return None
            if value is None:
                return None
        return value
    
    def _extract_property_value(self, source_data: Dict, prop_config: Dict) -> Any:
        """Extract property value from source data based on configuration"""
        # If source_field is null, use default value
        if prop_config.get('source_field') is None:
            default = prop_config.get('default_value')
            # Handle special default values
            if default is None and prop_config.get('data_type') == 'datetime':
                # For required datetime fields, use current time if no default
                if prop_config.get('required', False):
                    return datetime.now()
                return None
            return default
        
        # Extract from source_field (supports nested paths like "os.name")
        source_field = prop_config['source_field']
        
        # Handle nested field paths (e.g., "os.name")
        if '.' in source_field:
            value = self._get_nested_value(source_data, source_field)
        else:
            value = source_data.get(source_field)
        
        # Handle data type conversions
        data_type = prop_config.get('data_type', 'string')
        
        # Check if value is None or empty, and try fallback field
        if value is None or (isinstance(value, str) and not value.strip()):
            # Try fallback field if configured
            fallback_field = prop_config.get('fallback_field')
            if fallback_field:
                fallback_path = prop_config.get('fallback_path', '')
                if '.' in fallback_field or fallback_path:
                    # Use fallback_path if provided, otherwise use fallback_field
                    if fallback_path:
                        value = self._get_nested_value(source_data, fallback_path.replace('data.affected_items[].', ''))
                    else:
                        value = self._get_nested_value(source_data, fallback_field)
                else:
                    value = source_data.get(fallback_field)
        
        # If still None and required, use default or current time for datetime
        if value is None:
            if prop_config.get('required', False) and data_type == 'datetime':
                # For required datetime fields, use current time if no default
                return datetime.now()
            return prop_config.get('default_value')
        
        # Type conversions
        if data_type == 'datetime':
            # Try to parse ISO 8601 datetime
            if isinstance(value, str):
                try:
                    # Handle empty strings
                    if not value or value.strip() == '':
                        # If required, use current time, otherwise return None
                        if prop_config.get('required', False):
                            return datetime.now()
                        return None
                    # Handle special case: "9999-12-31T23:59:59+00:00" (never disconnected)
                    # For last_modified, use current time; for created_at, use fallback or current time
                    if '9999-12-31' in value:
                        # Try fallback field first
                        fallback_field = prop_config.get('fallback_field')
                        if fallback_field:
                            fallback_path = prop_config.get('fallback_path', '')
                            if fallback_path:
                                fallback_value = self._get_nested_value(source_data, fallback_path.replace('data.affected_items[].', ''))
                            else:
                                fallback_value = source_data.get(fallback_field)
                            if fallback_value and isinstance(fallback_value, str) and fallback_value.strip() and '9999-12-31' not in fallback_value:
                                try:
                                    return datetime.fromisoformat(fallback_value.replace('Z', '+00:00'))
                                except ValueError:
                                    pass
                        # If required, use current time instead of None
                        if prop_config.get('required', False):
                            return datetime.now()
                        return None
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse datetime: {value}")
                    # If required, use current time as fallback
                    if prop_config.get('required', False):
                        return datetime.now()
                    return None
            # If value is already a datetime object, return it
            if isinstance(value, datetime):
                return value
            # If required but value is None, use current time
            if value is None and prop_config.get('required', False):
                return datetime.now()
            return value
        elif data_type == 'float':
            # Handle float conversion
            if value is None or value == '':
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    logger.warning(f"Could not convert to float: {value}")
                    return None
            return value
        elif data_type == 'array':
            # Handle array type - return as-is if already a list, otherwise wrap in list
            if value is None:
                return prop_config.get('default_value', [])
            if isinstance(value, list):
                return value
            # If not a list, wrap in list (or return empty list)
            return [value] if value else []
        elif data_type == 'boolean':
            return bool(value) if value is not None else prop_config.get('default_value', False)
        elif data_type == 'enum':
            # Validate enum value
            enum_values = prop_config.get('enum_values', [])
            # Handle empty strings and None
            if not value or (isinstance(value, str) and value.strip() == ''):
                return prop_config.get('default_value', None)
            
            if enum_values and value not in enum_values:
                # Try case-insensitive match
                value_lower = str(value).lower() if value else ''
                matched = None
                for ev in enum_values:
                    if str(ev).lower() == value_lower:
                        matched = ev
                        break
                if matched:
                    return matched
                # If no match and value is empty or invalid, use default
                if not value or value == '-' or value == '':
                    return prop_config.get('default_value', None)
                logger.warning(f"Value '{value}' not in enum values {enum_values}, using as-is")
            return value
        
        return value
    
    def _extract_node(self, source_data: Dict) -> Dict[str, Any]:
        """Extract a single node from source data"""
        node = {}
        properties_config = self.node_config.get('properties', {})
        
        for prop_name, prop_config in properties_config.items():
            try:
                value = self._extract_property_value(source_data, prop_config)
                node[prop_name] = value
            except Exception as e:
                logger.error(f"Error extracting property '{prop_name}': {e}")
                # Use default value if extraction fails
                node[prop_name] = prop_config.get('default_value')
        
        return node
    
    def load_data_for_agents(self, agent_ids: List[str]) -> List[Dict]:
        """Load data files for specific agents"""
        file_sources = self.graph_config.get('file_sources', {})
        source_config = file_sources.get(self.node_type)
        
        if not source_config:
            raise ValueError(f"File source configuration not found for node type '{self.node_type}'")
        
        directory_key = source_config.get('directory_key')
        file_pattern = source_config.get('file_pattern', '')
        per_agent = source_config.get('per_agent', False)
        
        data_dir = self.base_dir / self.paths_config['paths']['data_sources'][directory_key]['directory']
        all_data = []
        
        if per_agent:
            # Load per-agent files
            for agent_id in agent_ids:
                agent_file = file_pattern.replace('{agent_id}', agent_id)
                file_path = data_dir / f"agent_{agent_id}" / agent_file
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Extract affected_items
                        items = data.get('data', {}).get('affected_items', [])
                        all_data.extend(items)
                        logger.debug(f"Loaded {len(items)} items from {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
                else:
                    logger.warning(f"File not found: {file_path}")
        else:
            # Load single summary file
            summary_file = self.paths_config['paths']['data_sources'][directory_key].get('file_pattern', '')
            if '{agent_id}' in summary_file:
                # Try to load summary file without agent_id
                summary_file = summary_file.replace('_{agent_id}', '').replace('{agent_id}', '')
            
            file_path = data_dir / summary_file
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different data structures
                    if 'data' in data and 'affected_items' in data['data']:
                        all_data = data['data']['affected_items']
                    elif isinstance(data, dict):
                        # Handle per-agent structure (e.g., {"000": {...}, "001": {...}})
                        for agent_id, agent_data in data.items():
                            if 'data' in agent_data and 'affected_items' in agent_data['data']:
                                all_data.extend(agent_data['data']['affected_items'])
                    
                    logger.debug(f"Loaded {len(all_data)} items from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return all_data
    
    def extract_all_nodes(self, agent_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract all nodes of this type from collected data"""
        # Load source data
        if agent_ids:
            source_data_list = self.load_data_for_agents(agent_ids)
        else:
            # Try to get agent IDs from existing asset nodes JSON file
            try:
                # Get output directory (root extracted_data folder)
                script_dir = Path(__file__).parent
                project_root = script_dir.parent
                extracted_dir = Path(self.paths_config['paths'].get('output_directory', 'extracted_data'))
                if not extracted_dir.is_absolute():
                    extracted_dir = project_root / extracted_dir
                
                # Look for asset files directly in extracted_data folder (not in subfolders)
                asset_files = list(extracted_dir.glob('asset_nodes_*.json'))
                
                # Also check timestamped subfolders for backward compatibility
                if not asset_files:
                    for subfolder in extracted_dir.iterdir():
                        if subfolder.is_dir() and not subfolder.name.startswith('.'):
                            asset_files.extend(subfolder.glob('asset_nodes_*.json'))
                
                if asset_files:
                    latest_asset_file = max(asset_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_asset_file, 'r', encoding='utf-8') as f:
                        asset_nodes = json.load(f)
                    agent_ids = [node.get('asset_id') for node in asset_nodes if node.get('asset_id')]
                    logger.info(f"Found {len(agent_ids)} agent IDs from asset nodes file: {latest_asset_file}")
                    source_data_list = self.load_data_for_agents(agent_ids)
                else:
                    logger.warning("No asset nodes file found. Cannot determine agent IDs automatically.")
                    source_data_list = []
            except Exception as e:
                logger.warning(f"Could not get agent IDs automatically: {e}")
                source_data_list = []
        
        # Extract nodes
        nodes = []
        properties_config = self.node_config.get('properties', {})
        required_properties = [prop_name for prop_name, prop_config in properties_config.items() 
                              if prop_config.get('required', False)]
        
        for source_data in source_data_list:
            try:
                node = self._extract_node(source_data)
                
                # Validate required properties are not null
                missing_required = [prop for prop in required_properties if node.get(prop) is None]
                if missing_required:
                    logger.warning(f"Skipping {self.node_type} node due to missing required properties: {missing_required}")
                    continue
                
                nodes.append(node)
                logger.debug(f"Extracted {self.node_type} node")
            except Exception as e:
                logger.error(f"Error extracting {self.node_type} node: {e}")
                continue
        
        logger.info(f"Successfully extracted {len(nodes)} {self.node_type} nodes")
        return nodes
    
    def _get_output_directory(self) -> Path:
        """Get the output directory for extracted nodes (root extracted_data folder)"""
        # Get output directory from config, default to 'extracted_data'
        # Use absolute path relative to project root (parent of scripts folder)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = Path(self.paths_config['paths'].get('output_directory', 'extracted_data'))
        
        # If relative path, make it relative to project root
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_extracted_nodes(self, nodes: List[Dict[str, Any]], output_file: Optional[str] = None) -> str:
        """Save extracted nodes to JSON file directly in extracted_data folder"""
        output_dir = self._get_output_directory()
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.node_type}_nodes_{timestamp}.json"
        
        output_path = output_dir / output_file
        
        # Convert datetime objects to ISO format strings for JSON serialization
        serializable_nodes = []
        for node in nodes:
            serializable_node = {}
            for key, value in node.items():
                if isinstance(value, datetime):
                    serializable_node[key] = value.isoformat()
                else:
                    serializable_node[key] = value
            serializable_nodes.append(serializable_node)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_nodes, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(nodes)} {self.node_type} nodes to: {output_path}")
        return str(output_path)


def get_all_node_types(graph_config: Dict[str, Any]) -> List[str]:
    """Get all node types defined in graph_config.yaml"""
    nodes_config = graph_config.get('nodes', {})
    # Filter out relationship definitions (they don't have 'properties' key)
    node_types = []
    for node_type, node_config in nodes_config.items():
        # Node types have 'properties' key, relationships don't
        if isinstance(node_config, dict) and 'properties' in node_config:
            node_types.append(node_type)
    return node_types


def extract_single_node_type(config_loader: ConfigLoader, node_type: str) -> tuple[int, str]:
    """Extract nodes for a single node type"""
    try:
        logger.info("=" * 60)
        logger.info(f"{node_type.upper()} Node Extraction Started")
        logger.info("=" * 60)
        
        # Initialize extractor
        extractor = NodeExtractor(config_loader, node_type)
        
        # Extract nodes
        logger.info(f"\nExtracting {node_type} nodes from collected data...")
        nodes = extractor.extract_all_nodes()
        
        # Save extracted nodes
        logger.info(f"\nSaving extracted {node_type} nodes...")
        output_file = extractor.save_extracted_nodes(nodes)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"{node_type.upper()} Node Extraction Completed!")
        logger.info(f"Extracted {len(nodes)} {node_type} nodes")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 60)
        
        return len(nodes), output_file
    except Exception as e:
        logger.error(f"Error during {node_type} extraction: {e}", exc_info=True)
        raise


def main():
    """Main execution function"""
    import sys
    
    # Get node type from command line argument
    # If no argument or 'all', extract all node types from graph_config
    node_type_arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    try:
        # Load configurations
        config_loader = ConfigLoader()
        
        # Determine which node types to extract
        if node_type_arg.lower() == 'all':
            # Extract all node types defined in graph_config
            node_types = get_all_node_types(config_loader.graph_config)
            logger.info("=" * 60)
            logger.info("EXTRACTING ALL NODE TYPES")
            logger.info("=" * 60)
            logger.info(f"Found {len(node_types)} node types to extract: {', '.join(node_types)}")
            
            results = {}
            for node_type in node_types:
                try:
                    count, output_file = extract_single_node_type(config_loader, node_type)
                    results[node_type] = {'count': count, 'output_file': output_file}
                except Exception as e:
                    logger.error(f"Failed to extract {node_type}: {e}")
                    results[node_type] = {'count': 0, 'output_file': None, 'error': str(e)}
            
            # Print summary
            print("\n" + "=" * 60)
            print("ALL NODE TYPES EXTRACTION SUMMARY")
            print("=" * 60)
            total_nodes = 0
            for node_type, result in results.items():
                count = result.get('count', 0)
                total_nodes += count
                output_file = result.get('output_file', 'N/A')
                error = result.get('error')
                if error:
                    print(f"  {node_type.upper()}: FAILED - {error}")
                else:
                    print(f"  {node_type.upper()}: {count} nodes - {output_file}")
            print(f"\nTotal nodes extracted across all types: {total_nodes}")
            print("=" * 60)
        else:
            # Extract single node type
            node_type = node_type_arg
            count, output_file = extract_single_node_type(config_loader, node_type)
            
            # Print summary
            print("\n" + "=" * 60)
            print(f"{node_type.upper()} Node Extraction Summary")
            print("=" * 60)
            print(f"Total {node_type} nodes extracted: {count}")
            print(f"Output file: {output_file}")
            if count > 0:
                # Load and show sample node
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        nodes = json.load(f)
                        if nodes:
                            print(f"\nSample {node_type} node:")
                            sample = nodes[0]
                            for key, value in sample.items():
                                print(f"  {key}: {value}")
                except Exception:
                    pass
            print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

