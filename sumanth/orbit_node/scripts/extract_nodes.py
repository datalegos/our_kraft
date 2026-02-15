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
        """Get the base directory for collected data - always use latest folder"""
        # Always find the latest timestamped folder, regardless of config
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        collected_data_dir = project_root / 'collected_data'
        
        if collected_data_dir.exists():
            import re
            # Find all timestamped folders (format: YYYYMMDD_HHMMSS)
            timestamped_folders = [
                d for d in collected_data_dir.iterdir() 
                if d.is_dir() and re.match(r'^\d{8}_\d{6}$', d.name)
            ]
            
            if timestamped_folders:
                # Sort by folder name (which is timestamp) and get latest
                latest = max(timestamped_folders, key=lambda x: x.name)
                logger.info(f"Using latest collected data folder: {latest.name}")
                logger.info(f"Full path: {latest}")
                return latest
        
        # Fallback to configured base_directory
        base_path = self.paths_config['paths'].get('base_directory', 'collected_data')
        base_dir = Path(base_path)
        if not base_dir.is_absolute():
            base_dir = project_root / base_dir
        
        if base_dir.exists():
            logger.warning(f"Using configured base directory (no timestamped folders found): {base_dir}")
            return base_dir
        
        raise ValueError(f"No valid collected_data directory found. Checked: {collected_data_dir}")
    
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
    
    def load_data_for_agents(self, agent_ids: Optional[List[str]] = None) -> List[Dict]:
        """Load data files for specific agents or summary file"""
        file_sources = self.graph_config.get('file_sources', {})
        source_config = file_sources.get(self.node_type)
        
        if not source_config:
            raise ValueError(f"File source configuration not found for node type '{self.node_type}'")
        
        directory_key = source_config.get('directory_key')
        file_pattern = source_config.get('file_pattern', '')
        per_agent = source_config.get('per_agent', False)
        
        data_dir = self.base_dir / self.paths_config['paths']['data_sources'][directory_key]['directory']
        all_data = []
        
        logger.info(f"Loading {self.node_type} data from: {data_dir}")
        logger.info(f"File pattern: {file_pattern}, per_agent: {per_agent}, agent_ids: {agent_ids}")
        
        if per_agent:
            # Load per-agent files
            if not agent_ids:
                logger.error(f"Cannot load per-agent files for {self.node_type} without agent IDs")
                return []
            
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
            # Load single summary file (for asset, assetgroup, etc.)
            # Use file_pattern from file_sources config, not from paths_config
            summary_file = file_pattern
            if '{agent_id}' in summary_file:
                # Remove agent_id placeholder for summary files
                summary_file = summary_file.replace('_{agent_id}', '').replace('{agent_id}', '')
            
            file_path = data_dir / summary_file
            
            logger.info(f"Loading summary file: {file_path}")
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different data structures
                    if 'data' in data and 'affected_items' in data['data']:
                        all_data = data['data']['affected_items']
                        logger.info(f"Loaded {len(all_data)} items from {file_path}")
                    elif isinstance(data, dict):
                        # Handle per-agent structure (e.g., {"000": {...}, "001": {...}})
                        for agent_id, agent_data in data.items():
                            if 'data' in agent_data and 'affected_items' in agent_data['data']:
                                all_data.extend(agent_data['data']['affected_items'])
                        logger.info(f"Loaded {len(all_data)} items from {file_path} (per-agent structure)")
                    else:
                        logger.warning(f"Unexpected data structure in {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.error(f"Summary file not found: {file_path}")
                logger.error(f"Expected directory: {data_dir}")
                logger.error(f"Expected file: {summary_file}")
        
        logger.info(f"Total items loaded for {self.node_type}: {len(all_data)}")
        return all_data
    
    def extract_all_nodes(self, agent_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract all nodes of this type from collected data"""
        # Check if this node type requires per-agent files
        file_sources = self.graph_config.get('file_sources', {})
        source_config = file_sources.get(self.node_type, {})
        per_agent = source_config.get('per_agent', False)
        
        # Load source data
        if agent_ids:
            source_data_list = self.load_data_for_agents(agent_ids)
        elif not per_agent:
            # For non-per-agent files (like asset, assetgroup), load directly from summary file
            logger.info(f"Loading {self.node_type} data from summary file (per_agent: false)")
            source_data_list = self.load_data_for_agents(None)  # Pass None to load summary file
        else:
            # For per-agent files, need agent IDs - try to get from existing asset nodes or All_Agents.json
            logger.info(f"Loading {self.node_type} data requires agent IDs (per_agent: true)")
            
            # First, try to get agent IDs from All_Agents.json directly
            try:
                agents_dir_key = 'agents_manager'
                agents_data_dir = self.base_dir / self.paths_config['paths']['data_sources'][agents_dir_key]['directory']
                agents_file = agents_data_dir / 'All_Agents.json'
                
                if agents_file.exists():
                    with open(agents_file, 'r', encoding='utf-8') as f:
                        agents_data = json.load(f)
                    
                    # Extract agent IDs from All_Agents.json
                    if 'data' in agents_data and 'affected_items' in agents_data['data']:
                        agent_ids = [item.get('id') for item in agents_data['data']['affected_items'] if item.get('id')]
                        logger.info(f"Found {len(agent_ids)} agent IDs from All_Agents.json: {agent_ids}")
                        source_data_list = self.load_data_for_agents(agent_ids)
                    else:
                        logger.warning("All_Agents.json does not have expected structure")
                        source_data_list = []
                else:
                    logger.warning(f"All_Agents.json not found at {agents_file}")
                    source_data_list = []
            except Exception as e:
                logger.warning(f"Could not get agent IDs from All_Agents.json: {e}")
                # Fallback: try to get from existing asset nodes file
                try:
                    script_dir = Path(__file__).parent
                    project_root = script_dir.parent
                    extracted_dir = Path(self.paths_config['paths'].get('output_directory', 'extracted_data'))
                    if not extracted_dir.is_absolute():
                        extracted_dir = project_root / extracted_dir
                    
                    asset_files = []
                    import re
                    for subfolder in extracted_dir.iterdir():
                        if subfolder.is_dir() and re.match(r'^\d{8}_\d{6}$', subfolder.name):
                            asset_files.extend(subfolder.glob('asset_nodes_*.json'))
                    
                    if not asset_files:
                        asset_files = list(extracted_dir.glob('asset_nodes_*.json'))
                    
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
                except Exception as e2:
                    logger.warning(f"Could not get agent IDs automatically: {e2}")
                    source_data_list = []
        
        # Extract nodes
        nodes = []
        properties_config = self.node_config.get('properties', {})
        # Exclude event_id from required validation during extraction - it will be added during graph building
        required_properties = [
            prop_name for prop_name, prop_config in properties_config.items() 
            if prop_config.get('required', False) and prop_name != 'event_id'
        ]
        
        for source_data in source_data_list:
            try:
                node = self._extract_node(source_data)
                
                # Validate required properties are not null (excluding event_id)
                missing_required = [prop for prop in required_properties if node.get(prop) is None]
                if missing_required:
                    logger.warning(f"Skipping {self.node_type} node due to missing required properties: {missing_required}")
                    continue
                
                # Note: event_id will be added during graph building, not during extraction
                # Remove event_id from node if it's None (it shouldn't be set during extraction)
                if 'event_id' in node and node['event_id'] is None:
                    del node['event_id']
                
                nodes.append(node)
                logger.debug(f"Extracted {self.node_type} node")
            except Exception as e:
                logger.error(f"Error extracting {self.node_type} node: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"Successfully extracted {len(nodes)} {self.node_type} nodes")
        return nodes
    
    def _get_output_directory(self) -> Path:
        """Get the output directory for extracted nodes (timestamped subfolder in extracted_data)"""
        # Get output directory from config, default to 'extracted_data'
        # Use absolute path relative to project root (parent of scripts folder)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        base_output_dir = Path(self.paths_config['paths'].get('output_directory', 'extracted_data'))
        
        # If relative path, make it relative to project root
        if not base_output_dir.is_absolute():
            base_output_dir = project_root / base_output_dir
        
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create or use existing timestamped session folder
        session_folder = self._get_extraction_session_folder(base_output_dir)
        return session_folder
    
    def _get_extraction_session_folder(self, base_output_dir: Path) -> Path:
        """Create a new timestamped extraction session folder (always creates new folder)"""
        # Always create a new timestamped folder for each extraction run
        # Use milliseconds to ensure uniqueness even if called multiple times in the same second
        current_time = datetime.now()
        # Format: YYYYMMDD_HHMMSS_MMM (with milliseconds for uniqueness)
        microseconds = current_time.microsecond
        milliseconds = microseconds // 1000  # Convert to milliseconds (0-999)
        session_timestamp = current_time.strftime("%Y%m%d_%H%M%S") + f"_{milliseconds:03d}"
        session_folder = base_output_dir / session_timestamp
        session_folder.mkdir(parents=True, exist_ok=True)
        
        # Update session marker to track the latest session
        session_marker = base_output_dir / '.current_session'
        try:
            with open(session_marker, 'w') as f:
                f.write(session_timestamp)
        except Exception:
            pass
        
        # Store session timestamp as instance variable for consistent filenames
        self._session_timestamp = session_timestamp
        
        logger.info(f"Created new extraction session folder: {session_folder}")
        return session_folder
    
    def save_extracted_nodes(self, nodes: List[Dict[str, Any]], output_file: Optional[str] = None) -> str:
        """Save extracted nodes to JSON file in timestamped session folder"""
        output_dir = self._get_output_directory()
        
        if output_file is None:
            # Always use standard filename: node_type_nodes.json
            # This ensures consistent naming and easy lookup by build_graph.py
            output_file = f"{self.node_type}_nodes.json"
        
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
        # Exclude scanevent as it's created separately
        if isinstance(node_config, dict) and 'properties' in node_config and node_type != 'scanevent':
            node_types.append(node_type)
    return node_types


def verify_collections_successful(base_dir: Path, paths_config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Verify that all collections were successful (no failed transactions)
    
    Args:
        base_dir: Base directory for collected data
        paths_config: Paths configuration
    
    Returns:
        Tuple of (all_successful, list_of_failed_collectors)
    """
    failed_collectors = []
    data_sources = paths_config.get('paths', {}).get('data_sources', {})
    
        # Check each data source
    logger.info(f"Verifying collections in base directory: {base_dir}")
    for source_key, source_config in data_sources.items():
        directory = source_config.get('directory', '')
        source_dir = base_dir / directory
        
        logger.debug(f"Checking {source_key} in directory: {source_dir}")
        
        if not source_dir.exists():
            logger.warning(f"Directory not found: {source_dir}")
            failed_collectors.append(f"{source_key} (directory not found: {source_dir})")
            continue
        
        # Check for summary files or per-agent files
        file_pattern = source_config.get('file_pattern', '')
        
        # Look for summary files
        if 'Summary' in file_pattern or 'List' in file_pattern or 'All_' in file_pattern:
            # Check for summary file
            summary_files = list(source_dir.glob('*Summary*.json')) + list(source_dir.glob('*List*.json')) + list(source_dir.glob('All_*.json'))
            if summary_files:
                # Check if file has errors (error != 0 means failure, error == 0 or missing means success)
                has_error = False
                for summary_file in summary_files:
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # Check error field: 0 means success, non-zero means failure
                            error_code = data.get('error', 0)
                            logger.debug(f"{source_key} - error code: {error_code}")
                            
                            if error_code != 0:
                                logger.warning(f"{source_key} has error code {error_code} in {summary_file.name}")
                                failed_collectors.append(f"{source_key} (error code {error_code} in {summary_file.name})")
                                has_error = True
                                break
                            
                            # Check for failed items in data
                            if isinstance(data, dict) and 'data' in data:
                                total_failed = data['data'].get('total_failed_items', 0)
                                total_affected = data['data'].get('total_affected_items', 0)
                                logger.debug(f"{source_key} - total_affected: {total_affected}, total_failed: {total_failed}")
                                
                                if total_failed > 0:
                                    logger.warning(f"{source_key} has {total_failed} failed items in {summary_file.name}")
                                    failed_collectors.append(f"{source_key} ({total_failed} failed items in {summary_file.name})")
                                    has_error = True
                                    break
                                
                                # Success - file has error: 0 and total_failed_items: 0
                                logger.debug(f"✓ {source_key} file {summary_file.name} is valid (error: {error_code}, failed: {total_failed})")
                    except Exception as e:
                        logger.warning(f"Could not verify {summary_file}: {e}")
                        failed_collectors.append(f"{source_key} (could not read {summary_file.name}: {e})")
                        has_error = True
                        break
                
                if not has_error:
                    logger.debug(f"✓ {source_key} verification passed")
            else:
                # No summary file found
                failed_collectors.append(f"{source_key} (no summary file found)")
        else:
            # Per-agent files - check if any agent files exist
            agent_dirs = [d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith('agent_')]
            if not agent_dirs:
                failed_collectors.append(f"{source_key} (no agent data found)")
            else:
                # Check a sample of agent files for errors
                errors_found = False
                checked_count = 0
                for agent_dir in agent_dirs[:5]:  # Check first 5 agents
                    agent_files = list(agent_dir.glob('*.json'))
                    if agent_files:
                        try:
                            with open(agent_files[0], 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                                # Check error field: 0 means success, non-zero means failure
                                error_code = data.get('error', 0)
                                if error_code != 0:
                                    errors_found = True
                                    failed_collectors.append(f"{source_key} (error code {error_code} in {agent_files[0].name})")
                                    break
                                
                                # Check for failed items
                                if isinstance(data, dict) and 'data' in data:
                                    total_failed = data['data'].get('total_failed_items', 0)
                                    if total_failed > 0:
                                        errors_found = True
                                        failed_collectors.append(f"{source_key} ({total_failed} failed items in {agent_files[0].name})")
                                        break
                                
                                checked_count += 1
                        except Exception as e:
                            logger.warning(f"Could not verify {agent_files[0]}: {e}")
                            # Don't fail on read errors, just log
                
                if not errors_found and checked_count > 0:
                    logger.debug(f"✓ {source_key} verification passed (checked {checked_count} agent files)")
                elif checked_count == 0:
                    # No valid files found to check
                    failed_collectors.append(f"{source_key} (no valid agent files found to verify)")
    
        all_successful = len(failed_collectors) == 0
    
    if all_successful:
        logger.info(f"✓ All {len(data_sources)} collectors verified successfully")
    else:
        logger.warning(f"✗ {len(failed_collectors)} out of {len(data_sources)} collectors failed verification")
        for failed in failed_collectors:
            logger.warning(f"  - {failed}")
    
    return all_successful, failed_collectors


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
        
        # Verify collections are successful before extraction
        # Get base directory by finding latest collected_data folder
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        collected_data_dir = project_root / 'collected_data'
        
        if collected_data_dir.exists():
            import re
            timestamped_folders = [
                d for d in collected_data_dir.iterdir() 
                if d.is_dir() and re.match(r'^\d{8}_\d{6}$', d.name)
            ]
            if timestamped_folders:
                base_dir = max(timestamped_folders, key=lambda x: x.name)
                logger.info(f"Using latest collected data folder for verification: {base_dir.name}")
            else:
                base_dir = collected_data_dir
                logger.warning("No timestamped folders found, using collected_data root")
        else:
            raise FileNotFoundError(f"collected_data directory not found: {collected_data_dir}")
        
        logger.info("=" * 60)
        logger.info("VERIFYING COLLECTION STATUS")
        logger.info("=" * 60)
        
        all_successful, failed_collectors = verify_collections_successful(base_dir, config_loader.paths_config)
        
        if not all_successful:
            logger.error("=" * 60)
            logger.error("COLLECTION VERIFICATION FAILED")
            logger.error("=" * 60)
            logger.error(f"Found {len(failed_collectors)} failed collectors:")
            for failed in failed_collectors:
                logger.error(f"  - {failed}")
            logger.error("\nPlease retry data collection before proceeding with extraction.")
            logger.error("The data collector will automatically retry failed collections.")
            raise Exception(f"Collection verification failed. {len(failed_collectors)} collectors have errors.")
        
        logger.info("✓ All collections verified successful - no failed transactions")
        logger.info("=" * 60)
        
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

