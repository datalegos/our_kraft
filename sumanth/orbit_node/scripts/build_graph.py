"""
Main Graph Builder Script
Builds Neo4j knowledge graph from extracted asset nodes.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_builder.graph_builder import GraphBuilder
from graph_builder.scanevent_manager import ScanEventManager
import json

# Setup logging
Path('logs').mkdir(exist_ok=True)
log_file = 'logs/graph_builder.log'

# Configure root logger to capture all module loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration if already configured
)

# Ensure all graph_builder module loggers write to the file
for logger_name in ['graph_builder', 'graph_builder.graph_builder', 'graph_builder.relationship_manager', 
                    'graph_builder.node_inserter', 'graph_builder.constraints_manager', 
                    'graph_builder.neo4j_connection', 'graph_builder.scanevent_manager']:
    module_logger = logging.getLogger(logger_name)
    module_logger.setLevel(logging.INFO)
    # Add file handler if not already present
    if not any(isinstance(h, logging.FileHandler) for h in module_logger.handlers):
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        module_logger.addHandler(file_handler)
    # Ensure it propagates to root logger
    module_logger.propagate = True

logger = logging.getLogger(__name__)


def find_latest_extraction_folder() -> Path:
    """Find the latest timestamped extraction folder"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    extracted_dir = project_root / 'extracted_data'
    
    if not extracted_dir.exists():
        raise FileNotFoundError(f"extracted_data directory not found: {extracted_dir}")
    
    # Find all timestamped subfolders (format: YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_MMM)
    subfolders = []
    import re
    for item in extracted_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it looks like a timestamp folder (YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_MMM)
            # Pattern: 8 digits, underscore, 6 digits, optional underscore and 3 digits
            if re.match(r'^\d{8}_\d{6}(_\d{3})?$', item.name):
                try:
                    # Try to parse as timestamp (with or without milliseconds)
                    if len(item.name) == 15:  # YYYYMMDD_HHMMSS format
                        datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                    elif len(item.name) == 19:  # YYYYMMDD_HHMMSS_MMM format
                        datetime.strptime(item.name[:15], "%Y%m%d_%H%M%S")
                    subfolders.append(item)
                except ValueError:
                    pass
    
    if not subfolders:
        raise FileNotFoundError(f"No timestamped extraction folders found in extracted_data directory: {extracted_dir}")
    
    # Sort by folder name (which is timestamp) and return latest
    latest_folder = max(subfolders, key=lambda x: x.name)
    logger.info(f"Using latest extraction folder: {latest_folder.name}")
    logger.info(f"Full path: {latest_folder.absolute()}")
    return latest_folder


def find_latest_nodes(node_type: str) -> str:
    """Find the latest extracted nodes JSON file for a given node type in timestamped subfolders"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    extracted_dir = project_root / 'extracted_data'
    
    # Always use the latest timestamped subfolder
    try:
        latest_folder = find_latest_extraction_folder()
        
        # First, try standard filename (node_type_nodes.json) - this is the expected format
        standard_file = latest_folder / f"{node_type}_nodes.json"
        if standard_file.exists():
            logger.info(f"Found {node_type} nodes file in latest extraction folder: {standard_file}")
            return str(standard_file)
        
        # Fallback: look for timestamped filenames (old format for backward compatibility)
        pattern = f"{node_type}_nodes_*.json"
        node_files = list(latest_folder.glob(pattern))
        if node_files:
            # Sort by modification time and return latest
            latest = max(node_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found {node_type} nodes file (timestamped format) in latest folder: {latest}")
            return str(latest)
        
        # If not found in latest folder, raise error with helpful message
        raise FileNotFoundError(
            f"No {node_type}_nodes.json or {node_type}_nodes_*.json files found in latest extraction folder: {latest_folder}\n"
            f"Please run 'python extract_nodes.py {node_type}' first to extract {node_type} nodes."
        )
    except FileNotFoundError as e:
        # Re-raise with more context if it's about missing folder
        if "extracted_data directory not found" in str(e) or "No timestamped extraction folders" in str(e):
            raise FileNotFoundError(
                f"{e}\n"
                f"Please run 'python extract_nodes.py {node_type}' first to extract {node_type} nodes."
            )
        raise


def main():
    """Main execution function"""
    try:
        logger.info("=" * 60)
        logger.info("Neo4j Graph Builder")
        logger.info("=" * 60)
        
        # Find node files
        try:
            asset_json_file = find_latest_nodes("asset")
        except FileNotFoundError as e:
            logger.error(f"Asset nodes file not found: {e}")
            logger.info("Please run extract_nodes.py asset first to extract asset nodes")
            sys.exit(1)
        
        try:
            host_json_file = find_latest_nodes("host")
        except FileNotFoundError as e:
            logger.warning(f"Host nodes file not found: {e}")
            logger.info("Skipping host nodes. Run extract_nodes.py host to extract host nodes")
            host_json_file = None
        
        try:
            software_json_file = find_latest_nodes("software")
        except FileNotFoundError as e:
            logger.warning(f"Software nodes file not found: {e}")
            logger.info("Skipping software nodes. Run extract_nodes.py software to extract software nodes")
            software_json_file = None
        
        try:
            vulnerability_json_file = find_latest_nodes("vulnerability")
        except FileNotFoundError as e:
            logger.warning(f"Vulnerability nodes file not found: {e}")
            logger.info("Skipping vulnerability nodes. Run extract_nodes.py vulnerability to extract vulnerability nodes")
            vulnerability_json_file = None
        
        try:
            assetgroup_json_file = find_latest_nodes("assetgroup")
        except FileNotFoundError as e:
            logger.warning(f"AssetGroup nodes file not found: {e}")
            logger.info("Skipping assetgroup nodes. Run extract_nodes.py assetgroup to extract assetgroup nodes")
            assetgroup_json_file = None
        
        try:
            hardware_json_file = find_latest_nodes("hardware")
        except FileNotFoundError as e:
            logger.warning(f"Hardware nodes file not found: {e}")
            logger.info("Skipping hardware nodes. Run extract_nodes.py hardware to extract hardware nodes")
            hardware_json_file = None
        
        # Initialize graph builder with config paths relative to project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        config_dir = project_root / 'config'
        
        logger.info("\nInitializing graph builder...")
        builder = GraphBuilder(
            neo4j_config_path=str(config_dir / "neo4j_config.yaml"),
            graph_config_path=str(config_dir / "graph_config.yaml")
        )
        
        # Setup constraints for all node types
        logger.info("\nSetting up constraints...")
        try:
            constraints_results = builder.setup_constraints()
            logger.info("Constraints setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up constraints: {e}", exc_info=True)
            raise
        
        # Create ScanEvent first (Day-0 baseline load)
        logger.info("\n" + "=" * 60)
        logger.info("Creating ScanEvent")
        logger.info("=" * 60)
        sys.stdout.flush()
        logging.getLogger().handlers[0].flush()
        
        scanevent_manager = ScanEventManager(
            builder.neo4j_conn,
            graph_config_path=str(config_dir / "graph_config.yaml")
        )
        
        # Get job start time from asset file modification time (or use current time)
        job_start_time = datetime.now()
        try:
            asset_file_path = Path(asset_json_file)
            if asset_file_path.exists():
                job_start_time = datetime.fromtimestamp(asset_file_path.stat().st_mtime)
        except Exception:
            pass
        
        scanevent_result = scanevent_manager.create_scanevent(
            description="STANDARD_BASELINE_LOAD",
            job_start_time=job_start_time,
            status="Created"
        )
        
        event_id = scanevent_result['event_id']
        logger.info(f"ScanEvent created: {event_id}")
        logger.info("=" * 60)
        
        # Helper function to add event_id to nodes in JSON file
        def add_event_id_to_nodes(json_file: str, event_id: str) -> str:
            """Add event_id to all nodes in JSON file and return updated file path"""
            if not json_file:
                return json_file
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    nodes = json.load(f)
                
                # Add event_id to each node
                for node in nodes:
                    node['event_id'] = event_id
                
                # Save updated nodes to temporary file
                temp_file = json_file.replace('.json', '_with_event_id.json')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(nodes, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Added event_id to {len(nodes)} nodes in {json_file}")
                return temp_file
            except Exception as e:
                logger.warning(f"Could not add event_id to {json_file}: {e}")
                return json_file
        
        # Add event_id to all node files
        logger.info(f"\nAdding event_id ({event_id}) to all nodes...")
        asset_json_file = add_event_id_to_nodes(asset_json_file, event_id)
        if host_json_file:
            host_json_file = add_event_id_to_nodes(host_json_file, event_id)
        if software_json_file:
            software_json_file = add_event_id_to_nodes(software_json_file, event_id)
        if vulnerability_json_file:
            vulnerability_json_file = add_event_id_to_nodes(vulnerability_json_file, event_id)
        if assetgroup_json_file:
            assetgroup_json_file = add_event_id_to_nodes(assetgroup_json_file, event_id)
        if hardware_json_file:
            hardware_json_file = add_event_id_to_nodes(hardware_json_file, event_id)
        
        # Insert asset nodes
        logger.info("\nInserting asset nodes...")
        asset_results = builder.insert_asset_nodes(asset_json_file)
        
        # Insert host nodes if available
        host_results = None
        if host_json_file:
            logger.info("\nInserting host nodes...")
            host_results = builder.insert_host_nodes(host_json_file)
        
        # Insert software nodes if available
        software_results = None
        if software_json_file:
            logger.info("\nInserting software nodes...")
            software_results = builder.insert_nodes(
                "software",
                software_json_file,
                merge_key="name"
            )
        
        # Insert vulnerability nodes if available
        vulnerability_results = None
        if vulnerability_json_file:
            logger.info("\nInserting vulnerability nodes...")
            vulnerability_results = builder.insert_nodes(
                "vulnerability",
                vulnerability_json_file,
                merge_key="cve_id"
            )
        
        # Insert assetgroup nodes if available
        assetgroup_results = None
        if assetgroup_json_file:
            logger.info("\nInserting assetgroup nodes...")
            assetgroup_results = builder.insert_nodes(
                "assetgroup",
                assetgroup_json_file,
                merge_key="name"
            )
        
        # Insert hardware nodes if available
        hardware_results = None
        if hardware_json_file:
            logger.info("\nInserting hardware nodes...")
            hardware_results = builder.insert_nodes(
                "hardware",
                hardware_json_file,
                composite_keys=["agent_id", "name"]
            )
        
        # Create relationships
        logger.info("\n" + "=" * 60)
        logger.info("CREATING RELATIONSHIPS")
        logger.info("=" * 60)
        logger.info("Starting relationship creation process...")
        try:
            relationship_results = builder.create_relationships()
            logger.info("Relationship creation process completed")
            logger.info(f"Relationship results: {relationship_results}")
        except Exception as e:
            logger.error(f"Error during relationship creation: {e}", exc_info=True)
            raise
        logger.info("=" * 60)
        
        # Compile results
        results = {
            'constraints': constraints_results,
            'nodes': {
                'asset': asset_results,
                'host': host_results,
                'software': software_results,
                'vulnerability': vulnerability_results,
                'assetgroup': assetgroup_results,
                'hardware': hardware_results
            },
            'relationships': relationship_results
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("Graph Building Summary")
        print("=" * 60)
        
        # Constraints summary
        constraints_results = results.get('constraints', {})
        if constraints_results:
            print("\nConstraints Created:")
            for node_type, constraints in constraints_results.items():
                if isinstance(constraints, dict):
                    success = sum(1 for v in constraints.values() if v)
                    total = len(constraints)
                    print(f"  {node_type}: {success}/{total} constraints created")
        
        # Nodes summary
        nodes_results = results.get('nodes', {})
        if nodes_results:
            print("\nNodes Inserted:")
            if nodes_results.get('asset'):
                asset = nodes_results['asset']
                print(f"  Asset - Success: {asset.get('success', 0)}, Failed: {asset.get('failed', 0)}, Total: {asset.get('total', 0)}")
            if nodes_results.get('host'):
                host = nodes_results['host']
                print(f"  Host - Success: {host.get('success', 0)}, Failed: {host.get('failed', 0)}, Total: {host.get('total', 0)}")
            if nodes_results.get('software'):
                software = nodes_results['software']
                print(f"  Software - Success: {software.get('success', 0)}, Failed: {software.get('failed', 0)}, Total: {software.get('total', 0)}")
            if nodes_results.get('vulnerability'):
                vulnerability = nodes_results['vulnerability']
                print(f"  Vulnerability - Success: {vulnerability.get('success', 0)}, Failed: {vulnerability.get('failed', 0)}, Total: {vulnerability.get('total', 0)}")
            if nodes_results.get('assetgroup'):
                assetgroup = nodes_results['assetgroup']
                print(f"  AssetGroup - Success: {assetgroup.get('success', 0)}, Failed: {assetgroup.get('failed', 0)}, Total: {assetgroup.get('total', 0)}")
            if nodes_results.get('hardware'):
                hardware = nodes_results['hardware']
                print(f"  Hardware - Success: {hardware.get('success', 0)}, Failed: {hardware.get('failed', 0)}, Total: {hardware.get('total', 0)}")
        
        # Relationships summary
        relationship_results = results.get('relationships', {})
        if relationship_results:
            print("\nRelationships Created:")
            for rel_name, rel_data in relationship_results.items():
                if isinstance(rel_data, dict):
                    success = rel_data.get('success', 0)
                    total = rel_data.get('total', 0)
                    print(f"  {rel_name}: {success} relationships created")
        
        print("=" * 60)
        
        # Close connection
        builder.close()
        
        logger.info("Graph building process completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during graph building: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

