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

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/graph_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_latest_extraction_folder() -> Path:
    """Find the latest timestamped extraction folder"""
    extracted_dir = Path('extracted_data')
    
    if not extracted_dir.exists():
        raise FileNotFoundError("extracted_data directory not found")
    
    # Find all timestamped subfolders (format: YYYYMMDD_HHMMSS)
    subfolders = []
    for item in extracted_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it looks like a timestamp folder (YYYYMMDD_HHMMSS)
            if len(item.name) == 15 and item.name[8] == '_':
                try:
                    # Validate it's a valid timestamp format
                    datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                    subfolders.append(item)
                except ValueError:
                    pass
    
    if not subfolders:
        raise FileNotFoundError("No timestamped extraction folders found in extracted_data directory")
    
    # Sort by folder name (which is timestamp) and return latest
    latest_folder = max(subfolders, key=lambda x: x.name)
    logger.info(f"Using latest extraction folder: {latest_folder}")
    return latest_folder


def find_latest_nodes(node_type: str) -> str:
    """Find the latest extracted nodes JSON file for a given node type"""
    # First, look in root extracted_data directory (primary location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    extracted_dir = project_root / 'extracted_data'
    pattern = f"{node_type}_nodes_*.json"
    node_files = list(extracted_dir.glob(pattern))
    
    if node_files:
        # Sort by modification time and return latest
        latest = max(node_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using latest {node_type} nodes file: {latest}")
        return str(latest)
    
    # Fallback: look in timestamped subfolders (for backward compatibility)
    try:
        latest_folder = find_latest_extraction_folder()
        node_files = list(latest_folder.glob(pattern))
        if node_files:
            latest = max(node_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Using latest {node_type} nodes file from timestamped folder: {latest}")
            return str(latest)
    except FileNotFoundError:
        pass
    
    # If still not found, raise error
    raise FileNotFoundError(f"No {pattern} files found in extracted_data directory")


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
        constraints_results = builder.setup_constraints()
        
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
        logger.info("\nCreating relationships...")
        relationship_results = builder.create_relationships()
        
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

