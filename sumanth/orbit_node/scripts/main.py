"""
Main script for Wazuh Knowledge Graph Data Collection
Collects data from Wazuh Manager and Indexer based on configuration
Converts collected JSON data to CSV format
"""
import sys
from pathlib import Path
import yaml
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.wazuh_indexer_client import WazuhIndexerClient
from utils.wazuh_manager_client import WazuhManagerClient
from utils.data_collector import (
    AgentCollector, HostCollector, PackagesCollector,
    HardwareCollector, GroupsCollector, FIMCollector,
    VulnerabilitiesCollector, CPECollector
)
from utils.csv_converter import convert_collected_data_to_csv


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_agent_ids(agents_data: dict) -> list:
    """Extract agent IDs from agents data"""
    agent_ids = []
    try:
        agents = agents_data.get('data', {}).get('affected_items', [])
        agent_ids = [agent.get('id') for agent in agents if agent.get('id')]
    except Exception as e:
        print(f"Error extracting agent IDs: {e}")
    return agent_ids


def main():
    """Main function to collect knowledge graph data"""
    print("=" * 70)
    print("Wazuh Knowledge Graph Data Collection System")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    
    # Create timestamped folder for this collection run
    base_output_dir = Path(config['collection']['output_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = base_output_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with timestamped directory
    config['collection']['output_dir'] = str(timestamped_dir)
    
    print(f"\nCollection run timestamp: {timestamp}")
    print(f"Output directory: {timestamped_dir}\n")
    
    # Initialize clients
    manager_client = None
    indexer_client = None
    
    # Test Manager connection if needed
    manager_needed = any(
        config['collectors'].get(c, {}).get('source') == 'manager'
        for c in config['collection']['enabled_collectors']
    )
    
    if manager_needed:
        print("Testing Wazuh Manager connection...")
        manager_client = WazuhManagerClient(config)
        manager_connected = manager_client.test_connection()
        if manager_connected:
            print("✓ Wazuh Manager connection successful!\n")
        else:
            print("✗ Failed to connect to Wazuh Manager.\n")
            manager_client = None
    
    # Test Indexer connection if needed
    indexer_needed = any(
        config['collectors'].get(c, {}).get('source') == 'indexer'
        for c in config['collection']['enabled_collectors']
    )
    
    if indexer_needed:
        print("Testing Wazuh Indexer connection...")
        indexer_client = WazuhIndexerClient(config)
        indexer_connected = indexer_client.test_connection()
        if indexer_connected:
            print("✓ Wazuh Indexer connection successful!\n")
        else:
            print("✗ Failed to connect to Wazuh Indexer.\n")
            indexer_client = None
    
    if not manager_client and not indexer_client:
        print("\nBoth connections failed. Please check your configuration.")
        return
    
    print("=" * 70)
    print("Starting Data Collection")
    print("=" * 70 + "\n")
    
    enabled_collectors = config['collection']['enabled_collectors']
    agent_ids = []
    
    # Step 1: Collect agents first (needed for other collectors)
    if 'agents' in enabled_collectors:
        print("\n" + "-" * 70)
        print("COLLECTING AGENTS")
        print("-" * 70)
        
        agent_source = config['collectors']['agents'].get('source', 'manager')
        if agent_source == 'manager' and manager_client:
            agent_collector = AgentCollector(manager_client, config, 'manager')
            agents_data = agent_collector.collect()
            agent_ids = extract_agent_ids(agents_data)
        elif agent_source == 'indexer' and indexer_client:
            agent_collector = AgentCollector(indexer_client, config, 'indexer')
            agents_data = agent_collector.collect()
            agent_ids = extract_agent_ids(agents_data)
        
        print(f"Found {len(agent_ids)} agents: {agent_ids}\n")
    
    # Step 2: Collect other data types
    if 'host' in enabled_collectors and manager_client and agent_ids:
        print("\n" + "-" * 70)
        print("COLLECTING HOST/OS INFORMATION")
        print("-" * 70)
        host_collector = HostCollector(manager_client, config)
        host_collector.collect(agent_ids)
    
    if 'packages' in enabled_collectors and manager_client and agent_ids:
        print("\n" + "-" * 70)
        print("COLLECTING PACKAGES")
        print("-" * 70)
        packages_collector = PackagesCollector(manager_client, config)
        packages_collector.collect(agent_ids)
    
    if 'hardware' in enabled_collectors and manager_client and agent_ids:
        print("\n" + "-" * 70)
        print("COLLECTING HARDWARE")
        print("-" * 70)
        hardware_collector = HardwareCollector(manager_client, config)
        hardware_collector.collect(agent_ids)
    
    if 'cpe' in enabled_collectors and manager_client and agent_ids:
        print("\n" + "-" * 70)
        print("COLLECTING CPE (COMMON PLATFORM ENUMERATION)")
        print("-" * 70)
        cpe_collector = CPECollector(manager_client, config)
        cpe_collector.collect(agent_ids)
    
    if 'groups' in enabled_collectors and manager_client:
        print("\n" + "-" * 70)
        print("COLLECTING GROUPS")
        print("-" * 70)
        groups_collector = GroupsCollector(manager_client, config)
        groups_collector.collect()
    
    if 'fim' in enabled_collectors and manager_client and agent_ids:
        print("\n" + "-" * 70)
        print("COLLECTING FIM (FILE INTEGRITY MONITORING)")
        print("-" * 70)
        fim_collector = FIMCollector(manager_client, config)
        fim_collector.collect(agent_ids)
    
    if 'vulnerabilities' in enabled_collectors and indexer_client:
        print("\n" + "-" * 70)
        print("COLLECTING VULNERABILITIES")
        print("-" * 70)
        vulnerabilities_collector = VulnerabilitiesCollector(indexer_client, config)
        vulnerabilities_collector.collect(agent_ids if agent_ids else None)
    
    # Step 3: Convert JSON to CSV
    print("\n" + "=" * 70)
    print("CONVERTING JSON TO CSV")
    print("=" * 70 + "\n")
    
    csv_output_dir = convert_collected_data_to_csv(timestamped_dir)
    
    print("\n" + "=" * 70)
    print("Data Collection Completed!")
    print("=" * 70)
    print(f"JSON data saved to: {timestamped_dir}")
    print(f"CSV data saved to: {csv_output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
