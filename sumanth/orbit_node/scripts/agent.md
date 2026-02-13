# Module: scripts

## Purpose
This module contains executable scripts that orchestrate the data collection, node extraction, and graph building processes. These scripts serve as the main entry points for the knowledge graph system.

## Key Components
- `main.py`: Main data collection script
  - Orchestrates data collection from Wazuh Manager and Indexer
  - Collects agents, host, packages, hardware, groups, FIM, and vulnerabilities data
  - Converts collected JSON data to CSV format
  - Creates timestamped output directories
  
- `extract_nodes.py`: Generic node extractor script
  - Extracts nodes from collected Wazuh data based on configuration
  - Supports multiple node types (Asset, Host, Software, Vulnerability, AssetGroup, Hardware)
  - Uses `graph_config.yaml` to determine node properties and extraction rules
  - Outputs extracted nodes as JSON files in timestamped folders
  
- `build_graph.py`: Graph builder script
  - Builds Neo4j knowledge graph from extracted asset nodes
  - Finds latest extraction folders automatically
  - Sets up constraints, inserts nodes, and creates relationships
  - Provides summary of graph building operations

- `debug/`: Debug utilities
  - `test_neo4j_connection.py`: Test Neo4j connection
  - `debug_neo4j_auth.py`: Debug Neo4j authentication issues
  - `fix_neo4j_auth.py`: Fix Neo4j authentication problems

## Dependencies
- **External Libraries**: `yaml`, `json`, `pathlib`, `logging`, `datetime`
- **Config Files**: 
  - `../config/config.yaml`: Wazuh Manager/Indexer connection settings and collection configuration
  - `../config/graph_config.yaml`: Node definitions, properties, and extraction rules
  - `../config/neo4j_config.yaml`: Neo4j connection settings (for build_graph.py)
  - `../config/paths_config.yaml`: File paths configuration (for relationship creation)
- **Other Modules**: 
  - `../graph_builder/`: GraphBuilder class and related modules
  - `../utils/`: Wazuh clients, data collectors, and CSV converter
- **Data Sources**: 
  - Input: `../collected_data/` - Collected Wazuh data
  - Output: `../extracted_data/` - Extracted node JSON files
  - Output: Neo4j database (for build_graph.py)

## Entry Points
- `main.py`: Main data collection entry point
  - Run with: `python scripts/main.py`
  - Collects data from Wazuh services based on configuration
  
- `extract_nodes.py`: Node extraction entry point
  - Run with: `python scripts/extract_nodes.py <node_type>`
  - Supported node types: `asset`, `host`, `software`, `vulnerability`, `assetgroup`, `hardware`
  - Example: `python scripts/extract_nodes.py asset`
  
- `build_graph.py`: Graph building entry point
  - Run with: `python scripts/build_graph.py`
  - Automatically finds latest extracted nodes
  - Builds complete Neo4j knowledge graph

## Configuration
- **Data Collection**: Configured via `config/config.yaml`
  - Wazuh Manager/Indexer connection settings
  - Enabled collectors list
  - Output directory settings
  
- **Node Extraction**: Configured via `config/graph_config.yaml`
  - Node type definitions
  - Property mappings
  - Extraction rules
  
- **Graph Building**: Configured via:
  - `config/neo4j_config.yaml`: Database connection
  - `config/graph_config.yaml`: Graph schema and relationships

## Data Flow

### Data Collection Flow (main.py):
1. Load configuration from `config/config.yaml`
2. Initialize Wazuh Manager and/or Indexer clients
3. Test connections
4. Collect agents first (needed for other collectors)
5. Collect other data types (host, packages, hardware, groups, FIM, vulnerabilities)
6. Save data to timestamped folders in `collected_data/`
7. Convert JSON to CSV format

### Node Extraction Flow (extract_nodes.py):
1. Load `graph_config.yaml` and `paths_config.yaml`
2. Find latest collected data folder
3. Extract nodes based on node type and configuration
4. Map properties according to graph_config
5. Save extracted nodes to timestamped folder in `extracted_data/`

### Graph Building Flow (build_graph.py):
1. Find latest extraction folder in `extracted_data/`
2. Load node JSON files (asset, host, software, etc.)
3. Initialize GraphBuilder with Neo4j and graph configs
4. Setup constraints and indexes
5. Insert nodes into Neo4j
6. Create relationships between nodes
7. Provide summary of operations

## Usage Examples

### Data Collection:
```bash
python scripts/main.py
```

### Extract Asset Nodes:
```bash
python scripts/extract_nodes.py asset
```

### Extract Multiple Node Types:
```bash
python scripts/extract_nodes.py asset
python scripts/extract_nodes.py host
python scripts/extract_nodes.py software
```

### Build Graph:
```bash
python scripts/build_graph.py
```

## Notes
- All scripts create timestamped output directories for traceability
- Scripts handle missing data gracefully with warnings
- Logging is configured to both files and console
- The extraction script supports both timestamped folders and legacy flat structure
- Graph builder automatically finds the latest extraction folder

