# Wazuh Knowledge Graph Builder

A configurable system for collecting data from Wazuh Manager and Indexer, extracting graph nodes, and building a Neo4j knowledge graph.

## Project Structure

```
orbit_node/
├── config/                  # Configuration files
│   ├── config.yaml          # Wazuh connection and collection settings
│   ├── graph_config.yaml    # Graph node definitions, properties, constraints, relationships
│   ├── neo4j_config.yaml    # Neo4j database connection settings
│   └── paths_config.yaml    # File paths for data sources
│
├── scripts/                 # Scripts to run
│   ├── main.py              # Collect data from Wazuh
│   ├── build_graph.py        # Build Neo4j graph (run this)
│   ├── extract_nodes.py      # Extract nodes from collected data
│   └── debug/                # Debug tools
│
├── graph_builder/           # Graph building code (used by build_graph.py)
│   ├── graph_builder.py     # Main class
│   ├── neo4j_connection.py  # Database connection
│   ├── constraints_manager.py
│   ├── node_inserter.py
│   └── relationship_manager.py
│
├── utils/                   # Helper functions
│   ├── data_collector.py
│   ├── wazuh_manager_client.py
│   ├── wazuh_indexer_client.py
│   └── csv_converter.py
│
├── collected_data/         # Raw collected data (timestamped folders)
├── extracted_data/         # Processed/extracted node data
├── logs/                   # Application logs
│
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── ARCHITECTURE.md        # Architecture and design decisions
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
Edit configuration files in `config/`:
- `config.yaml`: Wazuh connection settings
- `neo4j_config.yaml`: Neo4j database connection
- `graph_config.yaml`: Graph schema definitions
- `paths_config.yaml`: Data file paths

### 3. Collect Data
```bash
python scripts/main.py
```

### 4. Extract Nodes
```bash
python scripts/extract_nodes.py asset
python scripts/extract_nodes.py host
python scripts/extract_nodes.py software
python scripts/extract_nodes.py vulnerability
python scripts/extract_nodes.py assetgroup
python scripts/extract_nodes.py hardware
```

### 5. Build Graph
```bash
python scripts/build_graph.py
```
This builds the Neo4j graph from extracted nodes.

## Key Features

### Source-Based Relationship Matching
All relationships use **source-based matching** for efficiency:
- Relationships are created directly from source files
- No redundant data stored in nodes
- Always uses fresh data from collected files
- Fully configurable via `graph_config.yaml`

### Configurable Graph Schema
- **Node Properties**: Defined in `graph_config.yaml`
- **Constraints**: UNIQUE and composite constraints configurable
- **Relationships**: Source paths and matching criteria configurable
- **File Sources**: Data file locations configurable

### Professional Folder Structure
- Clear separation of concerns
- Easy to navigate and maintain
- Scalable for future additions

## Documentation

- **ARCHITECTURE.md**: Detailed architecture and design decisions
- **ENDPOINTS_REFERENCE.md**: Wazuh API endpoints documentation

## Configuration

All configurations are in YAML format in the `config/` directory:

- **graph_config.yaml**: Defines nodes, properties, constraints, and relationships
- **paths_config.yaml**: Defines data source file paths
- **neo4j_config.yaml**: Neo4j connection settings
- **config.yaml**: Wazuh API/Indexer connection settings

## Notes

- **Wazuh Version**: 4.14.2 (Docker deployment)
- **Neo4j**: Requires Neo4j database running
- **Data Collection**: Creates timestamped folders for each collection run
- **Graph Building**: Supports incremental updates (MERGE operations)
