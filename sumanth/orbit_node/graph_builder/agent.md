# Module: graph_builder

## Purpose
This module handles the construction and management of Neo4j knowledge graphs from extracted asset data. It provides functionality for creating database constraints, inserting nodes, and establishing relationships between nodes in the Neo4j graph database.

## Key Components
- `graph_builder.py`: Main GraphBuilder class that orchestrates graph construction operations
- `neo4j_connection.py`: Manages Neo4j database connection, session handling, and query execution
- `constraints_manager.py`: Creates and manages UNIQUE constraints and indexes based on graph configuration
- `node_inserter.py`: Handles insertion of nodes into Neo4j with property mapping and batch operations
- `relationship_manager.py`: Manages relationship creation between nodes based on graph configuration

## Dependencies
- **External Libraries**: `neo4j` (GraphDatabase driver), `yaml`, `json`
- **Config Files**: 
  - `../config/neo4j_config.yaml`: Neo4j connection settings (host, port, credentials)
  - `../config/graph_config.yaml`: Node definitions, properties, constraints, and relationship configurations
  - `../config/paths_config.yaml`: File paths for data sources (used by relationship manager)
- **Data Sources**: Extracted node JSON files from `../extracted_data/` directory
- **Other Modules**: None (self-contained module)

## Entry Points
- `GraphBuilder` class: Main entry point for graph operations
  - `setup_constraints()`: Create constraints and indexes
  - `insert_asset_nodes()`: Insert asset nodes from JSON
  - `insert_host_nodes()`: Insert host nodes from JSON
  - `insert_nodes()`: Generic node insertion method
  - `create_relationships()`: Create relationships between nodes
- `../scripts/build_graph.py`: CLI script that uses GraphBuilder

## Configuration
- **Neo4j Connection**: Configured via `config/neo4j_config.yaml`
  - Connection URI, authentication credentials
  - Connection pooling and timeout settings
- **Graph Schema**: Defined in `config/graph_config.yaml`
  - Node type definitions with properties
  - Constraint definitions (UNIQUE keys)
  - Relationship definitions with matching rules

## Data Flow
1. **Initialization**: Load Neo4j config and graph config files
2. **Connection**: Establish Neo4j database connection via Neo4jConnection
3. **Constraints Setup**: Create UNIQUE constraints and indexes for node types
4. **Node Insertion**: Load nodes from JSON files and insert into Neo4j
   - Supports MERGE operations to avoid duplicates
   - Handles datetime conversion to ISO format
   - Batch insertion for performance
5. **Relationship Creation**: Create relationships between nodes based on:
   - Property-based matching (e.g., agent_id, name)
   - Source file-based matching
   - Configuration-defined relationship rules

## Usage Example
```python
from graph_builder.graph_builder import GraphBuilder

# Initialize graph builder
builder = GraphBuilder(
    neo4j_config_path="config/neo4j_config.yaml",
    graph_config_path="config/graph_config.yaml"
)

# Setup constraints
builder.setup_constraints()

# Insert nodes
builder.insert_asset_nodes("extracted_data/asset_nodes.json")
builder.insert_host_nodes("extracted_data/host_nodes.json")

# Create relationships
builder.create_relationships()

# Close connection
builder.close()
```

## Notes
- All database operations use parameterized queries for security
- The module supports idempotent operations (MERGE) to handle re-runs safely
- Relationship creation supports both property matching and file-based source matching
- Error handling includes logging and graceful failure modes

