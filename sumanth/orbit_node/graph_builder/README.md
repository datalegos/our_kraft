# Graph Builder Module

This module handles Neo4j graph database operations including constraint creation, index management, and node insertion.

## Module Structure

```
graph_builder/
├── __init__.py              # Module initialization
├── neo4j_connection.py      # Neo4j connection handler
├── constraints_manager.py   # Constraint and index management
├── node_inserter.py        # Node insertion operations
└── graph_builder.py        # Main orchestrator class
```

## Components

### 1. Neo4jConnection (`neo4j_connection.py`)
- Manages connection to Neo4j database
- Loads configuration from `neo4j_config.yaml`
- Provides session management and query execution
- Handles connection pooling and timeouts

### 2. ConstraintsManager (`constraints_manager.py`)
- Creates UNIQUE constraints and indexes based on `graph_config.yaml`
- Supports configurable constraints per node type
- Handles constraint existence checks
- Lists existing constraints and indexes

### 3. NodeInserter (`node_inserter.py`)
- Inserts nodes into Neo4j with property mapping
- Supports MERGE operations to avoid duplicates
- Handles batch insertion for performance
- Converts datetime objects to ISO format strings
- Loads nodes from JSON files

### 4. GraphBuilder (`graph_builder.py`)
- Main orchestrator class
- Coordinates constraint setup and node insertion
- Provides high-level API for graph building operations

## Usage

See `build_graph.py` in the root directory for example usage.

## Configuration Files

- `neo4j_config.yaml`: Neo4j connection settings
- `graph_config.yaml`: Node definitions, properties, and constraints
- `paths_config.yaml`: File paths for data sources

