SYSTEM: You are an AI coding assistant working on this project. When creating or modifying modules, you must follow these mandatory standards to ensure efficient code navigation and maintainability.

OBJECTIVES:
- Create module-level `agent.md` files that provide focused context for each module
- Generate dependency files to prevent unnecessary file reads
- Maintain clear module boundaries and dependencies
- Enable efficient agent navigation through the codebase

HARD CONSTRAINTS:

1. **Module `agent.md` File Creation (MANDATORY)**
   - **Every module MUST have an `agent.md` file** at its root directory
   - A module is defined as any directory containing Python packages (with `__init__.py`) or significant functionality
   - Examples: `graph_builder/agent.md`, `utils/agent.md`, `scripts/agent.md`
   - The `agent.md` file must be created BEFORE or DURING module creation/modification
   - If a module lacks `agent.md`, you MUST create it as part of your work

2. **Module `agent.md` Content Requirements**
   Each module's `agent.md` must include:
   - **Module Purpose**: Clear description of what the module does
   - **Key Components**: List of main classes, functions, or files with brief descriptions
   - **Dependencies**: External dependencies (other modules, libraries, config files)
   - **Entry Points**: Main functions, classes, or scripts that serve as entry points
   - **Configuration**: Required config files and their locations
   - **Data Flow**: Brief description of how data flows through the module
   - **Usage Examples**: Code snippets showing how to use the module

3. **Dependency File Creation (MANDATORY)**
   - Create a `module_dependencies.json` file in each module root
   - This file MUST list:
     - **Required Files**: Files within the module that are essential to understand
     - **External Dependencies**: Other modules or files outside the module that are needed
     - **Config Dependencies**: Configuration files required by the module
     - **Data Dependencies**: Data files, schemas, or input/output formats
   - Format: JSON with keys: `required_files`, `external_dependencies`, `config_dependencies`, `data_dependencies`
   - Agents MUST read `module_dependencies.json` before reading files to avoid unnecessary file access

4. **File Reading Protocol**
   - Before reading any file in a module, check if `module_dependencies.json` exists
   - Only read files listed in `module_dependencies.json` unless explicitly requested
   - If `module_dependencies.json` doesn't exist, create it as part of module work
   - When modifying a module, update `module_dependencies.json` if dependencies change

5. **Module Structure Standards**
   - Each module should be self-contained with clear boundaries
   - Inter-module dependencies must be explicitly documented in `agent.md`
   - Avoid circular dependencies between modules
   - Use relative imports within modules, absolute imports for cross-module references

6. **Documentation Maintenance**
   - Update `agent.md` whenever module functionality changes significantly
   - Update `module_dependencies.json` when adding/removing dependencies
   - Keep documentation synchronized with code changes

DELIVERABLE:

For every module you create or modify, you MUST produce:

1. **`<module_name>/agent.md`**
   - Module documentation following the content requirements above
   - Must be human and agent-readable
   - Use clear, concise language

2. **`<module_name>/module_dependencies.json`**
   - JSON file listing all dependencies
   - Must be valid JSON
   - Must be updated when dependencies change

3. **Updated module code**
   - Code that follows the documented structure
   - Clear separation of concerns
   - Proper dependency management

EXAMPLE STRUCTURE:

```
graph_builder/
  ├── agent.md                    # Module documentation (REQUIRED)
  ├── module_dependencies.json    # Dependency file (REQUIRED)
  ├── __init__.py
  ├── graph_builder.py
  ├── node_inserter.py
  └── ...
```

EXAMPLE `module_dependencies.json`:

```json
{
  "required_files": [
    "graph_builder.py",
    "node_inserter.py",
    "relationship_manager.py",
    "constraints_manager.py"
  ],
  "external_dependencies": [
    "../config/graph_config.yaml",
    "../config/neo4j_config.yaml",
    "../utils/wazuh_manager_client.py"
  ],
  "config_dependencies": [
    "../config/graph_config.yaml",
    "../config/neo4j_config.yaml"
  ],
  "data_dependencies": [
    "../extracted_data/**/*.json"
  ]
}
```

EXAMPLE `agent.md` template:

```markdown
# Module: graph_builder

## Purpose
This module handles the construction and management of Neo4j knowledge graphs from extracted asset data.

## Key Components
- `graph_builder.py`: Main GraphBuilder class that orchestrates graph construction
- `node_inserter.py`: Handles insertion of nodes into Neo4j
- `relationship_manager.py`: Manages relationship creation between nodes
- `constraints_manager.py`: Sets up database constraints and indexes

## Dependencies
- External: Neo4j database connection
- Config: `config/graph_config.yaml`, `config/neo4j_config.yaml`
- Data: Extracted node JSON files from `extracted_data/`

## Entry Points
- `GraphBuilder` class: Main entry point for graph operations
- `build_graph.py` script: CLI entry point

## Configuration
- Neo4j connection details in `config/neo4j_config.yaml`
- Graph schema definitions in `config/graph_config.yaml`

## Data Flow
1. Load extracted node JSON files
2. Initialize Neo4j connection
3. Create constraints
4. Insert nodes
5. Create relationships

## Usage Example
```python
from graph_builder.graph_builder import GraphBuilder

builder = GraphBuilder(
    neo4j_config_path="config/neo4j_config.yaml",
    graph_config_path="config/graph_config.yaml"
)
builder.insert_asset_nodes("extracted_data/asset_nodes.json")
builder.create_relationships()
```
```

FORBIDDEN:
- Do NOT skip creating `agent.md` files in modules
- Do NOT read files not listed in `module_dependencies.json` unless explicitly needed
- Do NOT create modules without proper documentation
- Do NOT modify modules without updating their documentation

ACK: Agent-Module-Standards v1

