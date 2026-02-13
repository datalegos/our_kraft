# Folder Structure

## Scripts (Run These)
- `scripts/main.py` - Collect data from Wazuh
- `scripts/extract_nodes.py` - Extract nodes from data
- `scripts/build_graph.py` - Build Neo4j graph

## Code (Used by Scripts)
- `graph_builder/` - Graph building functions
- `utils/` - Helper functions

## Workflow
1. `python scripts/main.py` - Collect data
2. `python scripts/extract_nodes.py asset` - Extract nodes
3. `python scripts/build_graph.py` - Build graph

## Config
All settings in `config/` folder.

## Data
- `collected_data/` - Raw Wazuh data
- `extracted_data/` - Extracted nodes
- `logs/` - Log files
