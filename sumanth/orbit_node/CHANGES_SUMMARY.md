# Changes Summary

## 1. Updated All Relationships to Use Source-Based Matching

### What Changed
All relationships in `config/graph_config.yaml` now use `from_source` configuration instead of `from_property` for efficiency.

### Relationships Updated
1. **asset_to_host**: Uses `All_Agents.json` → matches `id` with `Host.agent_id`
2. **host_to_software**: Uses `Syscollector_OS_Info_{agent_id}.json` → matches `agent_id`
3. **asset_to_software**: Uses `All_Agents.json` → matches `id` with `Software.agent_id`
4. **software_to_vulnerability**: Uses `Syscollector_Packages_{agent_id}.json` → multi-field matching (agent_id, name, version)
5. **vulnerability_to_asset**: Uses `Vulnerabilities_{agent_id}.json` → matches `agent_id` with `Asset.asset_id`
6. **asset_to_assetgroup**: Uses `All_Agents.json` → matches `group` array with `AssetGroup.name`
7. **asset_to_hardware**: Uses `All_Agents.json` → matches `id` with `Hardware.agent_id`

### Benefits
- **No Redundant Data**: Matching fields don't need to be stored in nodes
- **Always Fresh**: Relationships use data directly from source files
- **Efficient**: Smaller nodes, faster queries
- **Flexible**: Easy to change relationship logic without re-extracting nodes

## 2. Integrated Source Data Loader into Relationship Manager

### Why Integration?
**Original Reasoning (Separate File)**:
- Separation of concerns
- Reusability
- Testability

**Why Integrated Now**:
- **Tighter Coupling**: Source data loading is ONLY used by relationship manager
- **Simpler Architecture**: Fewer files, clearer code flow
- **Better Performance**: No unnecessary abstraction
- **Easier Debugging**: All relationship logic in one place

### Implementation
- Source data loading methods are now private methods in `RelationshipManager`:
  - `_load_source_data()`: Loads data from source files
  - `_get_nested_value()`: Extracts nested field values
  - `_load_paths_config()`: Loads paths configuration
  - `_get_base_directory()`: Gets base data directory

## 3. Reorganized Folder Structure

### New Structure
```
orbit_node/
├── config/          # All configuration files
├── scripts/         # All executable scripts
├── utils/           # Utility modules
├── graph_builder/   # Neo4j graph building module
├── collected_data/  # Raw collected data
├── extracted_data/  # Processed node data
└── logs/            # Application logs
```

### Files Moved
- **Config Files** → `config/`:
  - `config.yaml`
  - `graph_config.yaml`
  - `neo4j_config.yaml`
  - `paths_config.yaml`

- **Scripts** → `scripts/`:
  - `main.py`
  - `build_graph.py`
  - `extract_nodes.py`
  - `extract_asset_nodes.py`
  - `test_neo4j_connection.py`
  - `debug_neo4j_auth.py`
  - `fix_neo4j_auth.py`

- **Utilities** → `utils/`:
  - `csv_converter.py`
  - `data_collector.py`
  - `wazuh_manager_client.py`
  - `wazuh_indexer_client.py`
  - `vulnerability_detector.py`

### Benefits
- **Professional Structure**: Follows Python project best practices
- **Easy Navigation**: Related files grouped together
- **Scalable**: Easy to add new files
- **Clear Separation**: Config, scripts, utilities clearly separated

## 4. Updated Import Paths

All modules now correctly reference:
- Config files in `config/` directory
- Utility modules in `utils/` directory
- Graph builder modules in `graph_builder/` directory

Path resolution handles both absolute and relative paths, with fallback to `config/` directory.

## 5. Enhanced Relationship Manager

### New Features
- **Source-Based Matching**: Loads data directly from source files
- **Multi-Field Matching**: Supports complex relationships (e.g., Software → Vulnerability)
- **Array Matching**: Handles array fields (e.g., Asset.group)
- **Flexible Configuration**: All matching logic configurable via YAML

### Methods Added
- `create_relationships_from_source()`: Single-field source matching
- `create_relationships_from_source_multi_field()`: Multi-field source matching
- `_load_source_data()`: Integrated source data loading
- `_get_nested_value()`: Nested field extraction
- `_load_paths_config()`: Paths configuration loading

## Configuration Pattern

### Single-Field Matching
```yaml
match_criteria:
  from_source:
    source_field: "field_name"
    source_path: "data.affected_items[]"
    source_file: "File_Name.json"
    source_directory: "directory_key"
    from_node_key_property: "asset_id"
    source_key_field: "id"  # Field in source that matches from_node_key_property
  to_property: "target_property"
```

### Multi-Field Matching
```yaml
match_criteria:
  from_source:
    source_fields: ["agent_id", "name", "version"]
    source_path: "data.affected_items[]"
    source_file: "File_Name_{agent_id}.json"
    source_directory: "directory_key"
    from_node_key_properties: ["agent_id", "name", "version"]
  to_properties: ["agent_id", "package_name", "package_version"]
```

### Array Matching
```yaml
match_criteria:
  from_source:
    source_field: "group"
    array_match: true  # Creates relationship for each value in array
    ...
```

## Testing

To test the changes:
```bash
# From project root
python scripts/build_graph.py
```

Check logs for:
- "Loaded graph configuration from config/graph_config.yaml"
- "Prepared X relationships to create from source data"
- "Created X LOCATED_IN relationships" (for asset_to_assetgroup)

## Documentation

- **ARCHITECTURE.md**: Detailed architecture and design decisions
- **FOLDER_STRUCTURE.md**: Folder structure documentation
- **README.md**: Updated project documentation

