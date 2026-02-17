# Architecture and Design Decisions

## Why Source Data Loader Was Integrated into Relationship Manager

### Original Approach (Separate File)
Initially, `source_data_loader.py` was created as a separate module for:
- **Separation of Concerns**: Data loading logic separated from relationship creation
- **Reusability**: Could be used by other modules
- **Testability**: Easier to test in isolation

### Current Approach (Integrated)
The source data loading functionality has been **integrated directly into `relationship_manager.py`** because:

1. **Tighter Coupling**: Source data loading is ONLY used by relationship manager - no other modules need it
2. **Simpler Architecture**: Fewer files to maintain, clearer code flow
3. **Better Performance**: No unnecessary abstraction layer
4. **Easier Debugging**: All relationship logic in one place

### Implementation Details
- Source data loading methods are now private methods (`_load_source_data`, `_get_nested_value`) within `RelationshipManager`
- Configuration loading (`_load_paths_config`, `_get_base_directory`) is also integrated
- This reduces the module count while maintaining clean code organization

## Folder Structure

### Current Structure
```
orbit_node/
├── config/              # Configuration files
│   ├── config.yaml
│   ├── graph_config.yaml
│   ├── neo4j_config.yaml
│   └── paths_config.yaml
├── scripts/             # Executable scripts
│   ├── main.py
│   ├── build_graph.py
│   └── extract_nodes.py
├── utils/               # Utility modules
│   ├── csv_converter.py
│   ├── data_collector.py
│   ├── wazuh_manager_client.py
│   └── wazuh_indexer_client.py
├── graph_builder/       # Neo4j graph building module
│   ├── __init__.py
│   ├── neo4j_connection.py
│   ├── constraints_manager.py
│   ├── node_inserter.py
│   ├── relationship_manager.py  # Now includes source data loading
│   └── graph_builder.py
├── collected_data/      # Raw collected data (timestamped folders)
├── extracted_data/      # Processed/extracted node data
├── logs/                # Application logs
└── requirements.txt
```

### Benefits of This Structure
1. **Clear Separation**: Config, scripts, utilities, and modules are clearly separated
2. **Easy Navigation**: Related files are grouped together
3. **Scalability**: Easy to add new utilities or scripts
4. **Professional**: Follows standard Python project structure

## Source-Based Relationship Matching

### Why Source-Based Matching?
All relationships now use **source-based matching** instead of property-based matching for:

1. **Efficiency**: No redundant data stored in nodes
   - Example: `group` array doesn't need to be stored in Asset nodes
   - Relationships are created directly from source files

2. **Data Integrity**: Always uses fresh data from source files
   - No risk of stale data in node properties
   - Source of truth is the collected data files

3. **Flexibility**: Easy to change relationship logic without re-extracting nodes
   - Just update `graph_config.yaml`
   - No need to re-run node extraction

### Configuration Pattern
```yaml
relationship_name:
  match_criteria:
    from_source:
      source_field: "field_name"        # Field to extract
      source_path: "data.affected_items[]"  # Path to data
      source_file: "File_Name.json"     # Source file
      source_directory: "directory_key" # Directory from paths_config
      from_node_key_property: "asset_id"  # Property to match from_node
      array_match: false                 # For array fields
    to_property: "target_property"      # Property to match to_node
```

### Multi-Field Matching
For complex relationships (e.g., Software → Vulnerability):
```yaml
from_source:
  source_fields: ["agent_id", "name", "version"]  # Multiple fields
  from_node_key_properties: ["agent_id", "name", "version"]
to_properties: ["agent_id", "package_name", "package_version"]
```

## Graph Schema Efficiency

### Improvements Made
1. **Removed Redundant Properties**: 
   - `group` array removed from Asset nodes (now only in source)
   - Matching IDs only stored where necessary

2. **Source-Based Relationships**: 
   - All relationships fetch data directly from source files
   - No intermediate storage of matching data

3. **Configurable Everything**:
   - Node properties, constraints, relationships all configurable
   - Easy to add new node types or relationships

### Performance Benefits
- **Reduced Node Size**: Nodes contain only essential properties
- **Faster Queries**: Smaller nodes = faster graph traversal
- **Lower Memory**: Less data stored in Neo4j
- **Easier Updates**: Change relationships without re-inserting nodes



## Data Aggregation Layer

### Purpose
The aggregation layer processes extracted node data to generate statistical summaries and counts. This serves as an intermediate layer between the detailed graph (Neo4j) and the core analytics graph.

### Pipeline Flow
```
Day 0: Data Collection → collected_data/
Day 1: Data Staging → extracted_data/
Day 2: Graph Building → Neo4j (detailed graph)
Day 3: Data Aggregation → aggregated_data/ ← NEW
Day 4: PII/PCI Detection → filtered_data/ (future)
Day 5: Core Graph → Neo4j (core database) (future)
```

### Aggregation Module (`scripts/aggregate_data.py`)

**Key Features:**
1. **Automatic Session Detection**: Uses `.current_session` file or latest folder
2. **Multiple Aggregation Types**: Hosts, Software, Vulnerabilities, Hardware, Assets, Groups
3. **Statistical Analysis**: Counts, distributions, top N items
4. **Multiple Output Formats**: Complete JSON, individual files, human-readable summary

**Aggregations Generated:**

1. **Host/OS Aggregation**
   - OS distribution (Linux: 1, Windows: 3, macOS: 1)
   - Platform distribution
   - Architecture distribution
   - OS version distribution

2. **Software Aggregation**
   - Total and unique package counts
   - Top 50 most common packages
   - Vendor distribution
   - Package format distribution
   - Multi-version package tracking

3. **Vulnerability Aggregation**
   - Total and unique CVE counts
   - Severity distribution (Critical, High, Medium, Low)
   - Status distribution
   - Top 20 most common CVEs

4. **Hardware Aggregation**
   - CPU distribution
   - RAM statistics (total, avg, min, max)

5. **Asset Aggregation**
   - Total asset count
   - Status distribution

6. **Asset Group Aggregation**
   - Total group count
   - Group names list

### Configuration

**`config/aggregation_config.yaml`**
- Enable/disable specific aggregations
- Configure top N limits
- Set output formats
- Future: PII detection settings
- Future: Core graph settings

### Output Structure

```
aggregated_data/
├── .current_session                    # Points to latest session
└── {timestamp}/                        # Timestamped session
    ├── complete_aggregation.json       # All aggregations combined
    ├── summary_report.txt              # Human-readable summary
    ├── host_aggregation.json           # Individual aggregations
    ├── software_aggregation.json
    ├── vulnerability_aggregation.json
    ├── hardware_aggregation.json
    ├── asset_aggregation.json
    └── assetgroup_aggregation.json
```

### Usage

```bash
# Run aggregation
python scripts/aggregate_data.py

# Complete pipeline
python scripts/main.py                  # Collect data
python scripts/extract_nodes.py         # Extract nodes
python scripts/build_graph.py           # Build detailed graph
python scripts/aggregate_data.py        # Aggregate data
```

### Benefits

1. **Performance**: Core graph queries faster with aggregated data
2. **Privacy**: PII detection happens before core graph
3. **Analytics**: Easy dashboard generation from aggregated counts
4. **Scalability**: Aggregated data much smaller than detailed data
5. **Flexibility**: Can re-aggregate without re-collecting from Wazuh

### Next Steps

**Phase 4: PII/PCI Detection (Presidio)**
- Scan aggregated data for sensitive information
- Redact or flag PII/PCI data
- Store filtered data for core graph

**Phase 5: Core Graph Building**
- Create high-level analytics graph
- Store aggregated counts as nodes
- Connect to detailed graph for drill-down
- Use separate Neo4j database ("core")
