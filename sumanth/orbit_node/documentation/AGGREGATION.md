# Data Aggregation Module

## Overview

The aggregation module processes extracted node data to generate statistical summaries and counts. This aggregated data serves as an intermediate layer between the detailed graph (Neo4j) and the core analytics graph.

## Pipeline Position

```
Day 0: Data Collection (Wazuh) → collected_data/
Day 1: Data Staging → extracted_data/
Day 2: Graph Building → Neo4j (detailed graph)
Day 3: Data Aggregation → aggregated_data/ ← YOU ARE HERE
Day 4: PII/PCI Detection (Presidio) → filtered_data/
Day 5: Core Graph Building → Neo4j (core database)
```

## What Gets Aggregated

### 1. Host/OS Information
- OS distribution (Linux: 5, Windows: 4, macOS: 1)
- Platform distribution
- Architecture distribution (x86_64, arm64)
- OS version distribution

### 2. Software Packages
- Total package count
- Unique package count
- Top 50 most common packages
- Vendor distribution
- Package format distribution (rpm, deb, pkg)
- Multi-version packages (packages installed with different versions)

### 3. Vulnerabilities
- Total vulnerability count
- Unique CVE count
- Severity distribution (Critical, High, Medium, Low)
- Status distribution
- Top 20 most common CVEs

### 4. Hardware
- CPU distribution
- RAM statistics (total, average, min, max)

### 5. Assets
- Total asset count
- Status distribution

### 6. Asset Groups
- Total group count
- Group names list

## Usage

### Run Aggregation

```bash
# From project root
python scripts/aggregate_data.py
```

The script will:
1. Find the latest extraction session in `extracted_data/`
2. Load all node files
3. Generate aggregations
4. Save results to `aggregated_data/{timestamp}/`

### Output Structure

```
aggregated_data/
├── .current_session                    # Points to latest session
└── 20260216_143022/                    # Timestamped session
    ├── complete_aggregation.json       # All aggregations combined
    ├── summary_report.txt              # Human-readable summary
    ├── host_aggregation.json           # Individual aggregations
    ├── software_aggregation.json
    ├── vulnerability_aggregation.json
    ├── hardware_aggregation.json
    ├── asset_aggregation.json
    └── assetgroup_aggregation.json
```

### Example Output

```json
{
  "timestamp": "2026-02-16T14:30:22.123456",
  "source_path": "extracted_data/20260215_161934_530",
  "host": {
    "total_hosts": 5,
    "os_distribution": {
      "Linux": 1,
      "Windows": 3,
      "macOS": 1
    },
    "architecture_distribution": {
      "x86_64": 4,
      "arm64": 1
    }
  },
  "software": {
    "total_packages": 1247,
    "unique_packages": 856,
    "top_50_packages": {
      "python3-libs": 3,
      "openssl": 5,
      "curl": 4
    },
    "multi_version_count": 12
  },
  "vulnerability": {
    "total_vulnerabilities": 45,
    "unique_cves": 32,
    "severity_distribution": {
      "Critical": 5,
      "High": 12,
      "Medium": 18,
      "Low": 10
    }
  }
}
```

## Configuration

Edit `config/aggregation_config.yaml` to customize:

- Which aggregations to run
- Top N limits (e.g., top 50 packages)
- Output format (JSON/CSV)
- Future: PII detection settings
- Future: Core graph settings

## Next Steps

### Phase 4: PII/PCI Detection (Presidio)
- Scan aggregated data for sensitive information
- Redact or flag PII/PCI data
- Store filtered data for core graph

### Phase 5: Core Graph Building
- Create high-level analytics graph
- Store aggregated counts as nodes
- Connect to detailed graph for drill-down
- Use separate Neo4j database ("core")

## Integration with Existing Pipeline

The aggregation module integrates seamlessly:

1. **Input**: Uses same session tracking as `build_graph.py`
   - Reads from `extracted_data/.current_session`
   - Falls back to latest folder by timestamp

2. **Output**: Creates parallel structure
   - `aggregated_data/` mirrors `extracted_data/` structure
   - Maintains session timestamps for traceability

3. **Configuration**: Follows existing patterns
   - Uses `config/` directory
   - YAML-based configuration
   - Consistent logging format

## Example Workflow

```bash
# Complete pipeline
python scripts/main.py                  # Day 0: Collect data
python scripts/extract_nodes.py         # Day 1: Extract nodes
python scripts/build_graph.py           # Day 2: Build detailed graph
python scripts/aggregate_data.py        # Day 3: Aggregate data
# python scripts/detect_pii.py          # Day 4: PII detection (future)
# python scripts/build_core_graph.py    # Day 5: Build core graph (future)
```

## Benefits

1. **Performance**: Core graph queries are faster with aggregated data
2. **Privacy**: PII detection happens before core graph
3. **Analytics**: Easy to generate dashboards from aggregated counts
4. **Scalability**: Aggregated data is much smaller than detailed data
5. **Flexibility**: Can re-aggregate without re-collecting from Wazuh
