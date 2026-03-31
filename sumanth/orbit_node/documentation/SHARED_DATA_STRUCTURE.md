# NJS Shared Data Structure

## Overview

All data generated and used by the NJS Pipeline is stored in a single `njs_shared_data` directory located at the same level as the project folder (sibling directory). This centralized approach ensures clean separation between code and data, making backups, monitoring, and data management easier.

## Directory Location

```
parent_directory/
├── orbit_node/          # Project code
└── njs_shared_data/     # All pipeline data (created by startup scripts)
```

## Complete Directory Structure

```
njs_shared_data/
├── config/              # Runtime configuration (optional, copied from project)
│   ├── aggregation_config.yaml
│   ├── paths_config.yaml
│   ├── neo4j_config.yaml
│   └── graph_config.yaml
│
├── data/                # All data files
│   ├── collected/       # Raw data from Wazuh API
│   │   ├── 20260210_201344/
│   │   │   ├── agents_manager/
│   │   │   │   └── All_Agents.json
│   │   │   ├── host/
│   │   │   │   ├── agent_000/
│   │   │   │   │   └── Syscollector_OS_Info_000.json
│   │   │   │   └── ...
│   │   │   ├── packages/
│   │   │   ├── hardware/
│   │   │   ├── fim/
│   │   │   ├── groups/
│   │   │   └── vulnerabilities/
│   │   ├── 20260211_143022/
│   │   └── .done         # Marker: collection complete
│   │
│   ├── extracted/       # Normalized data for Node Graph
│   │   ├── 20260210_201344/
│   │   │   ├── agents.json
│   │   │   ├── hosts.json
│   │   │   ├── packages.json
│   │   │   ├── hardware.json
│   │   │   ├── vulnerabilities.json
│   │   │   └── relationships.json
│   │   ├── 20260211_143022/
│   │   └── .done         # Marker: extraction complete
│   │
│   ├── aggregated/      # Intermediate aggregations (optional)
│   │   ├── 20260210_201344/
│   │   │   ├── asset_aggregation.json
│   │   │   ├── software_aggregation.json
│   │   │   └── vulnerability_aggregation.json
│   │   └── ...
│   │
│   ├── aggregated_core/ # Final aggregations for Core Graph
│   │   ├── 20260210_201344/
│   │   │   ├── core_aggregation.json
│   │   │   ├── exposure_surface.json
│   │   │   ├── sensitivity_surface.json
│   │   │   ├── outcome_metrics.json
│   │   │   └── summary_report.txt
│   │   ├── 20260211_143022/
│   │   └── .done         # Marker: aggregation complete
│   │
│   └── pii_scan_results/ # PII/PCI detection results
│       ├── 20260210_201344/
│       │   ├── pii_scan_results.json
│       │   ├── pii_scan_summary.txt
│       │   └── pii_findings_detail.json
│       ├── 20260211_143022/
│       └── .done         # Marker: PII scan complete
│
├── logs/                # All pipeline logs
│   ├── pipeline.log     # Main orchestrator log
│   ├── collect_data.log
│   ├── extract_data.log
│   ├── build_node_graph.log
│   ├── aggregate_data.log
│   ├── detect_pii.log
│   └── build_core_graph.log
│
└── pipeline/            # Pipeline state markers
    ├── node_graph.done  # Node Graph build complete
    ├── core_graph.done  # Core Graph build complete
    └── .done            # Full pipeline complete
```

## Data Flow

### 1. Collected Data (`data/collected/`)
- **Source**: Wazuh API
- **Format**: Raw JSON responses from Wazuh endpoints
- **Purpose**: Original data as received from Wazuh
- **Created by**: `scripts/main.py` (collect_data step)
- **Used by**: `scripts/extract_data.py`

### 2. Extracted Data (`data/extracted/`)
- **Source**: Processed from collected data
- **Format**: Normalized JSON files
- **Purpose**: Clean, structured data ready for Node Graph
- **Created by**: `scripts/extract_data.py`
- **Used by**: `scripts/build_node_graph.py`

### 3. Aggregated Data (`data/aggregated/`)
- **Source**: Intermediate aggregations (optional)
- **Format**: JSON aggregation files
- **Purpose**: Intermediate privacy-preserving aggregations
- **Created by**: `scripts/aggregate_data_v2.py`
- **Used by**: Internal processing

### 4. Aggregated Core Data (`data/aggregated_core/`)
- **Source**: Final aggregations from Node Graph
- **Format**: JSON files with privacy-preserving aggregations
- **Purpose**: Data to be sent to Core Graph (bank-level)
- **Created by**: `scripts/aggregate_data_v2.py`
- **Used by**: `scripts/build_core_graph.py`
- **Important**: This is the FINAL OUTPUT that goes to Core Graph

### 5. PII Scan Results (`data/pii_scan_results/`)
- **Source**: PII/PCI detection on aggregated_core data
- **Format**: JSON scan results
- **Purpose**: Verify no sensitive data before sending to Core
- **Created by**: `scripts/detect_pii.py`
- **Used by**: Validation gate before Core Graph build

## Configuration

### Path Configuration (`config/paths_config.yaml`)

All scripts read paths from `config/paths_config.yaml`:

```yaml
paths:
  # Shared data root (from environment variable)
  shared_data_root: "${SHARED_DATA_PATH}"
  
  # Base directory for collected data
  base_directory: "data/collected/20260210_201344"
  
  # Use latest collected data automatically
  use_latest: true
  
  # Output directories
  output_directory: "data/extracted"
  aggregated_directory: "data/aggregated"
  aggregated_core_directory: "data/aggregated_core"
  pii_scan_results_directory: "data/pii_scan_results"
  log_directory: "logs"
  pipeline_directory: "pipeline"
```

### Environment Variables

- `SHARED_DATA_PATH`: Path to shared data directory
  - In Docker: `/shared_data`
  - On host: `../njs_shared_data`

## Docker Volume Mounts

```yaml
volumes:
  # Shared data - all pipeline data
  - ../njs_shared_data:/shared_data
  
  # Config files (read-only from project)
  - ../config:/shared_data/config:ro
```

## Startup Scripts

The deployment scripts automatically create the directory structure:

```bash
# Created by start_all.sh and Makefile setup
njs_shared_data/
├── data/
│   ├── collected/
│   ├── extracted/
│   ├── aggregated/
│   ├── aggregated_core/
│   └── pii_scan_results/
├── logs/
├── pipeline/
└── config/
```

## Data Lifecycle

### Collection Session
Each pipeline run creates a timestamped session:

```
20260210_201344/  # Format: YYYYMMDD_HHMMSS
```

### Data Retention

- **Collected data**: Kept for 7 days (configurable)
- **Extracted data**: Kept for 7 days (configurable)
- **Aggregated core**: Kept for 30 days (configurable)
- **PII scan results**: Kept for 30 days (configurable)
- **Logs**: Rotated daily, kept for 7 days

### Cleanup

```bash
# Manual cleanup
make clean

# Automatic cleanup (in scripts)
find njs_shared_data/data/collected/ -type d -mtime +7 -exec rm -rf {} +
find njs_shared_data/data/pii_scan_results/ -type d -mtime +30 -exec rm -rf {} +
```

## Accessing Data

### From Host

```bash
# View latest collected data
ls -la ../njs_shared_data/data/collected/

# View logs
tail -f ../njs_shared_data/logs/pipeline.log

# Check pipeline status
cat ../njs_shared_data/pipeline/.done
```

### From Docker Container

```bash
# Access container
docker exec -it njs-pipeline /bin/bash

# View data
ls -la /shared_data/data/collected/

# View logs
tail -f /shared_data/logs/pipeline.log
```

### From Scripts

All scripts use `SHARED_DATA_PATH` environment variable:

```python
import os
from pathlib import Path

shared_data = Path(os.getenv('SHARED_DATA_PATH', '/shared_data'))
collected_dir = shared_data / 'data' / 'collected'
logs_dir = shared_data / 'logs'
```

## Backup Strategy

### What to Backup

1. **Critical**: `data/aggregated_core/` - Final output
2. **Important**: `data/collected/` - Raw data (can be re-collected)
3. **Optional**: `data/extracted/` - Can be regenerated
4. **Optional**: `logs/` - For troubleshooting

### Backup Commands

```bash
# Backup aggregated core data
tar -czf njs_backup_$(date +%Y%m%d).tar.gz \
  ../njs_shared_data/data/aggregated_core/

# Backup Neo4j databases
make backup

# Full backup
tar -czf njs_full_backup_$(date +%Y%m%d).tar.gz \
  ../njs_shared_data/
```

## Monitoring

### Disk Usage

```bash
# Check shared data size
du -sh ../njs_shared_data/

# Check by directory
du -sh ../njs_shared_data/data/*/
```

### Pipeline Status

```bash
# Check if pipeline completed
make status

# View latest logs
make logs

# Check for errors
grep ERROR ../njs_shared_data/logs/pipeline.log
```

## Troubleshooting

### Directory Not Found

```bash
# Recreate structure
make setup

# Or manually
mkdir -p ../njs_shared_data/{data/{collected,extracted,aggregated,aggregated_core,pii_scan_results},logs,pipeline,config}
```

### Permission Issues

```bash
# Fix permissions
chmod -R 755 ../njs_shared_data/
```

### Disk Space Issues

```bash
# Clean old data
make clean

# Check disk usage
df -h ../njs_shared_data/
```

## Best Practices

1. **Always use config paths**: Never hardcode paths in scripts
2. **Check .done files**: Validate prerequisites before running steps
3. **Monitor disk space**: Set up alerts for disk usage
4. **Regular backups**: Backup aggregated_core data regularly
5. **Log rotation**: Implement log rotation for long-running systems
6. **Data retention**: Clean old data based on retention policy

## Summary

The `njs_shared_data` directory provides:

- ✅ Centralized data storage
- ✅ Clean separation from code
- ✅ Easy backup and monitoring
- ✅ Consistent path structure
- ✅ Docker-friendly volume mounts
- ✅ Clear data lifecycle
- ✅ Validation gates with .done files

All scripts automatically use this structure through the `SHARED_DATA_PATH` environment variable and `config/paths_config.yaml`.
