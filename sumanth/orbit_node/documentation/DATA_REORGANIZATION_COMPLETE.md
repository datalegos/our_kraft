# Data Reorganization Complete - NJS Shared Data Structure

## ✅ What Was Done

Consolidated all pipeline data into a single `njs_shared_data` directory at the sibling level of the project folder, with a clean, organized structure that separates code from data.

## 🎯 Key Changes

### 1. New Directory Structure

**Before** (scattered in project root):
```
orbit_node/
├── collected_data/
├── extracted_data/
├── aggregated_data/
├── aggregated_data_core/
├── pii_scan_results/
└── logs/
```

**After** (consolidated sibling directory):
```
parent_directory/
├── orbit_node/          # Project code only
└── njs_shared_data/     # All data
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

### 2. Data Organization

All data is now organized under `njs_shared_data/data/`:

- **`collected/`** - Raw data from Wazuh API (what you get initially)
- **`extracted/`** - Normalized data extracted from collected data (for Node Graph)
- **`aggregated/`** - Intermediate aggregations (optional)
- **`aggregated_core/`** - Final aggregations to send to Core Graph
- **`pii_scan_results/`** - PII/PCI detection results (validation before Core)

### 3. Updated Files

#### Configuration Files
- ✅ `config/paths_config.yaml` - Updated all paths to use new structure
  - Added `shared_data_root` variable
  - All paths now relative to shared data root
  - Added comprehensive comments explaining structure

#### Docker Files
- ✅ `docker/docker-compose.yml` - Updated volume mounts
  - Changed from `orbit_node_shared_data` to `njs_shared_data`
  - Added config mount to shared data
  - Added graph_builder and utils mounts

- ✅ `docker/entrypoint.sh` - Updated directory creation
  - Creates new directory structure
  - Uses `data/` subdirectories

#### Scripts
- ✅ `scripts/orchestrator.py` - Updated all path references
  - `collected_data` → `data/collected`
  - `extracted_data` → `data/extracted`
  - `aggregated_data_core` → `data/aggregated_core`
  - `pii_scan_results` → `data/pii_scan_results`
  - Updated all validation functions

#### Deployment Scripts
- ✅ `deployment_scripts/start_all.sh` - Updated paths
  - Changed `orbit_node_shared_data` to `njs_shared_data`
  - Creates new directory structure

#### Build Files
- ✅ `Makefile` - Updated all commands
  - `setup` creates new structure
  - `clean` uses new paths
  - `status` checks new paths

#### Documentation
- ✅ `README.md` - Updated project structure diagram
- ✅ `documentation/SHARED_DATA_STRUCTURE.md` - Comprehensive guide (NEW)
- ✅ `documentation/DATA_REORGANIZATION_COMPLETE.md` - This file (NEW)

## 📊 Data Flow

```
1. COLLECT (Wazuh API)
   ↓
   data/collected/YYYYMMDD_HHMMSS/
   - agents_manager/All_Agents.json
   - host/agent_XXX/Syscollector_OS_Info_XXX.json
   - packages/agent_XXX/Syscollector_Packages_XXX.json
   - hardware/agent_XXX/Syscollector_Hardware_XXX.json
   - vulnerabilities/agent_XXX/Vulnerabilities_XXX.json
   
2. EXTRACT (Normalize)
   ↓
   data/extracted/YYYYMMDD_HHMMSS/
   - agents.json
   - hosts.json
   - packages.json
   - hardware.json
   - vulnerabilities.json
   - relationships.json
   
3. BUILD NODE GRAPH (Neo4j)
   ↓
   Neo4j node_kg database
   
4. AGGREGATE (Privacy-preserving)
   ↓
   data/aggregated_core/YYYYMMDD_HHMMSS/
   - core_aggregation.json      ← FINAL OUTPUT FOR CORE
   - exposure_surface.json
   - sensitivity_surface.json
   - outcome_metrics.json
   
5. DETECT PII (Validation)
   ↓
   data/pii_scan_results/YYYYMMDD_HHMMSS/
   - pii_scan_results.json
   - pii_scan_summary.txt
   
6. BUILD CORE GRAPH (Neo4j)
   ↓
   Neo4j core database
```

## 🔧 Configuration Usage

All scripts now read from `config/paths_config.yaml`:

```yaml
paths:
  # Shared data root (from environment)
  shared_data_root: "${SHARED_DATA_PATH}"
  
  # Use latest collected data automatically
  use_latest: true
  
  # Output directories (relative to shared_data_root)
  output_directory: "data/extracted"
  aggregated_core_directory: "data/aggregated_core"
  pii_scan_results_directory: "data/pii_scan_results"
  log_directory: "logs"
  pipeline_directory: "pipeline"
```

### Environment Variables

- `SHARED_DATA_PATH` - Path to shared data directory
  - Docker: `/shared_data`
  - Host: `../njs_shared_data`

## 🚀 Usage

### First Time Setup

```bash
# Setup creates njs_shared_data directory
make setup

# Build and start
make build
make start
```

### Directory Structure Created

```bash
../njs_shared_data/
├── config/              # Runtime config (optional)
├── data/
│   ├── collected/       # Created by collect_data step
│   ├── extracted/       # Created by extract_data step
│   ├── aggregated/      # Created by aggregate_data step
│   ├── aggregated_core/ # Created by aggregate_data step
│   └── pii_scan_results/# Created by detect_pii step
├── logs/                # Created by orchestrator
└── pipeline/            # Created by orchestrator
```

### Accessing Data

```bash
# View collected data
ls -la ../njs_shared_data/data/collected/

# View final output (for Core Graph)
ls -la ../njs_shared_data/data/aggregated_core/

# View logs
tail -f ../njs_shared_data/logs/pipeline.log

# Check pipeline status
cat ../njs_shared_data/pipeline/.done
```

## 📦 Benefits

### 1. Clean Separation
- ✅ Code in `orbit_node/`
- ✅ Data in `njs_shared_data/`
- ✅ Easy to backup separately
- ✅ Easy to mount different volumes

### 2. Organized Structure
- ✅ All data under `data/` subdirectory
- ✅ Clear purpose for each directory
- ✅ Timestamped sessions
- ✅ Validation markers (.done files)

### 3. Easy Management
- ✅ Single directory to backup
- ✅ Single directory to monitor
- ✅ Single directory to clean
- ✅ Clear data lifecycle

### 4. Docker-Friendly
- ✅ Single volume mount
- ✅ Config accessible in container
- ✅ Logs accessible from host
- ✅ Data persists across container restarts

### 5. Configuration-Driven
- ✅ All paths in config file
- ✅ No hardcoded paths in scripts
- ✅ Easy to change structure
- ✅ Environment variable support

## 🔍 Verification

### Check Structure

```bash
# Verify directory exists
ls -la ../njs_shared_data/

# Verify subdirectories
ls -la ../njs_shared_data/data/

# Verify logs directory
ls -la ../njs_shared_data/logs/
```

### Check Configuration

```bash
# View path config
cat config/paths_config.yaml

# View docker compose
cat docker/docker-compose.yml | grep njs_shared_data
```

### Check Scripts

```bash
# Verify orchestrator uses new paths
grep "data/collected" scripts/orchestrator.py
grep "data/extracted" scripts/orchestrator.py
grep "data/aggregated_core" scripts/orchestrator.py
```

## 📚 Documentation

Complete documentation available in:
- `documentation/SHARED_DATA_STRUCTURE.md` - Comprehensive guide
- `README.md` - Updated project structure
- `config/paths_config.yaml` - Path configuration with comments

## 🎯 Summary

The data reorganization provides:

1. **Single shared data directory** - `njs_shared_data/` at sibling level
2. **Organized data structure** - All data under `data/` subdirectory
3. **Clear data flow** - collected → extracted → aggregated_core → pii_scan_results
4. **Configuration-driven** - All paths in `paths_config.yaml`
5. **Docker-friendly** - Single volume mount
6. **Easy to manage** - Backup, monitor, clean in one place

All scripts, Docker containers, and deployment tools now use this unified structure through the `SHARED_DATA_PATH` environment variable and configuration files.

---

**Reorganization completed on:** February 17, 2026
**Project:** NJS Orbit Node Pipeline
**Version:** 1.0.0
