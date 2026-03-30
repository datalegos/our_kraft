# Quick Start - New NJS Shared Data Structure

## 🚀 Setup (First Time)

```bash
# 1. Setup (creates njs_shared_data directory)
make setup

# 2. Edit configuration
nano .env

# 3. Build Docker images
make build

# 4. Start services
make start
```

## 📁 Directory Structure

```
parent_directory/
├── orbit_node/              # Your project (code only)
└── njs_shared_data/         # All data (created by setup)
    ├── data/
    │   ├── collected/       # Raw Wazuh data
    │   ├── extracted/       # Normalized data
    │   ├── aggregated_core/ # Final output → Core Graph
    │   └── pii_scan_results/# PII validation
    ├── logs/                # All logs
    └── pipeline/            # State markers
```

## 📊 Data Flow

```
Wazuh API
   ↓
data/collected/          (Raw JSON from Wazuh)
   ↓
data/extracted/          (Normalized for Node Graph)
   ↓
Neo4j node_kg           (Node Knowledge Graph)
   ↓
data/aggregated_core/   (Privacy-preserving aggregations)
   ↓
data/pii_scan_results/  (Validation: No PII/PCI)
   ↓
Neo4j core              (Core Graph - Bank Level)
```

## 🔍 Quick Commands

```bash
# View logs
make logs

# Check status
make status

# View collected data
ls -la ../njs_shared_data/data/collected/

# View final output (for Core Graph)
ls -la ../njs_shared_data/data/aggregated_core/

# View PII scan results
ls -la ../njs_shared_data/data/pii_scan_results/

# Check pipeline completion
cat ../njs_shared_data/pipeline/.done

# Stop services
make stop

# Backup databases
make backup

# Clean old data
make clean
```

## 📝 Key Files

### Configuration
- `.env` - Secrets and environment variables
- `config/paths_config.yaml` - All data paths
- `config/aggregation_config.yaml` - Aggregation rules

### Data Locations
- `../njs_shared_data/data/collected/` - Raw Wazuh data
- `../njs_shared_data/data/extracted/` - Normalized data
- `../njs_shared_data/data/aggregated_core/` - **FINAL OUTPUT**
- `../njs_shared_data/data/pii_scan_results/` - PII validation

### Logs
- `../njs_shared_data/logs/pipeline.log` - Main log
- `../njs_shared_data/logs/collect_data.log` - Collection log
- `../njs_shared_data/logs/aggregate_data.log` - Aggregation log
- `../njs_shared_data/logs/detect_pii.log` - PII scan log

## 🎯 Important Notes

1. **All data is in `njs_shared_data/`** - Not in project folder
2. **Config drives paths** - Edit `config/paths_config.yaml` to change paths
3. **Timestamped sessions** - Each run creates `YYYYMMDD_HHMMSS/` folder
4. **Final output** - `data/aggregated_core/` goes to Core Graph
5. **PII validation** - Must pass before Core Graph build

## 🔧 Troubleshooting

### Directory not found
```bash
make setup  # Recreates structure
```

### View errors
```bash
tail -f ../njs_shared_data/logs/pipeline.log
grep ERROR ../njs_shared_data/logs/*.log
```

### Check disk space
```bash
du -sh ../njs_shared_data/
df -h ../njs_shared_data/
```

### Clean old data
```bash
make clean  # Removes data older than 7 days
```

## 📚 Full Documentation

- `documentation/SHARED_DATA_STRUCTURE.md` - Complete guide
- `documentation/DATA_REORGANIZATION_COMPLETE.md` - What changed
- `README.md` - Project overview

## ✅ Verification Checklist

- [ ] `njs_shared_data/` directory exists at sibling level
- [ ] `.env` file configured with Wazuh credentials
- [ ] Docker services running (`make status`)
- [ ] Logs being written to `njs_shared_data/logs/`
- [ ] Data being collected to `njs_shared_data/data/collected/`

---

**Quick Reference for:** NJS Orbit Node Pipeline v1.0.0
