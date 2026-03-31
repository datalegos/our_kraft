# NJS Data Structure - Visual Diagram

## 📂 Complete Directory Layout

```
parent_directory/
│
├── orbit_node/                          # PROJECT CODE
│   │
│   ├── config/                          # Configuration templates
│   │   ├── paths_config.yaml           # ← Defines all data paths
│   │   ├── aggregation_config.yaml     # ← Aggregation rules
│   │   ├── neo4j_config.yaml           # ← Neo4j settings
│   │   └── graph_config.yaml           # ← Graph schema
│   │
│   ├── deployment_scripts/              # Deployment automation
│   │   ├── start_all.sh                # ← Creates njs_shared_data
│   │   ├── start.sh
│   │   ├── stop.sh
│   │   ├── logs.sh
│   │   ├── status.sh
│   │   └── backup.sh
│   │
│   ├── docker/                          # Docker configuration
│   │   ├── docker-compose.yml          # ← Mounts njs_shared_data
│   │   ├── Dockerfile
│   │   └── entrypoint.sh               # ← Creates directory structure
│   │
│   ├── scripts/                         # Pipeline scripts
│   │   ├── orchestrator.py             # ← Uses SHARED_DATA_PATH
│   │   ├── main.py                     # ← Collects to data/collected
│   │   ├── extract_data.py             # ← Extracts to data/extracted
│   │   ├── aggregate_data_v2.py        # ← Aggregates to data/aggregated_core
│   │   └── detect_pii.py               # ← Scans data/aggregated_core
│   │
│   ├── graph_builder/                   # Graph building modules
│   ├── utils/                           # Utility modules
│   ├── documentation/                   # All documentation
│   ├── pyproject.toml                   # Poetry dependencies
│   ├── Makefile                         # ← make setup creates njs_shared_data
│   └── README.md
│
│
└── njs_shared_data/                     # ALL PIPELINE DATA
    │
    ├── config/                          # Runtime config (optional)
    │   └── (copied from project)
    │
    ├── data/                            # ← ALL DATA FILES
    │   │
    │   ├── collected/                   # ← Step 1: Raw Wazuh data
    │   │   ├── 20260210_201344/        # Timestamped session
    │   │   │   ├── agents_manager/
    │   │   │   │   └── All_Agents.json
    │   │   │   ├── host/
    │   │   │   │   ├── agent_000/
    │   │   │   │   │   └── Syscollector_OS_Info_000.json
    │   │   │   │   ├── agent_001/
    │   │   │   │   └── ...
    │   │   │   ├── packages/
    │   │   │   │   ├── agent_000/
    │   │   │   │   │   └── Syscollector_Packages_000.json
    │   │   │   │   └── ...
    │   │   │   ├── hardware/
    │   │   │   ├── fim/
    │   │   │   ├── groups/
    │   │   │   └── vulnerabilities/
    │   │   ├── 20260211_143022/        # Next session
    │   │   └── .done                    # ✓ Collection complete
    │   │
    │   ├── extracted/                   # ← Step 2: Normalized data
    │   │   ├── 20260210_201344/
    │   │   │   ├── agents.json         # Normalized agents
    │   │   │   ├── hosts.json          # Normalized hosts
    │   │   │   ├── packages.json       # Normalized packages
    │   │   │   ├── hardware.json       # Normalized hardware
    │   │   │   ├── vulnerabilities.json# Normalized vulnerabilities
    │   │   │   └── relationships.json  # Entity relationships
    │   │   ├── 20260211_143022/
    │   │   └── .done                    # ✓ Extraction complete
    │   │
    │   ├── aggregated/                  # ← Step 3: Intermediate (optional)
    │   │   ├── 20260210_201344/
    │   │   │   ├── asset_aggregation.json
    │   │   │   ├── software_aggregation.json
    │   │   │   └── vulnerability_aggregation.json
    │   │   └── ...
    │   │
    │   ├── aggregated_core/             # ← Step 4: FINAL OUTPUT ★
    │   │   ├── 20260210_201344/
    │   │   │   ├── core_aggregation.json      # ★ SEND TO CORE GRAPH
    │   │   │   ├── exposure_surface.json      # Attack surface metrics
    │   │   │   ├── sensitivity_surface.json   # Data sensitivity metrics
    │   │   │   ├── outcome_metrics.json       # Security outcomes
    │   │   │   └── summary_report.txt         # Human-readable summary
    │   │   ├── 20260211_143022/
    │   │   └── .done                    # ✓ Aggregation complete
    │   │
    │   └── pii_scan_results/            # ← Step 5: PII Validation
    │       ├── 20260210_201344/
    │       │   ├── pii_scan_results.json      # Scan results
    │       │   ├── pii_scan_summary.txt       # Summary
    │       │   └── pii_findings_detail.json   # Detailed findings
    │       ├── 20260211_143022/
    │       └── .done                    # ✓ PII scan complete
    │
    ├── logs/                            # ← ALL LOGS
    │   ├── pipeline.log                 # Main orchestrator log
    │   ├── collect_data.log             # Step 1 log
    │   ├── extract_data.log             # Step 2 log
    │   ├── build_node_graph.log         # Step 3 log
    │   ├── aggregate_data.log           # Step 4 log
    │   ├── detect_pii.log               # Step 5 log
    │   └── build_core_graph.log         # Step 6 log
    │
    └── pipeline/                        # ← PIPELINE STATE
        ├── node_graph.done              # ✓ Node Graph built
        ├── core_graph.done              # ✓ Core Graph built
        └── .done                        # ✓ Full pipeline complete
```

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         WAZUH API                                │
│                    (Security Monitoring)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: COLLECT DATA                                            │
│  Script: scripts/main.py                                         │
│  Output: njs_shared_data/data/collected/YYYYMMDD_HHMMSS/       │
│  ├── agents_manager/All_Agents.json                             │
│  ├── host/agent_XXX/Syscollector_OS_Info_XXX.json              │
│  ├── packages/agent_XXX/Syscollector_Packages_XXX.json         │
│  ├── hardware/agent_XXX/Syscollector_Hardware_XXX.json         │
│  └── vulnerabilities/agent_XXX/Vulnerabilities_XXX.json        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: EXTRACT DATA                                            │
│  Script: scripts/extract_data.py                                 │
│  Input:  data/collected/YYYYMMDD_HHMMSS/                       │
│  Output: njs_shared_data/data/extracted/YYYYMMDD_HHMMSS/       │
│  ├── agents.json          (Normalized agents)                   │
│  ├── hosts.json           (Normalized hosts)                    │
│  ├── packages.json        (Normalized packages)                 │
│  ├── hardware.json        (Normalized hardware)                 │
│  ├── vulnerabilities.json (Normalized vulnerabilities)          │
│  └── relationships.json   (Entity relationships)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: BUILD NODE GRAPH                                        │
│  Script: scripts/build_node_graph.py                             │
│  Input:  data/extracted/YYYYMMDD_HHMMSS/                       │
│  Output: Neo4j node_kg database                                 │
│  ├── Agent nodes                                                 │
│  ├── Host nodes                                                  │
│  ├── Package nodes                                               │
│  ├── Hardware nodes                                              │
│  ├── Vulnerability nodes                                         │
│  └── Relationships                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: AGGREGATE DATA (Privacy-Preserving)                     │
│  Script: scripts/aggregate_data_v2.py                            │
│  Input:  Neo4j node_kg database                                 │
│  Output: njs_shared_data/data/aggregated_core/YYYYMMDD_HHMMSS/ │
│  ├── core_aggregation.json      ★ FINAL OUTPUT FOR CORE        │
│  ├── exposure_surface.json      (Attack surface metrics)        │
│  ├── sensitivity_surface.json   (Data sensitivity)              │
│  ├── outcome_metrics.json       (Security outcomes)             │
│  └── summary_report.txt         (Human-readable)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: DETECT PII (Validation Gate)                           │
│  Script: scripts/detect_pii.py                                   │
│  Input:  data/aggregated_core/YYYYMMDD_HHMMSS/                 │
│  Output: njs_shared_data/data/pii_scan_results/YYYYMMDD_HHMMSS/│
│  ├── pii_scan_results.json      (Scan results)                  │
│  ├── pii_scan_summary.txt       (Summary)                       │
│  └── pii_findings_detail.json   (Detailed findings)             │
│  │                                                               │
│  ├─ ✓ No PII/PCI detected → Continue to Core Graph             │
│  └─ ✗ PII/PCI detected    → STOP (Not safe for Core)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓ (Only if PII scan passes)
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: BUILD CORE GRAPH                                        │
│  Script: scripts/build_core_graph.py                             │
│  Input:  data/aggregated_core/YYYYMMDD_HHMMSS/                 │
│  Output: Neo4j core database                                    │
│  ├── NJS_Bank node (Bank-level aggregations)                    │
│  ├── Exposure metrics                                            │
│  ├── Sensitivity metrics                                         │
│  └── Outcome metrics                                             │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Points

### Data Directories Purpose

| Directory | Purpose | Created By | Used By |
|-----------|---------|------------|---------|
| `data/collected/` | Raw Wazuh API responses | `main.py` | `extract_data.py` |
| `data/extracted/` | Normalized data for Node Graph | `extract_data.py` | `build_node_graph.py` |
| `data/aggregated/` | Intermediate aggregations (optional) | `aggregate_data_v2.py` | Internal |
| `data/aggregated_core/` | **FINAL OUTPUT for Core Graph** | `aggregate_data_v2.py` | `detect_pii.py`, `build_core_graph.py` |
| `data/pii_scan_results/` | PII/PCI validation results | `detect_pii.py` | Validation gate |

### Important Files

| File | Purpose |
|------|---------|
| `config/paths_config.yaml` | Defines all data paths |
| `njs_shared_data/pipeline/.done` | Pipeline completion marker |
| `njs_shared_data/data/aggregated_core/*/core_aggregation.json` | **FINAL OUTPUT** |
| `njs_shared_data/logs/pipeline.log` | Main execution log |

### Environment Variables

| Variable | Docker Value | Host Value |
|----------|--------------|------------|
| `SHARED_DATA_PATH` | `/shared_data` | `../njs_shared_data` |

## 📋 Quick Reference

```bash
# View raw Wazuh data
ls -la ../njs_shared_data/data/collected/

# View normalized data
ls -la ../njs_shared_data/data/extracted/

# View FINAL OUTPUT (for Core Graph)
ls -la ../njs_shared_data/data/aggregated_core/

# View PII scan results
ls -la ../njs_shared_data/data/pii_scan_results/

# View logs
tail -f ../njs_shared_data/logs/pipeline.log

# Check pipeline status
cat ../njs_shared_data/pipeline/.done
```

---

**Diagram Version:** 1.0.0  
**Last Updated:** February 17, 2026
