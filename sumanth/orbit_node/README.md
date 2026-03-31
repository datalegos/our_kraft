# NJS Orbit Node Pipeline

Privacy-preserving data aggregation and graph database pipeline for security monitoring.

## 🚀 Quick Start

```bash
# Setup and start
make setup
make build
make start

# View logs
make logs

# Check status
make status
```

## 📋 Overview

NJS Orbit Node collects security data from Wazuh, creates detailed Node Knowledge Graphs, performs privacy-preserving aggregations, and builds Core Graphs for bank-level analysis.

### Features

- 🔄 Automated Data Collection from Wazuh
- 🗄️ Node Knowledge Graph in Neo4j
- 🔒 Privacy-Preserving Aggregation
- 🛡️ PII/PCI Detection with Presidio
- 📊 Core Graph for Bank-Level Analysis
- 🐳 Complete Docker Deployment
- ✅ Sequential Validation Gates

## 📁 Project Structure

```
parent_directory/
├── orbit_node/              # Project code
│   ├── deployment_scripts/  # Deployment scripts (6 essential)
│   ├── documentation/       # All documentation
│   ├── config/              # Configuration files
│   ├── docker/              # Docker setup
│   ├── scripts/             # Pipeline scripts
│   ├── graph_builder/       # Graph building modules
│   ├── utils/               # Utility modules
│   ├── pyproject.toml       # Poetry dependencies
│   └── Makefile             # Make commands
│
└── njs_shared_data/         # All pipeline data (sibling directory)
    ├── config/              # Runtime config (optional)
    ├── data/
    │   ├── collected/       # Raw data from Wazuh
    │   ├── extracted/       # Normalized data for Node Graph
    │   ├── aggregated/      # Intermediate aggregations
    │   ├── aggregated_core/ # Final output for Core Graph
    │   └── pii_scan_results/# PII detection results
    ├── logs/                # All pipeline logs
    └── pipeline/            # Pipeline state markers
```

## 🛠️ Make Commands

```bash
# Setup
make setup          # Initial setup
make install        # Install Poetry dependencies locally
make build          # Build Docker images

# Operations
make start          # Start services
make stop           # Stop services
make restart        # Restart services
make logs           # View logs
make status         # Check status

# Pipeline
make run            # Run complete pipeline
make shell          # Access container shell

# Maintenance
make backup         # Backup databases
make clean          # Clean old data
make reset          # Reset everything
```

## ⚙️ Configuration

All configuration is externalized:

```bash
.env                         # Environment variables (secrets)
config/aggregation_config.yaml    # Aggregation rules
config/paths_config.yaml          # Data paths
config/neo4j_config.yaml          # Neo4j settings
config/graph_config.yaml          # Graph schema
```

## 🔄 Pipeline Flow

```
1. COLLECT DATA      → Collect from Wazuh API
2. EXTRACT DATA      → Normalize and extract
3. BUILD NODE GRAPH  → Create Node KG in Neo4j
4. AGGREGATE DATA    → Privacy-preserving aggregations
5. DETECT PII        → Scan for PII/PCI
6. BUILD CORE GRAPH  → Create Core Graph in Neo4j
```

## 🐳 Docker Deployment

### First Time Setup
```bash
make setup
make build
make start
```

### Daily Operations
```bash
make start          # Start
make logs           # Logs
make status         # Status
make stop           # Stop
```

### Access Neo4j
```bash
# Development
http://localhost:7474

# Production (SSH tunnel)
ssh -L 7474:localhost:7474 user@server
http://localhost:7474
```

## 💻 Local Development

```bash
# Install Poetry dependencies
make install

# Activate environment
poetry shell

# Run scripts
python scripts/main.py
python scripts/detect_pii.py
```

## 📚 Documentation

Complete documentation in [documentation/](documentation/) folder.

## 🔒 Security

- 🔐 All credentials in `.env` (never committed)
- 🛡️ PII/PCI detection before data sharing
- 🔒 Privacy-preserving aggregations
- 🚫 No sensitive data in logs

## 📖 Quick Reference

```bash
# Complete setup
make setup && make build && make start

# Daily operations
make start          # Start
make logs           # Logs
make status         # Status
make stop           # Stop

# Maintenance
make backup         # Backup
make clean          # Clean old data
```

## 🤝 Support

For issues or questions:
1. Check [documentation/](documentation/)
2. Review logs: `make logs`
3. Check status: `make status`
4. Contact NJS team

## 📄 License

Copyright © 2026 NJS Team
