# DataLegos - Orbit Node Pipeline

Privacy-preserving data aggregation and graph database pipeline for security monitoring.

## 🚀 Quick Start

```bash
# Setup and install
./datalegos.sh install

# Edit configuration
nano .env

# Start services
./datalegos.sh start

# View logs
./datalegos.sh logs
```

## 📋 Overview

DataLegos collects security data from Wazuh, creates detailed Node Knowledge Graphs, performs privacy-preserving aggregations, and builds Core Graphs for bank-level analysis.

### Features

- 🔄 **Automated Data Collection** - Collects data from Wazuh Manager API
- 🗄️ **Node Knowledge Graph** - Detailed per-agent security graph in Neo4j
- 🔒 **Privacy-Preserving Aggregation** - Statistical aggregations without PII
- 🛡️ **PII/PCI Detection** - Scans data using Microsoft Presidio
- 📊 **Core Graph** - Bank-level aggregated security graph
- 🐳 **Docker Deployment** - Complete containerized pipeline
- ✅ **Validation Gates** - Sequential execution with validation at each step

## 📁 Project Structure

```
orbit_node/
├── datalegos.sh             # Main management script
├── documentation/           # All documentation
├── deployment_scripts/      # Deployment scripts
├── config/                  # Configuration files
├── docker/                  # Docker setup
├── scripts/                 # Pipeline scripts
├── graph_builder/           # Graph building modules
├── utils/                   # Utility modules
└── standards/               # Engineering standards
```

Data is stored in `../orbit_node_shared_data/` (sibling directory).

## 🛠️ Management Script

All operations through one script:

```bash
# Setup & Build
./datalegos.sh install       # Complete installation
./datalegos.sh setup         # Initial setup only
./datalegos.sh build         # Build Docker images

# Service Management
./datalegos.sh start         # Start services
./datalegos.sh stop          # Stop services
./datalegos.sh restart       # Restart services
./datalegos.sh status        # Show status
./datalegos.sh logs          # View logs

# Pipeline Operations
./datalegos.sh run           # Run complete pipeline
./datalegos.sh run-step <name>  # Run specific step

# Production Mode
./datalegos.sh prod-start    # Start production services
./datalegos.sh prod-logs     # View production logs

# Maintenance
./datalegos.sh backup        # Backup Neo4j
./datalegos.sh health        # Check health
./datalegos.sh clean-data    # Clean old data

# Help
./datalegos.sh help          # Show all commands
```

## 📚 Documentation

Complete documentation in [documentation/](documentation/) folder:

- **[Quick Start Guide](documentation/QUICK_START_SERVER.md)** - Fast deployment
- **[Server Deployment Guide](documentation/SERVER_DEPLOYMENT_GUIDE.md)** - Complete deployment
- **[Deployment Flow](documentation/DEPLOYMENT_FLOW.md)** - Visual deployment guide
- **[Architecture](documentation/ARCHITECTURE.md)** - System architecture
- **[Pipeline Guide](documentation/PIPELINE_GUIDE.md)** - Pipeline usage

See [documentation/README.md](documentation/README.md) for all documentation.

## ⚙️ Configuration

All configuration is externalized:

```bash
# Environment variables
.env                         # Secrets and environment config

# Application configuration
config/aggregation_config.yaml    # Aggregation rules, PII filters
config/paths_config.yaml          # Data directory paths
config/neo4j_config.yaml          # Neo4j connection settings
config/graph_config.yaml          # Graph schema, bank_id
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

Each step includes validation gates and creates `.done` markers.

## 🐳 Docker Deployment

### Development Mode
```bash
./datalegos.sh start
./datalegos.sh logs
```

### Production Mode
```bash
./datalegos.sh prod-start
./datalegos.sh prod-logs
```

### Access Neo4j
```bash
# Development (ports exposed)
http://localhost:7474

# Production (SSH tunnel)
ssh -L 7474:localhost:7474 user@server
http://localhost:7474
```

## 🔒 Security

- 🔐 All credentials in `.env` (never committed)
- 🛡️ PII/PCI detection before data sharing
- 🔒 Privacy-preserving aggregations
- 🚫 No sensitive data in logs

## 📊 Monitoring

```bash
# Check status
./datalegos.sh status

# View logs
./datalegos.sh logs

# Check health
./datalegos.sh health

# Check pipeline completion
./datalegos.sh check-pipeline

# Check PII scan results
./datalegos.sh check-pii
```

## 🔧 Maintenance

```bash
# Backup databases
./datalegos.sh backup

# Clean old data (30+ days)
./datalegos.sh clean-data

# Update application
./datalegos.sh update

# Check disk usage
./datalegos.sh disk-usage
```

## 🚨 Troubleshooting

```bash
# Check logs
./datalegos.sh logs

# Check health
./datalegos.sh health

# Check status
./datalegos.sh status

# Access shell
./datalegos.sh shell
```

See [documentation/SERVER_DEPLOYMENT_GUIDE.md](documentation/SERVER_DEPLOYMENT_GUIDE.md#troubleshooting) for detailed troubleshooting.

## 📖 Quick Reference

```bash
# Complete installation
./datalegos.sh install && ./datalegos.sh start

# View logs
./datalegos.sh logs

# Run pipeline
./datalegos.sh run

# Backup
./datalegos.sh backup

# Help
./datalegos.sh help
```

## 🤝 Support

For issues or questions:
1. Check [documentation/](documentation/)
2. Review logs: `./datalegos.sh logs`
3. Check health: `./datalegos.sh health`
4. Contact DataLegos team

## 📄 License

Copyright © 2026 DataLegos Team

## 🔗 Links

- [Complete Documentation](documentation/README.md)
- [Deployment Scripts](deployment_scripts/README.md)
- [Engineering Standards](standards/)
