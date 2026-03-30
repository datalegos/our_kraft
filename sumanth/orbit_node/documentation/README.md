# DataLegos Documentation

All project documentation organized in one place.

## 📚 Documentation Files

### Architecture & Design
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) - Project folder structure
- [FOLDER_STRUCTURE_ORGANIZED.md](FOLDER_STRUCTURE_ORGANIZED.md) - Organized structure plan

### Deployment & Setup
- [SERVER_DEPLOYMENT_GUIDE.md](SERVER_DEPLOYMENT_GUIDE.md) - Complete server deployment guide
- [QUICK_START_SERVER.md](QUICK_START_SERVER.md) - Quick start guide
- [DEPLOYMENT_FLOW.md](DEPLOYMENT_FLOW.md) - Visual deployment flow
- [README_DOCKER.md](README_DOCKER.md) - Docker deployment details
- [DOCKER_IMPLEMENTATION_SUMMARY.md](DOCKER_IMPLEMENTATION_SUMMARY.md) - Docker implementation

### Features & Implementation
- [DAY0_NODE_GRAPH_CREATION.md](DAY0_NODE_GRAPH_CREATION.md) - Day 0 Node Graph creation
- [AGGREGATION_IMPLEMENTATION.md](AGGREGATION_IMPLEMENTATION.md) - Aggregation implementation
- [AGGREGATION_COMPARISON.md](AGGREGATION_COMPARISON.md) - Aggregation approaches
- [PRIVACY_IMPLEMENTATION_SUMMARY.md](PRIVACY_IMPLEMENTATION_SUMMARY.md) - Privacy implementation
- [PII_DETECTION_COMPLETE.md](PII_DETECTION_COMPLETE.md) - PII detection details

### Guides & Tutorials
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Pipeline usage guide
- [INSTALL_PRESIDIO.md](INSTALL_PRESIDIO.md) - Presidio installation
- [PII_SCAN_ANALYSIS.md](PII_SCAN_ANALYSIS.md) - PII scan analysis
- [README_AGGREGATION.md](README_AGGREGATION.md) - Aggregation guide

### Reference
- [ENDPOINTS_REFERENCE.md](ENDPOINTS_REFERENCE.md) - API endpoints reference
- [wazuh_endpoints_structure.csv](wazuh_endpoints_structure.csv) - Wazuh endpoints

### Project History
- [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - Summary of changes
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - Implementation status
- [ORGANIZATION_SUMMARY.md](ORGANIZATION_SUMMARY.md) - Organization summary

## 🚀 Quick Links

**Getting Started:**
- [Quick Start Guide](QUICK_START_SERVER.md)
- [Server Deployment](SERVER_DEPLOYMENT_GUIDE.md)

**For Developers:**
- [Architecture](ARCHITECTURE.md)
- [Pipeline Guide](PIPELINE_GUIDE.md)

**For Operations:**
- [Deployment Flow](DEPLOYMENT_FLOW.md)
- [Docker Guide](README_DOCKER.md)

## 📁 Project Structure

```
orbit_node/
├── documentation/           # All documentation (this folder)
├── deployment_scripts/      # Deployment and management scripts
├── config/                  # Configuration files
├── docker/                  # Docker setup
├── scripts/                 # Pipeline scripts
├── graph_builder/           # Graph building modules
├── utils/                   # Utility modules
├── standards/               # Engineering standards
└── datalegos.sh            # Main management script
```

## 🛠️ Management Script

Use the main management script for all operations:

```bash
# Setup and start
./datalegos.sh install
./datalegos.sh start

# View logs
./datalegos.sh logs

# Run pipeline
./datalegos.sh run

# Backup
./datalegos.sh backup

# Help
./datalegos.sh help
```

See [../datalegos.sh](../datalegos.sh) for all available commands.
