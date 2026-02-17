#!/bin/bash
# Migration Script - Reorganize Folder Structure
# Run this script to reorganize the project structure

set -e  # Exit on error

echo "=========================================="
echo "DataLegos Folder Structure Migration"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "scripts" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_info "Starting migration..."
echo ""

# Step 1: Create new documentation structure
print_info "Step 1: Creating new documentation structure..."
mkdir -p docs/{architecture,deployment,features,implementation,guides,reference}

# Step 2: Move architecture docs
print_info "Step 2: Moving architecture documentation..."
[ -f "ARCHITECTURE.md" ] && mv ARCHITECTURE.md docs/architecture/ && print_info "  ✓ Moved ARCHITECTURE.md"
[ -f "FOLDER_STRUCTURE.md" ] && mv FOLDER_STRUCTURE.md docs/architecture/ && print_info "  ✓ Moved FOLDER_STRUCTURE.md"
[ -f "docs/ARCHITECTURE_DIAGRAM.md" ] && mv docs/ARCHITECTURE_DIAGRAM.md docs/architecture/ && print_info "  ✓ Moved ARCHITECTURE_DIAGRAM.md"

# Step 3: Move deployment docs
print_info "Step 3: Moving deployment documentation..."
[ -f "README_DOCKER.md" ] && mv README_DOCKER.md docs/deployment/DOCKER_DEPLOYMENT.md && print_info "  ✓ Moved README_DOCKER.md -> DOCKER_DEPLOYMENT.md"
[ -f "docker-quickstart.sh" ] && mv docker-quickstart.sh docs/deployment/ && print_info "  ✓ Moved docker-quickstart.sh"
[ -f "SERVER_DEPLOYMENT_GUIDE.md" ] && mv SERVER_DEPLOYMENT_GUIDE.md docs/deployment/ && print_info "  ✓ Moved SERVER_DEPLOYMENT_GUIDE.md"

# Step 4: Move feature docs
print_info "Step 4: Moving feature documentation..."
[ -f "docs/AGGREGATION.md" ] && mv docs/AGGREGATION.md docs/features/ && print_info "  ✓ Moved AGGREGATION.md"
[ -f "DAY0_NODE_GRAPH_CREATION.md" ] && mv DAY0_NODE_GRAPH_CREATION.md docs/features/DAY0_NODE_GRAPH.md && print_info "  ✓ Moved DAY0_NODE_GRAPH_CREATION.md"
[ -f "docs/PII_DETECTION.md" ] && mv docs/PII_DETECTION.md docs/features/ && print_info "  ✓ Moved PII_DETECTION.md"
[ -f "docs/PRIVACY_PRESERVING_AGGREGATION.md" ] && mv docs/PRIVACY_PRESERVING_AGGREGATION.md docs/features/PRIVACY_AGGREGATION.md && print_info "  ✓ Moved PRIVACY_PRESERVING_AGGREGATION.md"

# Step 5: Move implementation docs
print_info "Step 5: Moving implementation documentation..."
[ -f "AGGREGATION_COMPARISON.md" ] && mv AGGREGATION_COMPARISON.md docs/implementation/ && print_info "  ✓ Moved AGGREGATION_COMPARISON.md"
[ -f "AGGREGATION_IMPLEMENTATION.md" ] && mv AGGREGATION_IMPLEMENTATION.md docs/implementation/ && print_info "  ✓ Moved AGGREGATION_IMPLEMENTATION.md"
[ -f "DOCKER_IMPLEMENTATION_SUMMARY.md" ] && mv DOCKER_IMPLEMENTATION_SUMMARY.md docs/implementation/DOCKER_IMPLEMENTATION.md && print_info "  ✓ Moved DOCKER_IMPLEMENTATION_SUMMARY.md"
[ -f "PRIVACY_IMPLEMENTATION_SUMMARY.md" ] && mv PRIVACY_IMPLEMENTATION_SUMMARY.md docs/implementation/PRIVACY_IMPLEMENTATION.md && print_info "  ✓ Moved PRIVACY_IMPLEMENTATION_SUMMARY.md"
[ -f "CHANGES_SUMMARY.md" ] && mv CHANGES_SUMMARY.md docs/implementation/ && print_info "  ✓ Moved CHANGES_SUMMARY.md"

# Step 6: Move guide docs
print_info "Step 6: Moving guide documentation..."
[ -f "PIPELINE_GUIDE.md" ] && mv PIPELINE_GUIDE.md docs/guides/ && print_info "  ✓ Moved PIPELINE_GUIDE.md"
[ -f "INSTALL_PRESIDIO.md" ] && mv INSTALL_PRESIDIO.md docs/guides/ && print_info "  ✓ Moved INSTALL_PRESIDIO.md"
[ -f "PII_SCAN_ANALYSIS.md" ] && mv PII_SCAN_ANALYSIS.md docs/guides/ && print_info "  ✓ Moved PII_SCAN_ANALYSIS.md"

# Step 7: Move reference docs
print_info "Step 7: Moving reference documentation..."
[ -f "ENDPOINTS_REFERENCE.md" ] && mv ENDPOINTS_REFERENCE.md docs/reference/ && print_info "  ✓ Moved ENDPOINTS_REFERENCE.md"
[ -f "wazuh_endpoints_structure.csv" ] && mv wazuh_endpoints_structure.csv docs/reference/ && print_info "  ✓ Moved wazuh_endpoints_structure.csv"

# Step 8: Clean up old/redundant docs
print_info "Step 8: Cleaning up redundant documentation..."
[ -f "IMPLEMENTATION_COMPLETE.md" ] && rm -f IMPLEMENTATION_COMPLETE.md && print_info "  ✓ Removed IMPLEMENTATION_COMPLETE.md"
[ -f "PII_DETECTION_COMPLETE.md" ] && rm -f PII_DETECTION_COMPLETE.md && print_info "  ✓ Removed PII_DETECTION_COMPLETE.md"
[ -f "README_AGGREGATION.md" ] && rm -f README_AGGREGATION.md && print_info "  ✓ Removed README_AGGREGATION.md"

# Step 9: Update .gitignore
print_info "Step 9: Updating .gitignore..."

# Check if entries already exist
if ! grep -q "# Data directories" .gitignore 2>/dev/null; then
    cat >> .gitignore <<EOF

# Data directories (stored in ../orbit_node_shared_data)
aggregated_data/
aggregated_data_core/
collected_data/
extracted_data/
logs/
pii_scan_results/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local
EOF
    print_info "  ✓ Updated .gitignore"
else
    print_warning "  ! .gitignore already contains data directory entries"
fi

# Step 10: Create tests directory structure
print_info "Step 10: Creating tests directory structure..."
mkdir -p tests/{unit,integration,fixtures}
touch tests/__init__.py
print_info "  ✓ Created tests directory structure"

# Step 11: Create documentation index
print_info "Step 11: Creating documentation index..."
cat > docs/README.md <<'EOF'
# DataLegos Documentation

Welcome to the DataLegos documentation. This directory contains all project documentation organized by category.

## Documentation Structure

### 📐 Architecture
- [ARCHITECTURE.md](architecture/ARCHITECTURE.md) - System architecture overview
- [ARCHITECTURE_DIAGRAM.md](architecture/ARCHITECTURE_DIAGRAM.md) - Visual architecture diagrams
- [FOLDER_STRUCTURE.md](architecture/FOLDER_STRUCTURE.md) - Project folder organization

### 🚀 Deployment
- [DOCKER_DEPLOYMENT.md](deployment/DOCKER_DEPLOYMENT.md) - Docker deployment guide
- [SERVER_DEPLOYMENT_GUIDE.md](deployment/SERVER_DEPLOYMENT_GUIDE.md) - Production server deployment
- [docker-quickstart.sh](deployment/docker-quickstart.sh) - Quick start script

### ✨ Features
- [AGGREGATION.md](features/AGGREGATION.md) - Data aggregation features
- [DAY0_NODE_GRAPH.md](features/DAY0_NODE_GRAPH.md) - Day 0 Node Graph creation
- [PII_DETECTION.md](features/PII_DETECTION.md) - PII/PCI detection
- [PRIVACY_AGGREGATION.md](features/PRIVACY_AGGREGATION.md) - Privacy-preserving aggregation

### 🔧 Implementation
- [AGGREGATION_COMPARISON.md](implementation/AGGREGATION_COMPARISON.md) - Aggregation approaches comparison
- [AGGREGATION_IMPLEMENTATION.md](implementation/AGGREGATION_IMPLEMENTATION.md) - Aggregation implementation details
- [DOCKER_IMPLEMENTATION.md](implementation/DOCKER_IMPLEMENTATION.md) - Docker implementation details
- [PRIVACY_IMPLEMENTATION.md](implementation/PRIVACY_IMPLEMENTATION.md) - Privacy implementation details
- [CHANGES_SUMMARY.md](implementation/CHANGES_SUMMARY.md) - Summary of changes

### 📚 Guides
- [PIPELINE_GUIDE.md](guides/PIPELINE_GUIDE.md) - Pipeline usage guide
- [INSTALL_PRESIDIO.md](guides/INSTALL_PRESIDIO.md) - Presidio installation guide
- [PII_SCAN_ANALYSIS.md](guides/PII_SCAN_ANALYSIS.md) - PII scan analysis guide

### 📖 Reference
- [ENDPOINTS_REFERENCE.md](reference/ENDPOINTS_REFERENCE.md) - API endpoints reference
- [wazuh_endpoints_structure.csv](reference/wazuh_endpoints_structure.csv) - Wazuh endpoints structure

## Quick Links

- **Getting Started**: See [../README.md](../README.md)
- **Docker Deployment**: [deployment/DOCKER_DEPLOYMENT.md](deployment/DOCKER_DEPLOYMENT.md)
- **Server Deployment**: [deployment/SERVER_DEPLOYMENT_GUIDE.md](deployment/SERVER_DEPLOYMENT_GUIDE.md)
- **Architecture**: [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md)

## Contributing

When adding new documentation:
1. Place it in the appropriate category folder
2. Update this README.md with a link
3. Follow the existing documentation style
4. Use clear, descriptive filenames

## Support

For questions or issues, contact the DataLegos team.
EOF
print_info "  ✓ Created docs/README.md"

# Step 12: Create main README update
print_info "Step 12: Creating updated main README..."
cat > README_NEW.md <<'EOF'
# DataLegos - Orbit Node Pipeline

Privacy-preserving data aggregation and graph database pipeline for security monitoring.

## Overview

DataLegos collects security data from Wazuh, creates detailed Node Knowledge Graphs, performs privacy-preserving aggregations, and builds Core Graphs for bank-level analysis.

## Features

- 🔄 **Automated Data Collection** - Collects data from Wazuh Manager API
- 🗄️ **Node Knowledge Graph** - Detailed per-agent security graph in Neo4j
- 🔒 **Privacy-Preserving Aggregation** - Statistical aggregations without PII
- 🛡️ **PII/PCI Detection** - Scans data using Microsoft Presidio
- 📊 **Core Graph** - Bank-level aggregated security graph
- 🐳 **Docker Deployment** - Complete containerized pipeline
- ✅ **Validation Gates** - Sequential execution with validation at each step

## Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- Access to Wazuh Manager API

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/orbit_node.git
cd orbit_node

# Configure environment
cp .env.example .env
nano .env  # Edit with your settings

# Start services
cd docker
docker compose up -d

# Monitor progress
docker compose logs -f pipeline
```

## Documentation

📚 **[Complete Documentation](docs/README.md)**

### Quick Links
- 🚀 [Server Deployment Guide](docs/deployment/SERVER_DEPLOYMENT_GUIDE.md)
- 🐳 [Docker Deployment](docs/deployment/DOCKER_DEPLOYMENT.md)
- 📐 [Architecture](docs/architecture/ARCHITECTURE.md)
- ✨ [Features](docs/features/)
- 📚 [Guides](docs/guides/)

## Pipeline Flow

```
1. COLLECT DATA      → Collect from Wazuh API
2. EXTRACT DATA      → Normalize and extract
3. BUILD NODE GRAPH  → Create Node KG in Neo4j
4. AGGREGATE DATA    → Privacy-preserving aggregations
5. DETECT PII        → Scan for PII/PCI
6. BUILD CORE GRAPH  → Create Core Graph in Neo4j
```

## Project Structure

```
orbit_node/
├── config/              # Configuration files
├── docker/              # Docker deployment
├── docs/                # Documentation
├── graph_builder/       # Graph building modules
├── scripts/             # Pipeline scripts
├── standards/           # Engineering standards
├── tests/               # Test suite
└── utils/               # Utility modules
```

Data is stored in `../orbit_node_shared_data/` (sibling directory).

## Configuration

All configuration is externalized:
- `.env` - Environment variables and secrets
- `config/*.yaml` - Application configuration
- No hard-coded values

## Usage

### Run Complete Pipeline
```bash
docker compose up -d
docker compose logs -f pipeline
```

### Run Single Step
```bash
docker compose exec pipeline python /app/scripts/orchestrator.py --step collect_data
```

### Access Neo4j
```bash
# Open browser
http://localhost:7474

# Login with credentials from .env
```

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements_presidio.txt

# Run scripts locally
python scripts/main.py
python scripts/detect_pii.py
```

### Testing
```bash
# Run tests (future)
pytest tests/
```

## Monitoring

### View Logs
```bash
docker compose logs -f pipeline
docker compose exec pipeline cat /shared_data/logs/pipeline.log
```

### Check Status
```bash
docker compose ps
docker compose exec pipeline cat /shared_data/pipeline/.done
```

## Security

- 🔒 All credentials in `.env` (never committed)
- 🛡️ PII/PCI detection before data sharing
- 🔐 Privacy-preserving aggregations
- 🚫 No sensitive data in logs

## Support

For issues or questions:
1. Check [Troubleshooting Guide](docs/deployment/SERVER_DEPLOYMENT_GUIDE.md#troubleshooting)
2. Review logs in `/shared_data/logs/`
3. Contact DataLegos team

## License

Copyright © 2026 DataLegos Team

## Contributing

See [standards/](standards/) for engineering standards and best practices.
EOF
print_info "  ✓ Created README_NEW.md (review and rename to README.md)"

echo ""
print_info "=========================================="
print_info "Migration Complete!"
print_info "=========================================="
echo ""
print_info "Next Steps:"
echo "  1. Review the new structure:"
echo "     ls -la docs/"
echo ""
echo "  2. Review the new README:"
echo "     cat README_NEW.md"
echo ""
echo "  3. If satisfied, replace old README:"
echo "     mv README.md README_OLD.md"
echo "     mv README_NEW.md README.md"
echo ""
echo "  4. Commit changes:"
echo "     git add ."
echo "     git commit -m 'Reorganize folder structure'"
echo ""
echo "  5. Remove data directories from git (if tracked):"
echo "     git rm -r --cached aggregated_data/ collected_data/ extracted_data/ logs/ pii_scan_results/"
echo "     git commit -m 'Remove data directories from git'"
echo ""
print_warning "Note: Review all changes before committing!"
echo ""
