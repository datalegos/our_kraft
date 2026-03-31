# NJS Orbit Node - Complete Project Context

## 🎯 Project Overview

**Project Name**: NJS Orbit Node Pipeline  
**Organization**: NJSecure (NJS)  
**Purpose**: Privacy-preserving security data aggregation and graph database pipeline  
**Version**: 1.0.0  
**Status**: Production Ready

### What is NJS Orbit Node?

NJS Orbit Node is a comprehensive data pipeline that collects security monitoring data from Wazuh, creates detailed Node Knowledge Graphs, performs privacy-preserving aggregations, and builds Core Graphs for bank-level security analysis.

The system enables:
- Automated security data collection from Wazuh
- Detailed node-level knowledge graphs
- Privacy-preserving data aggregation
- PII/PCI detection and validation
- Bank-level security metrics and analysis

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         WAZUH PLATFORM                           │
│                    (Security Monitoring)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Wazuh Manager│  │ Wazuh Indexer│  │ Wazuh Agents │         │
│  │   (API)      │  │ (OpenSearch) │  │  (Endpoints) │         │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
└─────────┼──────────────────┼──────────────────────────────────┘
          │                  │
          ↓                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    NJS ORBIT NODE PIPELINE                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STEP 1: DATA COLLECTION                                 │  │
│  │  - Agents, Hosts, Packages, Hardware, CPE, FIM, Vulns   │  │
│  │  Output: data/collected/YYYYMMDD_HHMMSS/                │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STEP 2: DATA EXTRACTION                                 │  │
│  │  - Normalize and structure data                          │  │
│  │  Output: data/extracted/YYYYMMDD_HHMMSS/                │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STEP 3: BUILD NODE GRAPH                                │  │
│  │  - Create detailed Node Knowledge Graph                  │  │
│  │  Output: Neo4j node_kg database                          │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STEP 4: AGGREGATE DATA (Privacy-Preserving)            │  │
│  │  - Create bank-level aggregations                        │  │
│  │  Output: data/aggregated_core/YYYYMMDD_HHMMSS/          │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STEP 5: DETECT PII (Validation Gate)                   │  │
│  │  - Scan for PII/PCI data using Presidio                 │  │
│  │  Output: data/pii_scan_results/YYYYMMDD_HHMMSS/         │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STEP 6: BUILD CORE GRAPH                                │  │
│  │  - Create bank-level Core Graph                          │  │
│  │  Output: Neo4j core database                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```


### System Components

#### 1. Data Sources
- **Wazuh Manager API**: Agent data, syscollector data (OS, packages, hardware, CPE), FIM, groups
- **Wazuh Indexer**: Vulnerability data from dedicated vulnerability index

#### 2. Pipeline Components
- **Data Collector**: Collects raw data from Wazuh
- **Data Extractor**: Normalizes and structures data
- **Node Graph Builder**: Creates detailed knowledge graph
- **Data Aggregator**: Privacy-preserving aggregations
- **PII Detector**: Scans for sensitive data using Presidio
- **Core Graph Builder**: Creates bank-level graph

#### 3. Storage
- **Neo4j Databases**:
  - `node_kg`: Node-level knowledge graph (detailed)
  - `core`: Bank-level core graph (aggregated)
- **File Storage**: All data in `njs_shared_data/` directory

#### 4. Orchestration
- **Docker Compose**: Container orchestration
- **Pipeline Orchestrator**: Sequential execution with validation gates
- **Makefile**: Simplified command interface

## 📁 Project Structure

```
parent_directory/
├── orbit_node/                      # Project code
│   ├── config/                      # Configuration files
│   │   ├── aggregation_config.yaml # Aggregation rules
│   │   ├── paths_config.yaml       # Data paths
│   │   ├── neo4j_config.yaml       # Neo4j settings
│   │   ├── graph_config.yaml       # Graph schema
│   │   └── config.yaml             # Main config
│   │
│   ├── deployment_scripts/          # 6 essential scripts
│   │   ├── start_all.sh            # Complete setup
│   │   ├── start.sh                # Start services
│   │   ├── stop.sh                 # Stop services
│   │   ├── logs.sh                 # View logs
│   │   ├── status.sh               # Check status
│   │   └── backup.sh               # Backup databases
│   │
│   ├── docker/                      # Docker configuration
│   │   ├── docker-compose.yml      # Services definition
│   │   ├── Dockerfile              # Pipeline container
│   │   └── entrypoint.sh           # Container entrypoint
│   │
│   ├── scripts/                     # Pipeline scripts
│   │   ├── orchestrator.py         # Main orchestrator
│   │   ├── main.py                 # Data collection
│   │   ├── extract_data.py         # Data extraction
│   │   ├── build_node_graph.py     # Node graph builder
│   │   ├── aggregate_data_v2.py    # Aggregation
│   │   ├── detect_pii.py           # PII detection
│   │   └── build_core_graph.py     # Core graph builder
│   │
│   ├── graph_builder/               # Graph building modules
│   ├── utils/                       # Utility modules
│   │   ├── wazuh_manager_client.py # Wazuh Manager API
│   │   ├── wazuh_indexer_client.py # Wazuh Indexer API
│   │   ├── data_collector.py       # Data collectors
│   │   └── csv_converter.py        # CSV conversion
│   │
│   ├── documentation/               # All documentation (40+ files)
│   ├── standards/                   # Engineering standards
│   ├── steering/                    # Steering files
│   ├── pyproject.toml               # Poetry dependencies
│   ├── Makefile                     # Make commands
│   └── README.md                    # Main README
│
└── njs_shared_data/                 # All pipeline data
    ├── config/                      # Runtime config
    ├── data/
    │   ├── collected/               # Raw Wazuh data
    │   ├── extracted/               # Normalized data
    │   ├── aggregated/              # Intermediate
    │   ├── aggregated_core/         # Final output
    │   └── pii_scan_results/        # PII validation
    ├── logs/                        # All logs
    └── pipeline/                    # State markers
```


## 🔄 Data Flow

### Complete Pipeline Flow

```
1. COLLECT DATA (scripts/main.py)
   ├─ Source: Wazuh Manager API + Indexer
   ├─ Collects: Agents, Hosts, Packages, Hardware, CPE, FIM, Groups, Vulnerabilities
   └─ Output: njs_shared_data/data/collected/YYYYMMDD_HHMMSS/
      ├── agents_manager/All_Agents.json
      ├── host/agent_XXX/Syscollector_OS_Info_XXX.json
      ├── packages/agent_XXX/Syscollector_Packages_XXX.json
      ├── hardware/agent_XXX/Syscollector_Hardware_XXX.json
      ├── cpe/agent_XXX/Syscollector_CPE_XXX.json
      ├── fim/agent_XXX/File_Integrity_Monitoring_XXX.json
      ├── groups/group_XXX/Group_Agents_XXX.json
      └── vulnerabilities/agent_XXX/Vulnerabilities_XXX.json

2. EXTRACT DATA (scripts/extract_data.py)
   ├─ Source: data/collected/
   ├─ Process: Normalize and structure data
   └─ Output: njs_shared_data/data/extracted/YYYYMMDD_HHMMSS/
      ├── agents.json
      ├── hosts.json
      ├── packages.json
      ├── hardware.json
      ├── vulnerabilities.json
      └── relationships.json

3. BUILD NODE GRAPH (scripts/build_node_graph.py)
   ├─ Source: data/extracted/
   ├─ Process: Create Neo4j nodes and relationships
   └─ Output: Neo4j node_kg database
      ├── Nodes: Agent, Host, Software, Hardware, Vulnerability, AssetGroup
      └── Relationships: RUNS_ON, HAS_SOFTWARE, HAS_HARDWARE, HAS_VULNERABILITY, etc.

4. AGGREGATE DATA (scripts/aggregate_data_v2.py)
   ├─ Source: Neo4j node_kg database
   ├─ Process: Privacy-preserving aggregations
   └─ Output: njs_shared_data/data/aggregated_core/YYYYMMDD_HHMMSS/
      ├── core_aggregation.json       ← FINAL OUTPUT FOR CORE GRAPH
      ├── exposure_surface.json
      ├── sensitivity_surface.json
      ├── outcome_metrics.json
      └── summary_report.txt

5. DETECT PII (scripts/detect_pii.py)
   ├─ Source: data/aggregated_core/
   ├─ Process: Scan for PII/PCI using Presidio
   └─ Output: njs_shared_data/data/pii_scan_results/YYYYMMDD_HHMMSS/
      ├── pii_scan_results.json
      ├── pii_scan_summary.txt
      └── pii_findings_detail.json
      └─ Validation: MUST pass (no PII) to continue

6. BUILD CORE GRAPH (scripts/build_core_graph.py)
   ├─ Source: data/aggregated_core/
   ├─ Process: Create bank-level graph
   └─ Output: Neo4j core database
      └── Nodes: NJS_Bank (with aggregated metrics)
```

### Data Types Collected

| Data Type | Source | Endpoint | Description |
|-----------|--------|----------|-------------|
| Agents | Manager | `/agents` | All registered agents |
| Host/OS | Manager | `/syscollector/{agent_id}/os` | Operating system info |
| Packages | Manager | `/syscollector/{agent_id}/packages` | Installed software |
| Hardware | Manager | `/syscollector/{agent_id}/hardware` | Hardware details |
| CPE | Manager | `/syscollector/{agent_id}/packages` | Platform enumeration |
| Groups | Manager | `/groups` | Agent groups |
| FIM | Manager | `/syscheck/{agent_id}` | File integrity monitoring |
| Vulnerabilities | Indexer | `/wazuh-alerts-4.x/_search` | CVE vulnerabilities |


## 🗄️ Data Models

### Node Graph Schema (node_kg database)

**Nodes:**
- `Agent`: Wazuh agents (endpoints)
- `Host`: Operating systems and hosts
- `Software`: Installed packages/software
- `Hardware`: Hardware components
- `Vulnerability`: CVE vulnerabilities
- `AssetGroup`: Agent groups

**Relationships:**
- `Agent -[RUNS_ON]-> Host`
- `Agent -[HAS_SOFTWARE]-> Software`
- `Agent -[HAS_HARDWARE]-> Hardware`
- `Agent -[HAS_VULNERABILITY]-> Vulnerability`
- `Agent -[BELONGS_TO]-> AssetGroup`
- `Software -[HAS_VULNERABILITY]-> Vulnerability`

### Core Graph Schema (core database)

**Nodes:**
- `NJS_Bank`: Bank-level aggregated node

**Properties:**
- Exposure Surface metrics
- Sensitivity Surface metrics
- Outcome metrics
- Aggregated counts and statistics

### Aggregation Structure

```json
{
  "bank_id": "bank_001",
  "timestamp": "2026-02-17T10:30:00Z",
  "exposure_surface": {
    "total_agents": 150,
    "total_hosts": 150,
    "os_distribution": {...},
    "vulnerability_summary": {...}
  },
  "sensitivity_surface": {
    "critical_assets": 25,
    "high_value_systems": 45,
    "data_classification": {...}
  },
  "outcome_metrics": {
    "security_score": 85.5,
    "compliance_score": 92.3,
    "risk_level": "medium"
  }
}
```

## 🔐 Privacy & Security

### Privacy-Preserving Aggregation

The pipeline implements privacy-preserving techniques:

1. **Aggregation**: Individual node data aggregated to bank level
2. **Anonymization**: No individual agent identifiers in Core Graph
3. **Statistical Summaries**: Counts, distributions, percentages only
4. **PII Detection**: Mandatory validation before Core Graph

### PII/PCI Detection

Uses Microsoft Presidio for detecting:
- Personal Identifiable Information (PII)
- Payment Card Industry (PCI) data
- Sensitive data patterns

**Validation Gate**: Pipeline stops if PII/PCI detected in aggregated data.

### Security Measures

- JWT authentication for Wazuh API
- SSL/TLS for all connections
- Credentials in `.env` (never committed)
- Read-only config mounts in Docker
- Isolated network for containers


## 🛠️ Technology Stack

### Core Technologies

- **Language**: Python 3.12
- **Dependency Management**: Poetry
- **Database**: Neo4j 5.15.0
- **Containerization**: Docker & Docker Compose
- **Security Monitoring**: Wazuh 4.12.2

### Python Dependencies

**Core:**
- `requests` - HTTP client for Wazuh API
- `pyyaml` - Configuration management
- `neo4j` - Neo4j driver
- `pandas` - Data manipulation
- `python-dotenv` - Environment variables

**PII Detection:**
- `presidio-analyzer` - PII detection
- `presidio-anonymizer` - Data anonymization
- `spacy` - NLP for entity recognition

**Development:**
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

### Infrastructure

- **Docker Compose**: Multi-container orchestration
- **Neo4j Container**: Graph database
- **Pipeline Container**: Python application
- **Shared Volumes**: Data persistence

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Wazuh Configuration
WAZUH_API_URL=https://wazuh-server:55000
WAZUH_API_USERNAME=wazuh-wui
WAZUH_API_PASSWORD=SecretPassword

# Neo4j Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme
NEO4J_NODE_DATABASE=node_kg
NEO4J_CORE_DATABASE=core

# Bank Configuration
BANK_ID=bank_001

# Pipeline Configuration
PIPELINE_MODE=run-once
LOG_LEVEL=INFO
SHARED_DATA_PATH=/shared_data
```

### Configuration Files

1. **config/config.yaml**: Main configuration
   - Wazuh connection settings
   - Collector configuration
   - Retry settings

2. **config/paths_config.yaml**: Data paths
   - Input/output directories
   - File patterns
   - Data sources

3. **config/aggregation_config.yaml**: Aggregation rules
   - Aggregation types
   - Privacy settings
   - Metric definitions

4. **config/neo4j_config.yaml**: Neo4j settings
   - Database names
   - Connection settings
   - Query configurations

5. **config/graph_config.yaml**: Graph schema
   - Node types
   - Relationship types
   - Property definitions


## 🚀 Deployment

### Docker Deployment

**Services:**
- `neo4j`: Neo4j database (port 7474, 7687)
- `pipeline`: NJS pipeline container

**Volumes:**
- `neo4j_data`: Neo4j database files
- `neo4j_logs`: Neo4j logs
- `neo4j_import`: Import directory
- `../njs_shared_data`: Shared data (host mount)

**Network:**
- `njs_network`: Bridge network for inter-container communication

### Deployment Commands

```bash
# First time setup
make setup          # Create directories, copy .env
make build          # Build Docker images
make start          # Start services

# Daily operations
make logs           # View logs
make status         # Check status
make stop           # Stop services
make restart        # Restart services

# Maintenance
make backup         # Backup Neo4j databases
make clean          # Clean old data
make reset          # Reset everything
```

### Deployment Scripts

1. **start_all.sh**: Complete setup and start
2. **start.sh**: Start existing services
3. **stop.sh**: Stop services
4. **logs.sh**: View logs
5. **status.sh**: Check status
6. **backup.sh**: Backup databases

## 📊 Monitoring & Logging

### Log Files

All logs in `njs_shared_data/logs/`:
- `pipeline.log` - Main orchestrator log
- `collect_data.log` - Data collection log
- `extract_data.log` - Extraction log
- `build_node_graph.log` - Node graph log
- `aggregate_data.log` - Aggregation log
- `detect_pii.log` - PII detection log
- `build_core_graph.log` - Core graph log

### Pipeline State

State markers in `njs_shared_data/pipeline/`:
- `.done` - Pipeline completion marker
- `node_graph.done` - Node graph complete
- `core_graph.done` - Core graph complete
- Per-step `.done` files in data directories

### Monitoring Commands

```bash
# View logs
make logs
tail -f ../njs_shared_data/logs/pipeline.log

# Check status
make status
cat ../njs_shared_data/pipeline/.done

# Check disk usage
du -sh ../njs_shared_data/

# View Neo4j
http://localhost:7474
```


## 🔄 Pipeline Execution

### Sequential Execution

The pipeline uses sequential execution with validation gates:

1. Each step validates prerequisites before running
2. Each step validates output after completion
3. `.done` files mark successful completion
4. Pipeline stops on validation failure

### Validation Gates

**Prerequisites Check:**
- Verifies previous step completed
- Checks for required `.done` files
- Validates input data exists

**Output Validation:**
- Verifies output files created
- Checks data structure
- Validates data quality

**PII Validation Gate:**
- Scans aggregated data for PII/PCI
- MUST pass (no PII) to continue
- Pipeline stops if PII detected

### Retry Logic

All collectors implement retry logic:
- **Max Retries**: 3 (configurable)
- **Retry Delay**: 5 seconds (exponential backoff)
- **Error Handling**: Graceful failure with logging

### Execution Modes

1. **Run-Once**: Execute pipeline once and exit
2. **Scheduled**: Run on schedule (cron-based)
3. **Single Step**: Run individual pipeline step

## 📈 Performance

### Scalability

- **Agents**: Tested with 150+ agents
- **Data Volume**: Handles 10,000+ packages per agent
- **Vulnerabilities**: Processes 1,000+ CVEs
- **Execution Time**: ~15-30 minutes for complete pipeline

### Optimization

- Parallel agent data collection
- Pagination for large datasets
- Efficient Neo4j queries
- Incremental data processing

### Resource Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 50 GB

**Recommended:**
- CPU: 8 cores
- RAM: 16 GB
- Disk: 100 GB


## 📚 Documentation

### Complete Documentation Library

**Architecture & Design:**
- `ARCHITECTURE.md` - System architecture
- `ARCHITECTURE_DIAGRAM.md` - Visual diagrams
- `DAY0_NODE_GRAPH_CREATION.md` - Node graph design
- `FOLDER_STRUCTURE.md` - Project organization

**Data & Storage:**
- `SHARED_DATA_STRUCTURE.md` - Data directory structure
- `DATA_STRUCTURE_DIAGRAM.md` - Visual data flow
- `DATA_REORGANIZATION_COMPLETE.md` - Data organization
- `ENDPOINTS_REFERENCE.md` - Wazuh API endpoints

**Deployment & Operations:**
- `DEPLOYMENT_SCRIPTS_GUIDE.md` - Deployment scripts
- `SERVER_DEPLOYMENT_GUIDE.md` - Server deployment
- `DOCKER_IMPLEMENTATION_SUMMARY.md` - Docker setup
- `QUICK_START_SERVER.md` - Quick start guide

**Features:**
- `AGGREGATION.md` - Aggregation overview
- `AGGREGATION_IMPLEMENTATION.md` - Implementation details
- `PII_DETECTION.md` - PII detection
- `CPE_COLLECTION.md` - CPE collection
- `PRIVACY_PRESERVING_AGGREGATION.md` - Privacy features

**Configuration:**
- `DEPENDENCIES.md` - Dependencies guide
- `POETRY_MIGRATION.md` - Poetry setup
- `QUICK_REFERENCE.md` - Quick reference

**Project Management:**
- `PROJECT_CONTEXT.md` - This file
- `PROJECT_ORGANIZATION_COMPLETE.md` - Organization summary
- `FINAL_SUMMARY.md` - Implementation summary

## 🎯 Key Features

### 1. Automated Data Collection
- Collects from Wazuh Manager and Indexer
- 8 data types: Agents, Hosts, Packages, Hardware, CPE, FIM, Groups, Vulnerabilities
- Retry logic for reliability
- Timestamped sessions

### 2. Node Knowledge Graph
- Detailed node-level graph in Neo4j
- 6 node types, 7+ relationship types
- Rich property data
- Cypher query support

### 3. Privacy-Preserving Aggregation
- Bank-level aggregations
- No individual identifiers
- Statistical summaries only
- Configurable aggregation rules

### 4. PII/PCI Detection
- Microsoft Presidio integration
- Scans aggregated data
- Validation gate before Core Graph
- Detailed findings reports

### 5. Core Graph
- Bank-level security metrics
- Exposure surface analysis
- Sensitivity surface metrics
- Outcome metrics

### 6. Docker Deployment
- Complete containerization
- One-command deployment
- Data persistence
- Easy scaling

### 7. Configuration-Driven
- All settings in YAML files
- Environment variables for secrets
- No hardcoded values
- Easy customization

### 8. Comprehensive Logging
- Structured logging
- Per-step log files
- Error tracking
- Audit trail


## 🔧 Development

### Local Development Setup

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Download spaCy model
poetry run python -m spacy download en_core_web_lg

# Activate environment
poetry shell

# Run scripts
python scripts/main.py
python scripts/detect_pii.py
```

### Project Standards

**Maintainability Standards** (`standards/mantainability.md`):
- Parameter-driven configuration
- Structured logging
- Error codes
- Validation at each step
- No hardcoded values

**Code Style:**
- Black formatting (line length: 100)
- Type hints where applicable
- Docstrings for all functions
- Comprehensive error handling

### Testing

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov

# Run specific test
poetry run pytest tests/test_collector.py
```

## 🤝 Integration Points

### Wazuh Integration
- REST API for Manager
- OpenSearch API for Indexer
- JWT authentication
- SSL/TLS connections

### Neo4j Integration
- Bolt protocol
- Cypher queries
- Multiple databases (node_kg, core)
- Transaction support

### External Systems
- Can export aggregated data
- API-ready for Core Graph consumption
- CSV export support
- JSON data format

## 🔍 Troubleshooting

### Common Issues

**1. Connection Failures**
```bash
# Check Wazuh connection
curl -k -u username:password https://wazuh-server:55000/

# Check Neo4j connection
docker exec -it orbit-neo4j cypher-shell
```

**2. Data Collection Errors**
```bash
# View collection logs
tail -f ../njs_shared_data/logs/collect_data.log

# Check retry attempts
grep "retry" ../njs_shared_data/logs/collect_data.log
```

**3. Pipeline Failures**
```bash
# Check pipeline status
make status

# View error logs
grep ERROR ../njs_shared_data/logs/pipeline.log

# Check validation failures
grep "VALIDATE" ../njs_shared_data/logs/pipeline.log
```

**4. Disk Space Issues**
```bash
# Check disk usage
du -sh ../njs_shared_data/

# Clean old data
make clean

# Manual cleanup
find ../njs_shared_data/data/collected/ -mtime +7 -delete
```


## 📋 Quick Reference

### Essential Commands

```bash
# Setup
make setup          # Initial setup
make build          # Build images
make start          # Start services

# Operations
make logs           # View logs
make status         # Check status
make stop           # Stop services
make restart        # Restart

# Pipeline
make run            # Run pipeline
make shell          # Container shell

# Maintenance
make backup         # Backup databases
make clean          # Clean old data
make reset          # Reset everything
```

### Important Paths

```bash
# Project
orbit_node/                 # Project code

# Shared Data
../njs_shared_data/         # All data
../njs_shared_data/data/collected/      # Raw data
../njs_shared_data/data/aggregated_core/# Final output
../njs_shared_data/logs/    # Logs

# Configuration
config/config.yaml          # Main config
.env                        # Secrets

# Docker
docker/docker-compose.yml   # Services
```

### Key Files

```bash
# Configuration
.env                        # Environment variables
config/config.yaml          # Main configuration
config/paths_config.yaml    # Data paths

# Scripts
scripts/orchestrator.py     # Main orchestrator
scripts/main.py             # Data collection
scripts/aggregate_data_v2.py# Aggregation
scripts/detect_pii.py       # PII detection

# Deployment
deployment_scripts/start_all.sh  # Complete setup
Makefile                    # Make commands
```

### Data Locations

```bash
# Collected Data (Raw)
../njs_shared_data/data/collected/YYYYMMDD_HHMMSS/

# Extracted Data (Normalized)
../njs_shared_data/data/extracted/YYYYMMDD_HHMMSS/

# Aggregated Core (Final Output)
../njs_shared_data/data/aggregated_core/YYYYMMDD_HHMMSS/

# PII Scan Results
../njs_shared_data/data/pii_scan_results/YYYYMMDD_HHMMSS/

# Logs
../njs_shared_data/logs/pipeline.log
```


## 🎓 Learning Resources

### Getting Started

1. **Read**: `README.md` - Project overview
2. **Setup**: `QUICK_START_NEW_STRUCTURE.md` - Quick start
3. **Deploy**: `SERVER_DEPLOYMENT_GUIDE.md` - Deployment guide
4. **Understand**: `ARCHITECTURE.md` - System architecture

### Deep Dive

1. **Data Flow**: `DATA_STRUCTURE_DIAGRAM.md` - Visual data flow
2. **Aggregation**: `AGGREGATION_IMPLEMENTATION.md` - How aggregation works
3. **Privacy**: `PRIVACY_PRESERVING_AGGREGATION.md` - Privacy features
4. **PII Detection**: `PII_DETECTION.md` - PII scanning

### Operations

1. **Deployment**: `DEPLOYMENT_SCRIPTS_GUIDE.md` - Script usage
2. **Monitoring**: `QUICK_REFERENCE.md` - Common commands
3. **Troubleshooting**: This file - Common issues section

## 🔮 Future Enhancements

### Planned Features

1. **Scheduled Execution**: Cron-based pipeline scheduling
2. **Incremental Updates**: Update only changed data
3. **Multi-Bank Support**: Handle multiple banks in one instance
4. **Advanced Analytics**: ML-based security insights
5. **Real-time Monitoring**: Live dashboard
6. **API Endpoints**: REST API for data access
7. **Alert System**: Automated alerting on anomalies
8. **Report Generation**: Automated security reports

### Potential Improvements

1. **Performance**: Parallel processing, caching
2. **Scalability**: Kubernetes deployment
3. **Monitoring**: Prometheus/Grafana integration
4. **Testing**: Comprehensive test suite
5. **Documentation**: Interactive tutorials
6. **CI/CD**: Automated testing and deployment

## 📞 Support & Contact

### Getting Help

1. **Documentation**: Check `documentation/` folder
2. **Logs**: Review `../njs_shared_data/logs/`
3. **Status**: Run `make status`
4. **Issues**: Check common issues in this file

### Project Information

- **Organization**: NJSecure (NJS)
- **Project**: Orbit Node Pipeline
- **Version**: 1.0.0
- **Status**: Production Ready
- **Last Updated**: February 17, 2026

## 📄 License

Copyright © 2026 NJS Team  
Proprietary - All Rights Reserved

---

## Summary

NJS Orbit Node is a comprehensive, privacy-preserving security data pipeline that:

✅ Collects security data from Wazuh (8 data types)  
✅ Creates detailed Node Knowledge Graphs in Neo4j  
✅ Performs privacy-preserving aggregations  
✅ Detects and prevents PII/PCI data leakage  
✅ Builds bank-level Core Graphs for analysis  
✅ Deploys easily with Docker  
✅ Provides comprehensive logging and monitoring  
✅ Follows maintainability and security standards  

The system is production-ready, fully documented, and designed for scalability and reliability.

---

**Document Version**: 1.0.0  
**Last Updated**: February 17, 2026  
**Maintained By**: NJS Team
