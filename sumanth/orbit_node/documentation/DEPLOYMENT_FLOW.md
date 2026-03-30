# DataLegos Deployment Flow

Visual guide showing how to deploy DataLegos on a server.

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Production Server                         │
│                     (Ubuntu/CentOS/RHEL)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  /opt/datalegos/orbit_node/          (Application Code)         │
│  ├── config/                         Configuration files        │
│  ├── docker/                         Docker setup               │
│  ├── scripts/                        Pipeline scripts           │
│  ├── graph_builder/                  Graph modules              │
│  └── .env                            Secrets (not in git)       │
│                                                                   │
│  /opt/datalegos_shared_data/         (Data Storage)             │
│  ├── collected_data/                 Raw Wazuh data             │
│  ├── extracted_data/                 Normalized data            │
│  ├── aggregated_data_core/           Aggregations               │
│  ├── pii_scan_results/               PII scan reports           │
│  ├── logs/                           Pipeline logs              │
│  └── pipeline/                       .done markers              │
│                                                                   │
│  Docker Containers:                                              │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │   Neo4j          │  │   Pipeline       │                    │
│  │   Container      │  │   Container      │                    │
│  │                  │  │                  │                    │
│  │  - node_kg DB    │  │  - Python 3.12   │                    │
│  │  - core DB       │  │  - Presidio      │                    │
│  │  - Port 7474     │  │  - Scripts       │                    │
│  │  - Port 7687     │  │  - Orchestrator  │                    │
│  └──────────────────┘  └──────────────────┘                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
   Neo4j Browser                        Wazuh Manager
   (localhost:7474)                     (API: port 55000)
```

## Deployment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PROCESS                            │
└─────────────────────────────────────────────────────────────────┘

1. SERVER PREPARATION
   ├── Install Docker Engine
   ├── Install Docker Compose
   ├── Create directories
   └── Configure firewall (optional)
         │
         ▼
2. APPLICATION SETUP
   ├── Clone git repository
   ├── Copy .env.example to .env
   ├── Configure environment variables
   │   ├── Wazuh API credentials
   │   ├── Neo4j password
   │   ├── Bank ID
   │   └── Pipeline mode
   └── Set file permissions
         │
         ▼
3. BUILD & DEPLOY
   ├── Build Docker images
   │   ├── Pull Neo4j image
   │   └── Build pipeline image
   ├── Start services
   │   ├── Start Neo4j container
   │   └── Start pipeline container
   └── Verify services running
         │
         ▼
4. PIPELINE EXECUTION
   ├── Step 1: Collect Data
   │   └── Fetch from Wazuh API
   ├── Step 2: Extract Data
   │   └── Normalize and stage
   ├── Step 3: Build Node Graph
   │   └── Create detailed graph
   ├── Step 4: Aggregate Data
   │   └── Privacy-preserving aggregation
   ├── Step 5: Detect PII
   │   └── Scan with Presidio
   └── Step 6: Build Core Graph
       └── Create bank-level graph
         │
         ▼
5. VERIFICATION
   ├── Check pipeline completion
   ├── Verify Neo4j data
   ├── Review PII scan results
   └── Access Neo4j browser
         │
         ▼
6. MAINTENANCE
   ├── Set up backups
   ├── Configure monitoring
   ├── Schedule pipeline runs
   └── Clean old data
```

## Quick Deployment Commands

```bash
# ============================================================================
# STEP 1: SERVER PREPARATION
# ============================================================================

# Install Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl start docker
sudo systemctl enable docker

# Create directories
sudo mkdir -p /opt/datalegos
sudo mkdir -p /opt/datalegos_shared_data

# ============================================================================
# STEP 2: APPLICATION SETUP
# ============================================================================

# Clone repository
cd /opt/datalegos
git clone https://github.com/your-org/orbit_node.git
cd orbit_node

# Configure
cp .env.example .env
nano .env  # Edit with your settings

# ============================================================================
# STEP 3: BUILD & DEPLOY
# ============================================================================

# Build images
cd docker
docker compose build

# Start services
docker compose up -d

# ============================================================================
# STEP 4: MONITOR EXECUTION
# ============================================================================

# View logs
docker compose logs -f pipeline

# Check status
docker compose ps

# ============================================================================
# STEP 5: VERIFY
# ============================================================================

# Check completion
docker compose exec pipeline cat /shared_data/pipeline/.done

# Access Neo4j
# Browser: http://your-server-ip:7474

# ============================================================================
# STEP 6: MAINTENANCE
# ============================================================================

# Backup
docker compose exec neo4j neo4j-admin database dump node_kg \
  --to=/var/lib/neo4j/import/backup.dump

# Clean old data
find /opt/datalegos_shared_data/collected_data/ -mtime +30 -delete
```

## Using Makefile (Recommended)

```bash
# ============================================================================
# QUICK DEPLOYMENT WITH MAKEFILE
# ============================================================================

# Setup
make setup
# Edit .env with your configuration

# Build and start
make build
make up

# Monitor
make logs

# Check status
make status

# Maintenance
make backup
make clean-data
```

## Production Deployment

```bash
# ============================================================================
# PRODUCTION MODE
# ============================================================================

# Use production configuration
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or with Makefile
make prod-up

# Production features:
# - Increased memory limits
# - Resource reservations
# - No exposed ports (use SSH tunnel)
# - Production restart policies
```

## Pipeline Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE EXECUTION                            │
└─────────────────────────────────────────────────────────────────┘

STEP 1: COLLECT DATA
├── Connect to Wazuh API
├── Fetch agents data
├── Fetch syscollector data
├── Fetch vulnerabilities
├── Save to collected_data/
└── Create .done marker
      │
      ▼ (Validation: JSON files exist)
      │
STEP 2: EXTRACT DATA
├── Read collected data
├── Normalize structure
├── Extract nodes and relationships
├── Save to extracted_data/
└── Create .done marker
      │
      ▼ (Validation: agents.json, hosts.json exist)
      │
STEP 3: BUILD NODE GRAPH
├── Connect to Neo4j (node_kg DB)
├── Create constraints
├── Insert nodes (agents, hosts, software, etc.)
├── Create relationships
└── Create .done marker
      │
      ▼ (Validation: Nodes created in Neo4j)
      │
STEP 4: AGGREGATE DATA
├── Query Node graph
├── Calculate aggregations
│   ├── Software counts
│   ├── OS distribution
│   ├── Vulnerability exposure
│   └── Hardware profiles
├── Save to aggregated_data_core/
└── Create .done marker
      │
      ▼ (Validation: core_aggregation.json exists)
      │
STEP 5: DETECT PII
├── Load aggregated data
├── Scan with Presidio
├── Apply false positive filters
├── Generate report
├── Save to pii_scan_results/
└── Create .done marker
      │
      ▼ (Validation: No PII detected)
      │
STEP 6: BUILD CORE GRAPH
├── Connect to Neo4j (core DB)
├── Create NJS_Bank node
├── Create NJS_ScanEvent node
├── Create aggregated nodes
│   ├── NJS_Software
│   ├── NJS_OperatingSystem
│   └── Link to existing CVE nodes
├── Create relationships with counts
└── Create .done marker
      │
      ▼ (Validation: NJS_Bank node exists)
      │
✅ PIPELINE COMPLETE
```

## Data Flow

```
Wazuh Manager
      │
      │ (API calls)
      ▼
collected_data/
      │
      │ (normalize)
      ▼
extracted_data/
      │
      │ (insert)
      ▼
Neo4j (node_kg)
      │
      │ (aggregate)
      ▼
aggregated_data_core/
      │
      │ (scan)
      ▼
pii_scan_results/
      │
      │ (if clean)
      ▼
Neo4j (core)
```

## Access Points

```
┌─────────────────────────────────────────────────────────────────┐
│                      ACCESS POINTS                               │
└─────────────────────────────────────────────────────────────────┘

Neo4j Browser (Development)
├── URL: http://server-ip:7474
├── Username: neo4j
├── Password: (from .env)
└── Databases: node_kg, core

Neo4j Browser (Production - SSH Tunnel)
├── SSH: ssh -L 7474:localhost:7474 user@server
├── URL: http://localhost:7474
├── Username: neo4j
└── Password: (from .env)

Pipeline Logs
├── All logs: docker compose logs -f
├── Pipeline: docker compose logs -f pipeline
├── Neo4j: docker compose logs -f neo4j
└── Step logs: /opt/datalegos_shared_data/logs/

Data Directories
├── Collected: /opt/datalegos_shared_data/collected_data/
├── Extracted: /opt/datalegos_shared_data/extracted_data/
├── Aggregated: /opt/datalegos_shared_data/aggregated_data_core/
└── PII Scans: /opt/datalegos_shared_data/pii_scan_results/

Shell Access
├── Pipeline: docker compose exec pipeline /bin/bash
└── Neo4j: docker compose exec neo4j /bin/bash
```

## Troubleshooting Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TROUBLESHOOTING                               │
└─────────────────────────────────────────────────────────────────┘

Pipeline Fails?
├── Check logs: docker compose logs pipeline
├── Check step log: cat /shared_data/logs/<step>.log
├── Check orchestrator: cat /shared_data/logs/pipeline.log
└── Restart step: docker compose exec pipeline python /app/scripts/orchestrator.py --step <name>

Neo4j Issues?
├── Check status: docker compose ps neo4j
├── Check logs: docker compose logs neo4j
├── Test connection: docker compose exec neo4j cypher-shell -u neo4j -p <pass> "RETURN 1"
└── Restart: docker compose restart neo4j

Wazuh API Issues?
├── Test from container: docker compose exec pipeline curl -k -u <user>:<pass> <url>
├── Check network: docker compose exec pipeline ping wazuh-manager
└── Verify credentials in .env

Disk Space Issues?
├── Check usage: df -h
├── Check Docker: docker system df
├── Clean old data: make clean-data
└── Clean Docker: docker system prune -a

Permission Issues?
├── Fix ownership: chown -R root:root /opt/datalegos_shared_data
└── Fix permissions: chmod -R 755 /opt/datalegos_shared_data
```

## Maintenance Schedule

```
Daily:
├── Check pipeline execution
├── Review logs for errors
└── Monitor disk space

Weekly:
├── Backup Neo4j databases
├── Clean old data (30+ days)
└── Review PII scan results

Monthly:
├── Update Docker images
├── Update application code
├── Review security settings
└── Test backup restoration

Quarterly:
├── Review and update documentation
├── Audit access logs
└── Performance optimization
```

## Quick Reference

```bash
# Start
make up

# Stop
make down

# Logs
make logs

# Status
make status

# Backup
make backup

# Clean
make clean-data

# Update
make update

# Production
make prod-up
```

---

**For detailed instructions, see:**
- [Quick Start Guide](QUICK_START_SERVER.md)
- [Complete Server Deployment Guide](docs/deployment/SERVER_DEPLOYMENT_GUIDE.md)
- [Docker Deployment Guide](docs/deployment/DOCKER_DEPLOYMENT.md)
