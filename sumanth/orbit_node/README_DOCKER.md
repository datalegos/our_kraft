# DataLegos Docker Deployment Guide

## Overview

This guide explains how to deploy the DataLegos pipeline using Docker containers. The pipeline runs sequentially with validation gates and creates both Node KG and Core Graph databases.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Neo4j      │  │   Pipeline   │  │   Shared     │      │
│  │   Container  │  │   Container  │  │   Volume     │      │
│  │              │  │              │  │              │      │
│  │  - node_kg   │  │  - Python    │  │  - configs   │      │
│  │  - core      │  │  - Scripts   │  │  - data      │      │
│  │              │  │  - Presidio  │  │  - logs      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum (8GB recommended)
- 20GB disk space

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/orbit_node.git
cd orbit_node
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
nano .env
```

Required configuration:
- `WAZUH_API_URL` - Your Wazuh Manager URL
- `WAZUH_API_USERNAME` - Wazuh API username
- `WAZUH_API_PASSWORD` - Wazuh API password
- `NEO4J_PASSWORD` - Neo4j database password
- `BANK_ID` - Unique identifier for your organization

### 3. Create Shared Data Directory

The shared data directory will be created automatically at `../orbit_node_shared_data` (sibling to project directory).

```bash
# Verify it will be created in the right place
ls -la ../
```

### 4. Start Services

```bash
# Start all services
cd docker
docker-compose up -d

# View logs
docker-compose logs -f pipeline

# Check status
docker-compose ps
```

## Pipeline Flow

The pipeline runs sequentially with validation at each step:

```
1. COLLECT DATA
   ├─ Run: main.py
   ├─ Output: ../orbit_node_shared_data/collected_data/{timestamp}/
   ├─ Validate: JSON files exist
   └─ Create: .done file

2. EXTRACT DATA
   ├─ Validate: collected_data/.done exists
   ├─ Run: extract_data.py
   ├─ Output: ../orbit_node_shared_data/extracted_data/{timestamp}/
   ├─ Validate: agents.json, hosts.json exist
   └─ Create: .done file

3. BUILD NODE GRAPH
   ├─ Validate: extracted_data/.done exists
   ├─ Run: build_node_graph.py
   ├─ Output: Neo4j node_kg database
   ├─ Validate: Nodes created
   └─ Create: .done file

4. AGGREGATE DATA
   ├─ Validate: node_graph/.done exists
   ├─ Run: aggregate_data_v2.py
   ├─ Output: ../orbit_node_shared_data/aggregated_data_core/{timestamp}/
   ├─ Validate: core_aggregation.json exists
   └─ Create: .done file

5. DETECT PII
   ├─ Validate: aggregated_data_core/.done exists
   ├─ Run: detect_pii.py
   ├─ Output: ../orbit_node_shared_data/pii_scan_results/{timestamp}/
   ├─ Validate: No PII detected
   └─ Create: .done file

6. BUILD CORE GRAPH
   ├─ Validate: pii_scan/.done exists
   ├─ Run: build_core_graph.py
   ├─ Output: Neo4j core database
   ├─ Validate: NJS_Bank node exists
   └─ Create: .done file
```

## Usage

### Run Complete Pipeline

```bash
# Start pipeline (run-once mode)
docker-compose up -d

# Monitor progress
docker-compose logs -f pipeline

# Check completion
docker-compose exec pipeline cat /shared_data/pipeline/.done
```

### Run Single Step

```bash
# Run specific step
docker-compose exec pipeline python /app/scripts/orchestrator.py --step collect_data

# Available steps:
# - collect_data
# - extract_data
# - build_node_graph
# - aggregate_data
# - detect_pii
# - build_core_graph
```

### Access Neo4j

```bash
# Open browser
http://localhost:7474

# Login with credentials from .env
Username: neo4j
Password: <NEO4J_PASSWORD>

# Or use cypher-shell
docker-compose exec neo4j cypher-shell -u neo4j -p <password>
```

### View Logs

```bash
# Pipeline logs
docker-compose logs -f pipeline

# Neo4j logs
docker-compose logs -f neo4j

# Specific step logs
docker-compose exec pipeline cat /shared_data/logs/collect_data.log
docker-compose exec pipeline cat /shared_data/logs/detect_pii.log
```

### Access Shell

```bash
# Pipeline container shell
docker-compose exec pipeline /bin/bash

# Neo4j container shell
docker-compose exec neo4j /bin/bash
```

## Data Persistence

All data is stored in `../orbit_node_shared_data/` (sibling to project):

```
../orbit_node_shared_data/
├── collected_data/          # Raw data from Wazuh
├── extracted_data/          # Normalized data
├── aggregated_data_core/    # Privacy-preserving aggregations
├── pii_scan_results/        # PII scan reports
├── logs/                    # Pipeline logs
└── pipeline/                # .done marker files
```

Neo4j data is stored in Docker volumes:
- `neo4j_data` - Database files
- `neo4j_logs` - Neo4j logs

## Scheduled Runs (Future)

To enable scheduled pipeline runs:

1. Edit `docker-compose.yml`:
   ```yaml
   pipeline:
     # Uncomment this line:
     command: ["run-scheduled"]
   ```

2. Set schedule in `.env`:
   ```bash
   PIPELINE_MODE=scheduled
   PIPELINE_SCHEDULE=0 2 * * *  # Daily at 2 AM
   ```

3. Restart services:
   ```bash
   docker-compose restart pipeline
   ```

## Troubleshooting

### Pipeline Fails at Step

```bash
# Check logs for the failed step
docker-compose exec pipeline cat /shared_data/logs/<step_name>.log

# Check orchestrator log
docker-compose exec pipeline cat /shared_data/logs/pipeline.log

# Restart from failed step
docker-compose exec pipeline python /app/scripts/orchestrator.py --step <step_name>
```

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker-compose ps neo4j

# Check Neo4j logs
docker-compose logs neo4j

# Test connection
docker-compose exec neo4j cypher-shell -u neo4j -p <password> "RETURN 1"
```

### Shared Data Not Mounted

```bash
# Verify shared data directory exists
ls -la ../orbit_node_shared_data/

# Check mount in container
docker-compose exec pipeline ls -la /shared_data/

# Recreate containers
docker-compose down
docker-compose up -d
```

### PII Scan Fails

```bash
# Check PII scan results
docker-compose exec pipeline cat /shared_data/pii_scan_results/latest/pii_scan_summary.txt

# Review detailed findings
docker-compose exec pipeline cat /shared_data/pii_scan_results/latest/detailed_findings.json

# Adjust false positive filters in config/aggregation_config.yaml
```

## Maintenance

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes Neo4j data)
docker-compose down -v
```

### Update Pipeline

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose up -d
```

### Clean Old Data

```bash
# Remove old collected data (keep last 7 days)
find ../orbit_node_shared_data/collected_data/ -type d -mtime +7 -exec rm -rf {} +

# Remove old PII scan results
find ../orbit_node_shared_data/pii_scan_results/ -type d -mtime +30 -exec rm -rf {} +
```

### Backup Neo4j

```bash
# Backup Neo4j data
docker-compose exec neo4j neo4j-admin database dump node_kg --to=/var/lib/neo4j/import/node_kg_backup.dump
docker-compose exec neo4j neo4j-admin database dump core --to=/var/lib/neo4j/import/core_backup.dump

# Copy backups to host
docker cp datalegos_neo4j:/var/lib/neo4j/import/node_kg_backup.dump ./backups/
docker cp datalegos_neo4j:/var/lib/neo4j/import/core_backup.dump ./backups/
```

## Configuration Files

All configuration is externalized:

- `.env` - Environment variables and secrets
- `config/paths_config.yaml` - Data directory paths
- `config/aggregation_config.yaml` - Aggregation rules, PII filters
- `config/neo4j_config.yaml` - Neo4j connection settings
- `config/graph_config.yaml` - Graph schema, bank_id

## Security Notes

1. **Never commit .env** - Contains secrets
2. **Change default passwords** - Neo4j, Wazuh
3. **Use HTTPS** - For Wazuh API connections
4. **Restrict Neo4j ports** - Only expose to trusted networks
5. **Regular updates** - Keep Docker images updated

## Support

For issues or questions:
1. Check logs in `/shared_data/logs/`
2. Review error codes in logs
3. Consult troubleshooting section above
4. Contact DataLegos team

## License

Copyright © 2026 DataLegos Team
