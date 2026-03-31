# Server Deployment Guide - DataLegos Pipeline

Complete guide for deploying the DataLegos pipeline on a production server using Docker.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Server Setup](#server-setup)
3. [Installation Steps](#installation-steps)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Security Best Practices](#security-best-practices)

---

## Prerequisites

### Server Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **CPU**: 4 cores minimum (8 cores recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 50GB minimum (100GB+ recommended for data storage)
- **Network**: Access to Wazuh Manager API

### Software Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- SSH access with sudo privileges

### Network Requirements
- Outbound HTTPS (443) to Wazuh Manager
- Inbound access to Neo4j ports (7474, 7687) - optional, for remote access
- Docker Hub access (or private registry)

---

## Server Setup

### 1. Connect to Server

```bash
# SSH into your server
ssh user@your-server-ip

# Switch to root or use sudo for all commands
sudo su -
```

### 2. Update System

```bash
# Ubuntu/Debian
apt update && apt upgrade -y

# CentOS/RHEL
yum update -y
```

### 3. Install Docker

#### Ubuntu/Debian
```bash
# Remove old versions
apt remove docker docker-engine docker.io containerd runc

# Install dependencies
apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up stable repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

#### CentOS/RHEL
```bash
# Remove old versions
yum remove docker docker-client docker-client-latest docker-common \
    docker-latest docker-latest-logrotate docker-logrotate docker-engine

# Install dependencies
yum install -y yum-utils

# Add Docker repository
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker Engine
yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
systemctl start docker
systemctl enable docker

# Verify installation
docker --version
docker compose version
```

### 4. Configure Docker (Optional but Recommended)

```bash
# Create Docker daemon config
cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF

# Restart Docker
systemctl restart docker
```

### 5. Install Git

```bash
# Ubuntu/Debian
apt install -y git

# CentOS/RHEL
yum install -y git

# Verify
git --version
```

---

## Installation Steps

### 1. Create Application Directory

```bash
# Create directory for the application
mkdir -p /opt/datalegos
cd /opt/datalegos

# Create shared data directory (sibling to project)
mkdir -p /opt/datalegos_shared_data
```

### 2. Clone Repository

```bash
# Clone from Git repository
cd /opt/datalegos
git clone https://github.com/your-org/orbit_node.git

# Or if using SSH
git clone git@github.com:your-org/orbit_node.git

# Navigate to project
cd orbit_node

# Checkout specific branch/tag if needed
git checkout main  # or production, v1.0.0, etc.
```

### 3. Verify Directory Structure

```bash
# Check structure
ls -la

# Should see:
# - config/
# - docker/
# - scripts/
# - graph_builder/
# - utils/
# - .env.example
# - requirements.txt
# etc.

# Verify shared data directory exists
ls -la /opt/datalegos_shared_data
```

---

## Configuration

### 1. Create Environment File

```bash
# Copy example environment file
cd /opt/datalegos/orbit_node
cp .env.example .env

# Edit with your configuration
nano .env
# or
vi .env
```

### 2. Configure Environment Variables

Edit `.env` file with your settings:

```bash
# ============================================================================
# Wazuh Configuration
# ============================================================================
WAZUH_API_URL=https://your-wazuh-manager.example.com:55000
WAZUH_API_USERNAME=wazuh-admin
WAZUH_API_PASSWORD=your_secure_wazuh_password

# ============================================================================
# Neo4j Configuration
# ============================================================================
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_neo4j_password_min_8_chars
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687

# Neo4j Databases
NEO4J_NODE_DATABASE=node_kg
NEO4J_CORE_DATABASE=core

# ============================================================================
# Bank Configuration
# ============================================================================
# Unique identifier for this bank/organization
BANK_ID=bank_001

# ============================================================================
# Pipeline Configuration
# ============================================================================
# Mode: run-once or scheduled
PIPELINE_MODE=run-once

# Cron schedule (if scheduled mode)
# Format: minute hour day month weekday
# Example: 0 2 * * * (daily at 2 AM)
PIPELINE_SCHEDULE=0 2 * * *

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# ============================================================================
# Git Configuration (for Docker build)
# ============================================================================
# Leave empty to build from local files
GIT_REPO_URL=
GIT_BRANCH=main
```

### 3. Set Proper Permissions

```bash
# Secure .env file
chmod 600 .env

# Set ownership (if running as non-root user)
chown -R your-user:your-user /opt/datalegos/orbit_node
chown -R your-user:your-user /opt/datalegos_shared_data

# Or if running as root
chown -R root:root /opt/datalegos/orbit_node
chown -R root:root /opt/datalegos_shared_data
```

### 4. Configure Firewall (if needed)

```bash
# Ubuntu/Debian (UFW)
ufw allow 7474/tcp  # Neo4j HTTP (optional, for remote access)
ufw allow 7687/tcp  # Neo4j Bolt (optional, for remote access)

# CentOS/RHEL (firewalld)
firewall-cmd --permanent --add-port=7474/tcp
firewall-cmd --permanent --add-port=7687/tcp
firewall-cmd --reload

# Note: Only open these ports if you need remote access to Neo4j
# For production, use SSH tunneling instead
```

---

## Running the Pipeline

### 1. Build Docker Images

```bash
cd /opt/datalegos/orbit_node/docker

# Build images
docker compose build

# This will:
# - Pull Neo4j image
# - Build pipeline image with Python + Presidio
# - Install all dependencies
```

### 2. Start Services

```bash
# Start all services in detached mode
docker compose up -d

# Check status
docker compose ps

# Should see:
# - datalegos_neo4j (running)
# - datalegos_pipeline (running or exited if run-once)
```

### 3. View Logs

```bash
# View all logs
docker compose logs

# Follow logs in real-time
docker compose logs -f

# View specific service logs
docker compose logs neo4j
docker compose logs pipeline

# View last 100 lines
docker compose logs --tail=100 pipeline
```

### 4. Monitor Pipeline Execution

```bash
# Check pipeline logs
docker compose exec pipeline cat /shared_data/logs/pipeline.log

# Check specific step logs
docker compose exec pipeline cat /shared_data/logs/collect_data.log
docker compose exec pipeline cat /shared_data/logs/detect_pii.log

# Check completion status
docker compose exec pipeline cat /shared_data/pipeline/.done
```

### 5. Access Neo4j Browser

```bash
# Open browser and navigate to:
http://your-server-ip:7474

# Login with credentials from .env:
Username: neo4j
Password: <NEO4J_PASSWORD>

# Or use SSH tunnel for secure access:
ssh -L 7474:localhost:7474 -L 7687:localhost:7687 user@your-server-ip

# Then access locally:
http://localhost:7474
```

---

## Pipeline Modes

### Run-Once Mode (Default)

Pipeline runs once and exits.

```bash
# Already configured in .env
PIPELINE_MODE=run-once

# Start pipeline
docker compose up -d

# Pipeline will:
# 1. Collect data from Wazuh
# 2. Extract and normalize
# 3. Build Node graph
# 4. Create aggregations
# 5. Scan for PII
# 6. Build Core graph
# 7. Exit

# Check if completed
docker compose ps pipeline
# Status should be "Exited (0)" if successful
```

### Scheduled Mode (Future)

Pipeline runs on a schedule.

```bash
# Edit .env
PIPELINE_MODE=scheduled
PIPELINE_SCHEDULE=0 2 * * *  # Daily at 2 AM

# Restart services
docker compose restart pipeline

# Pipeline will run on schedule
# Check logs to see scheduled runs
docker compose logs -f pipeline
```

### Manual Single Step Execution

```bash
# Run specific step
docker compose exec pipeline python /app/scripts/orchestrator.py --step collect_data

# Available steps:
# - collect_data
# - extract_data
# - build_node_graph
# - aggregate_data
# - detect_pii
# - build_core_graph
```

---

## Monitoring & Maintenance

### 1. Check Service Health

```bash
# Check all services
docker compose ps

# Check Neo4j health
docker compose exec neo4j cypher-shell -u neo4j -p <password> "RETURN 1"

# Check disk usage
df -h /opt/datalegos_shared_data

# Check Docker disk usage
docker system df
```

### 2. View Data

```bash
# List collected data sessions
ls -la /opt/datalegos_shared_data/collected_data/

# List aggregated data
ls -la /opt/datalegos_shared_data/aggregated_data_core/

# List PII scan results
ls -la /opt/datalegos_shared_data/pii_scan_results/

# View latest PII scan summary
cat /opt/datalegos_shared_data/pii_scan_results/*/pii_scan_summary.txt
```

### 3. Backup Neo4j Data

```bash
# Stop services
docker compose stop

# Backup Neo4j data volume
docker run --rm \
  -v datalegos_neo4j_data:/data \
  -v /opt/backups:/backup \
  ubuntu tar czf /backup/neo4j_backup_$(date +%Y%m%d_%H%M%S).tar.gz /data

# Or use Neo4j dump
docker compose exec neo4j neo4j-admin database dump node_kg \
  --to=/var/lib/neo4j/import/node_kg_backup.dump

docker compose exec neo4j neo4j-admin database dump core \
  --to=/var/lib/neo4j/import/core_backup.dump

# Copy backups to host
docker cp datalegos_neo4j:/var/lib/neo4j/import/node_kg_backup.dump /opt/backups/
docker cp datalegos_neo4j:/var/lib/neo4j/import/core_backup.dump /opt/backups/

# Restart services
docker compose start
```

### 4. Clean Old Data

```bash
# Remove data older than 30 days
find /opt/datalegos_shared_data/collected_data/ -type d -mtime +30 -exec rm -rf {} +
find /opt/datalegos_shared_data/pii_scan_results/ -type d -mtime +30 -exec rm -rf {} +

# Clean Docker resources
docker system prune -a --volumes -f
```

### 5. Update Application

```bash
# Stop services
cd /opt/datalegos/orbit_node/docker
docker compose down

# Pull latest code
cd /opt/datalegos/orbit_node
git pull origin main

# Rebuild images
cd docker
docker compose build --no-cache

# Start services
docker compose up -d

# Verify
docker compose ps
docker compose logs -f pipeline
```

---

## Troubleshooting

### Pipeline Fails at Step

```bash
# Check which step failed
docker compose logs pipeline | grep "ERROR"

# Check step-specific log
docker compose exec pipeline cat /shared_data/logs/<step_name>.log

# Check orchestrator log
docker compose exec pipeline cat /shared_data/logs/pipeline.log

# Restart from failed step
docker compose exec pipeline python /app/scripts/orchestrator.py --step <step_name>
```

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker compose ps neo4j

# Check Neo4j logs
docker compose logs neo4j

# Test connection
docker compose exec neo4j cypher-shell -u neo4j -p <password> "RETURN 1"

# Restart Neo4j
docker compose restart neo4j

# Wait for health check
docker compose ps neo4j
```

### Wazuh API Connection Issues

```bash
# Test Wazuh API from container
docker compose exec pipeline curl -k -u <username>:<password> \
  https://your-wazuh-manager:55000/

# Check network connectivity
docker compose exec pipeline ping your-wazuh-manager

# Verify credentials in .env
cat .env | grep WAZUH
```

### Disk Space Issues

```bash
# Check disk usage
df -h

# Check Docker disk usage
docker system df

# Clean old data
find /opt/datalegos_shared_data -type d -mtime +30 -exec rm -rf {} +

# Clean Docker
docker system prune -a --volumes
```

### Permission Issues

```bash
# Fix shared data permissions
chown -R root:root /opt/datalegos_shared_data
chmod -R 755 /opt/datalegos_shared_data

# Or if running as specific user
chown -R your-user:your-user /opt/datalegos_shared_data
```

### Container Won't Start

```bash
# Check logs
docker compose logs pipeline

# Check environment variables
docker compose exec pipeline env | grep -E "WAZUH|NEO4J|BANK"

# Recreate containers
docker compose down
docker compose up -d

# Check .env file
cat .env
```

---

## Security Best Practices

### 1. Secure Credentials

```bash
# Use strong passwords (minimum 16 characters)
# - Neo4j password
# - Wazuh API password

# Secure .env file
chmod 600 .env
chown root:root .env

# Never commit .env to git
# Verify .gitignore includes .env
```

### 2. Network Security

```bash
# Use firewall to restrict access
# Only allow necessary ports

# Use SSH tunneling for Neo4j access instead of exposing ports
ssh -L 7474:localhost:7474 -L 7687:localhost:7687 user@server

# Use HTTPS for Wazuh API (already configured)
```

### 3. Regular Updates

```bash
# Update system packages
apt update && apt upgrade -y  # Ubuntu/Debian
yum update -y                  # CentOS/RHEL

# Update Docker images
docker compose pull
docker compose up -d

# Update application code
git pull origin main
docker compose build --no-cache
docker compose up -d
```

### 4. Monitoring

```bash
# Set up log rotation
cat > /etc/logrotate.d/datalegos <<EOF
/opt/datalegos_shared_data/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF

# Set up monitoring alerts (future)
# - Disk space alerts
# - Pipeline failure alerts
# - Neo4j health alerts
```

### 5. Backup Strategy

```bash
# Daily backups
# Create cron job for automated backups

# Edit crontab
crontab -e

# Add backup job (daily at 3 AM)
0 3 * * * /opt/datalegos/orbit_node/scripts/backup.sh

# Create backup script
cat > /opt/datalegos/orbit_node/scripts/backup.sh <<'EOF'
#!/bin/bash
BACKUP_DIR=/opt/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Backup Neo4j
docker compose -f /opt/datalegos/orbit_node/docker/docker-compose.yml \
  exec -T neo4j neo4j-admin database dump node_kg \
  --to=/var/lib/neo4j/import/node_kg_${DATE}.dump

docker cp datalegos_neo4j:/var/lib/neo4j/import/node_kg_${DATE}.dump \
  ${BACKUP_DIR}/

# Remove backups older than 7 days
find ${BACKUP_DIR} -name "*.dump" -mtime +7 -delete
EOF

chmod +x /opt/datalegos/orbit_node/scripts/backup.sh
```

---

## Quick Reference Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Restart services
docker compose restart

# View logs
docker compose logs -f

# Check status
docker compose ps

# Run pipeline manually
docker compose exec pipeline python /app/scripts/orchestrator.py

# Access Neo4j browser
http://your-server-ip:7474

# View pipeline logs
docker compose exec pipeline cat /shared_data/logs/pipeline.log

# Check PII scan results
docker compose exec pipeline cat /shared_data/pii_scan_results/*/pii_scan_summary.txt

# Backup Neo4j
docker compose exec neo4j neo4j-admin database dump node_kg --to=/var/lib/neo4j/import/backup.dump

# Update application
git pull && docker compose build --no-cache && docker compose up -d
```

---

## Support

For issues or questions:
1. Check logs in `/opt/datalegos_shared_data/logs/`
2. Review error codes in logs
3. Consult troubleshooting section above
4. Contact DataLegos team

---

## License

Copyright © 2026 DataLegos Team
