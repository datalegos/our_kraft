# Quick Start - Server Deployment

Fast track guide to deploy DataLegos on a production server.

## Prerequisites Checklist

- [ ] Ubuntu 20.04+ / CentOS 8+ / RHEL 8+ server
- [ ] 8GB RAM minimum (16GB recommended)
- [ ] 50GB disk space minimum
- [ ] SSH access with sudo privileges
- [ ] Wazuh Manager API access

## 5-Minute Deployment

### 1. Install Docker (Ubuntu)

```bash
# Quick install script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl start docker
sudo systemctl enable docker

# Verify
docker --version
docker compose version
```

### 2. Clone & Setup

```bash
# Create directories
sudo mkdir -p /opt/datalegos
cd /opt/datalegos

# Clone repository
git clone https://github.com/your-org/orbit_node.git
cd orbit_node

# Create shared data directory
mkdir -p /opt/datalegos_shared_data
```

### 3. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Minimum required settings:**
```bash
WAZUH_API_URL=https://your-wazuh-manager:55000
WAZUH_API_USERNAME=wazuh-admin
WAZUH_API_PASSWORD=your_password
NEO4J_PASSWORD=your_secure_password
BANK_ID=bank_001
```

### 4. Deploy

```bash
# Build images
cd docker
docker compose build

# Start services
docker compose up -d

# Monitor
docker compose logs -f pipeline
```

### 5. Verify

```bash
# Check services
docker compose ps

# Check pipeline completion
docker compose exec pipeline cat /shared_data/pipeline/.done

# Access Neo4j (optional)
# Open browser: http://your-server-ip:7474
```

## Using Makefile (Recommended)

```bash
# Setup
make setup
# Edit .env with your settings

# Build and start
make build
make up

# Monitor
make logs

# Check status
make status
```

## Common Commands

```bash
# View logs
docker compose logs -f pipeline
docker compose logs -f neo4j

# Run specific step
docker compose exec pipeline python /app/scripts/orchestrator.py --step collect_data

# Access shell
docker compose exec pipeline /bin/bash

# Stop services
docker compose down

# Restart services
docker compose restart
```

## Troubleshooting

### Pipeline fails
```bash
# Check logs
docker compose logs pipeline | grep ERROR

# Check specific step
docker compose exec pipeline cat /shared_data/logs/collect_data.log
```

### Neo4j connection issues
```bash
# Check Neo4j
docker compose ps neo4j
docker compose logs neo4j

# Test connection
docker compose exec neo4j cypher-shell -u neo4j -p <password> "RETURN 1"
```

### Wazuh API issues
```bash
# Test from container
docker compose exec pipeline curl -k -u <user>:<pass> https://your-wazuh:55000/
```

## Production Deployment

For production with resource limits and security:

```bash
# Use production compose file
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or with Makefile
make prod-up
```

## Next Steps

1. ✅ Services running
2. ✅ Pipeline completed
3. ✅ Data in Neo4j

**Now:**
- Set up scheduled runs (edit docker-compose.yml)
- Configure backups (make backup)
- Set up monitoring
- Review security settings

## Full Documentation

- 📚 [Complete Server Deployment Guide](docs/deployment/SERVER_DEPLOYMENT_GUIDE.md)
- 🐳 [Docker Deployment Details](docs/deployment/DOCKER_DEPLOYMENT.md)
- 📐 [Architecture Overview](docs/architecture/ARCHITECTURE.md)

## Support

Issues? Check:
1. Logs: `/opt/datalegos_shared_data/logs/`
2. [Troubleshooting Guide](docs/deployment/SERVER_DEPLOYMENT_GUIDE.md#troubleshooting)
3. Contact DataLegos team
