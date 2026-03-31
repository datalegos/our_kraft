# DataLegos Quick Reference Card

One-page reference for the new management script.

## 🚀 Quick Start

```bash
# First time setup
chmod +x datalegos.sh
./datalegos.sh install
nano .env                    # Edit configuration
./datalegos.sh start
./datalegos.sh logs
```

## 📋 Common Commands

### Daily Operations
```bash
./datalegos.sh start         # Start services
./datalegos.sh stop          # Stop services
./datalegos.sh logs          # View logs
./datalegos.sh status        # Check status
./datalegos.sh run           # Run pipeline
```

### Maintenance
```bash
./datalegos.sh backup        # Backup databases
./datalegos.sh health        # Check health
./datalegos.sh clean-data    # Clean old data
./datalegos.sh update        # Update application
```

### Troubleshooting
```bash
./datalegos.sh logs          # View logs
./datalegos.sh health        # Check health
./datalegos.sh status        # Check status
./datalegos.sh shell         # Access shell
```

## 🎯 All Commands

| Command | Description |
|---------|-------------|
| `install` | Complete installation (setup + build) |
| `setup` | Initial setup only |
| `build` | Build Docker images |
| `start` | Start all services |
| `stop` | Stop all services |
| `restart` | Restart all services |
| `status` | Show service status |
| `logs` | View pipeline logs |
| `logs-neo4j` | View Neo4j logs |
| `logs-all` | View all logs |
| `run` | Run complete pipeline |
| `run-step <name>` | Run specific step |
| `prod-start` | Start production mode |
| `prod-stop` | Stop production |
| `prod-restart` | Restart production |
| `prod-logs` | View production logs |
| `shell` | Pipeline container shell |
| `shell-neo4j` | Neo4j container shell |
| `neo4j` | Open Neo4j browser |
| `backup` | Backup Neo4j databases |
| `restore <date>` | Restore from backup |
| `clean-data` | Clean old data (30+ days) |
| `clean-docker` | Clean Docker resources |
| `health` | Check system health |
| `update` | Update application |
| `update-images` | Update Docker images |
| `check-pii` | View PII scan results |
| `check-pipeline` | Check pipeline status |
| `disk-usage` | Show disk usage |
| `reset` | Reset everything (WARNING) |
| `help` | Show all commands |

## 📁 Project Structure

```
orbit_node/
├── datalegos.sh              # Main management script
├── documentation/            # All documentation
├── deployment_scripts/       # All scripts
├── config/                   # Configuration
├── docker/                   # Docker setup
├── scripts/                  # Pipeline scripts
└── ...
```

## ⚙️ Configuration Files

```bash
.env                          # Environment variables (secrets)
config/aggregation_config.yaml    # Aggregation rules
config/paths_config.yaml          # Data paths
config/neo4j_config.yaml          # Neo4j settings
config/graph_config.yaml          # Graph schema
```

## 🔄 Pipeline Steps

```
1. collect_data      → Collect from Wazuh
2. extract_data      → Normalize data
3. build_node_graph  → Create Node KG
4. aggregate_data    → Create aggregations
5. detect_pii        → Scan for PII
6. build_core_graph  → Create Core Graph
```

## 🐳 Docker Services

```
neo4j                # Neo4j database (ports 7474, 7687)
pipeline             # Pipeline container
```

## 📊 Monitoring

```bash
# Service status
./datalegos.sh status

# View logs
./datalegos.sh logs

# Check health
./datalegos.sh health

# Pipeline status
./datalegos.sh check-pipeline

# PII scan results
./datalegos.sh check-pii

# Disk usage
./datalegos.sh disk-usage
```

## 🔧 Maintenance Schedule

**Daily:**
- Check status
- Review logs
- Monitor disk space

**Weekly:**
- Backup databases
- Clean old data
- Review PII scans

**Monthly:**
- Update application
- Update Docker images
- Review security

## 🚨 Emergency Commands

```bash
# Stop everything
./datalegos.sh stop

# Check what's wrong
./datalegos.sh health
./datalegos.sh logs

# Access shell for debugging
./datalegos.sh shell

# Restart services
./datalegos.sh restart

# Check disk space
./datalegos.sh disk-usage
```

## 📖 Documentation

```bash
# Main documentation
documentation/README.md

# Quick start
documentation/QUICK_START_SERVER.md

# Complete deployment guide
documentation/SERVER_DEPLOYMENT_GUIDE.md

# Deployment flow
documentation/DEPLOYMENT_FLOW.md

# Architecture
documentation/ARCHITECTURE.md
```

## 🔗 Access Points

```bash
# Neo4j Browser (Development)
http://localhost:7474

# Neo4j Browser (Production - SSH Tunnel)
ssh -L 7474:localhost:7474 user@server
http://localhost:7474

# Pipeline logs
./datalegos.sh logs

# Container shell
./datalegos.sh shell
```

## 💡 Tips

1. **Always check status first**: `./datalegos.sh status`
2. **View logs for errors**: `./datalegos.sh logs`
3. **Backup before updates**: `./datalegos.sh backup`
4. **Use health check**: `./datalegos.sh health`
5. **Clean old data regularly**: `./datalegos.sh clean-data`

## 🆘 Getting Help

```bash
# Show all commands
./datalegos.sh help

# Check documentation
cat documentation/README.md

# View specific guide
cat documentation/QUICK_START_SERVER.md
```

## 📞 Support Checklist

When asking for help, provide:
- [ ] Output of `./datalegos.sh status`
- [ ] Output of `./datalegos.sh health`
- [ ] Recent logs from `./datalegos.sh logs`
- [ ] Error messages
- [ ] What you were trying to do

---

**Print this page for quick reference!**

**For complete documentation, see:** [FINAL_ORGANIZATION_GUIDE.md](FINAL_ORGANIZATION_GUIDE.md)
