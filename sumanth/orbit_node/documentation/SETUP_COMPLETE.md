# Setup Complete - DataLegos Deployment Scripts

## ✅ What Was Created

### 🚀 Deployment Scripts (5 Core + 4 Utility)

**Core Scripts:**
1. **start_all.sh** - Complete setup and start (first-time use)
2. **start.sh** - Start existing services (daily use)
3. **stop.sh** - Stop services
4. **cleanup.sh** - Remove containers and images (preserves data)
5. **clear.sh** - Remove everything including volumes (⚠️ deletes Neo4j data)

**Utility Scripts:**
6. **logs.sh** - View logs from services
7. **status.sh** - Check system status
8. **backup.sh** - Backup Neo4j databases
9. **run_pipeline.sh** - Run pipeline (complete or single step)

### 📚 Documentation

**Essential docs kept:**
- README.md (main)
- QUICK_START_SERVER.md
- SERVER_DEPLOYMENT_GUIDE.md
- DEPLOYMENT_FLOW.md
- ARCHITECTURE.md
- DAY0_NODE_GRAPH_CREATION.md
- PIPELINE_GUIDE.md
- INSTALL_PRESIDIO.md

**Unnecessary docs removed:**
- All redundant implementation summaries
- All comparison documents
- All organization guides
- All temporary documentation

### 🧹 Cleanup Script

**cleanup_project.sh** - Automated cleanup script that:
- Moves essential docs to `documentation/` folder
- Removes unnecessary documentation files
- Removes unnecessary scripts
- Creates clean README files
- Updates .gitignore

## 🎯 How to Use

### Option 1: Use Immediately (Recommended)

```bash
# Go to deployment scripts
cd deployment_scripts

# Complete setup and start
./start_all.sh

# Edit .env when prompted
# Services will start automatically
```

### Option 2: Clean Up First, Then Use

```bash
# Run cleanup script
chmod +x cleanup_project.sh
./cleanup_project.sh

# Then use deployment scripts
cd deployment_scripts
./start_all.sh
```

## 📋 Deployment Scripts Usage

### First Time Setup
```bash
cd deployment_scripts
./start_all.sh
# Edit .env when prompted
# Wait for services to start
```

### Daily Operations
```bash
# Start services
./start.sh

# View logs
./logs.sh

# Check status
./status.sh

# Run pipeline
./run_pipeline.sh

# Stop services
./stop.sh
```

### Maintenance
```bash
# Backup databases
./backup.sh

# Clean up old containers/images
./cleanup.sh

# Complete reset (⚠️ deletes Neo4j data)
./clear.sh
```

## 📁 Final Project Structure

```
orbit_node/
├── README.md                    # Main README
├── cleanup_project.sh           # Cleanup script (run once)
│
├── deployment_scripts/          # All deployment scripts
│   ├── README.md               # Scripts documentation
│   ├── start_all.sh            # Complete setup
│   ├── start.sh                # Start services
│   ├── stop.sh                 # Stop services
│   ├── cleanup.sh              # Remove containers/images
│   ├── clear.sh                # Remove everything
│   ├── logs.sh                 # View logs
│   ├── status.sh               # Check status
│   ├── backup.sh               # Backup databases
│   └── run_pipeline.sh         # Run pipeline
│
├── documentation/               # Essential documentation
│   ├── README.md
│   ├── QUICK_START_SERVER.md
│   ├── SERVER_DEPLOYMENT_GUIDE.md
│   ├── DEPLOYMENT_FLOW.md
│   ├── ARCHITECTURE.md
│   ├── DAY0_NODE_GRAPH_CREATION.md
│   ├── PIPELINE_GUIDE.md
│   └── INSTALL_PRESIDIO.md
│
├── config/                      # Configuration files
├── docker/                      # Docker setup
├── scripts/                     # Pipeline scripts
├── graph_builder/               # Graph modules
├── utils/                       # Utilities
└── standards/                   # Standards
```

## 🔄 Script Comparison

| Script | Purpose | Stops | Removes Containers | Removes Images | Removes Volumes |
|--------|---------|-------|-------------------|----------------|-----------------|
| start_all.sh | Complete setup | - | - | - | - |
| start.sh | Start services | - | - | - | - |
| stop.sh | Stop services | ✓ | ✗ | ✗ | ✗ |
| cleanup.sh | Clean up | ✓ | ✓ | ✓ | ✗ |
| clear.sh | Remove all | ✓ | ✓ | ✓ | ✓ |

**Note:** Shared data (../orbit_node_shared_data) is always preserved

## 📖 Quick Reference

### Setup Commands
```bash
# First time
cd deployment_scripts && ./start_all.sh

# After cleanup/clear
./start_all.sh
```

### Daily Commands
```bash
./start.sh              # Start
./logs.sh               # Logs
./status.sh             # Status
./run_pipeline.sh       # Run pipeline
./stop.sh               # Stop
```

### Maintenance Commands
```bash
./backup.sh             # Backup
./cleanup.sh            # Clean up
./clear.sh              # Reset (⚠️ deletes data)
```

### Log Commands
```bash
./logs.sh               # Pipeline logs
./logs.sh neo4j         # Neo4j logs
./logs.sh all           # All logs
```

### Pipeline Commands
```bash
./run_pipeline.sh                    # Complete pipeline
./run_pipeline.sh collect_data       # Specific step
./run_pipeline.sh extract_data
./run_pipeline.sh build_node_graph
./run_pipeline.sh aggregate_data
./run_pipeline.sh detect_pii
./run_pipeline.sh build_core_graph
```

## 🎨 Benefits

### Simple & Focused
- ✅ 5 core scripts for all operations
- ✅ 4 utility scripts for monitoring
- ✅ Clear purpose for each script
- ✅ No confusion about what to use

### Clean Structure
- ✅ All scripts in one folder
- ✅ All docs in one folder
- ✅ Clean project root
- ✅ Professional appearance

### Easy to Use
- ✅ Simple script names
- ✅ Clear documentation
- ✅ Consistent interface
- ✅ No complex commands

### Production Ready
- ✅ Complete setup automation
- ✅ Proper error handling
- ✅ Confirmation prompts
- ✅ Status checking

## 🚨 Important Notes

### About start_all.sh
- Use for first-time setup
- Use after cleanup.sh or clear.sh
- Builds images from scratch
- Takes several minutes

### About start.sh
- Use for daily operations
- Quick start (no building)
- Assumes images exist
- Use after stop.sh

### About cleanup.sh
- Removes containers and images
- **Preserves Neo4j data** (volumes)
- **Preserves shared data**
- Safe to use before rebuilding

### About clear.sh
- **Deletes Neo4j data** (volumes)
- Preserves shared data
- ⚠️ Use with caution!
- Requires confirmation

## 📞 Getting Help

### Check Documentation
```bash
# Deployment scripts
cat deployment_scripts/README.md

# Main documentation
cat documentation/README.md

# Specific guides
cat documentation/QUICK_START_SERVER.md
cat documentation/SERVER_DEPLOYMENT_GUIDE.md
```

### Check Status
```bash
cd deployment_scripts
./status.sh
./logs.sh
```

### Troubleshooting
```bash
# Check logs
./logs.sh

# Check status
./status.sh

# Clean and restart
./cleanup.sh
./start_all.sh
```

## ✨ Next Steps

### 1. Choose Your Path

**Path A: Use Immediately**
```bash
cd deployment_scripts
./start_all.sh
```

**Path B: Clean Up First**
```bash
./cleanup_project.sh
cd deployment_scripts
./start_all.sh
```

### 2. Configure
```bash
# Edit .env when prompted by start_all.sh
# Required settings:
#   - WAZUH_API_URL
#   - WAZUH_API_USERNAME
#   - WAZUH_API_PASSWORD
#   - NEO4J_PASSWORD
#   - BANK_ID
```

### 3. Verify
```bash
./status.sh
./logs.sh
```

### 4. Run Pipeline
```bash
./run_pipeline.sh
```

### 5. Monitor
```bash
./status.sh
./logs.sh
```

## 🎉 Summary

You now have:
- ✅ 9 focused deployment scripts
- ✅ Clean, organized documentation
- ✅ Simple, consistent interface
- ✅ Production-ready setup
- ✅ Easy maintenance

**Everything is ready to deploy!**

```bash
cd deployment_scripts
./start_all.sh
```

---

**Created:** 2026-02-17  
**Purpose:** Simplified deployment with focused scripts  
**Status:** Ready to use
