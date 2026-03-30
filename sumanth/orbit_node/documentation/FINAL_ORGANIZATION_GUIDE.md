# Final Organization Guide

Complete guide for the new organized project structure with single management script.

## 🎯 What Was Created

### 1. Single Management Script
**File:** `datalegos.sh`

One script to rule them all! Replaces the Makefile with a comprehensive bash script that handles:
- Setup & Build
- Service Management (start, stop, restart, status, logs)
- Pipeline Operations (run complete or single steps)
- Production Mode
- Access & Shell
- Maintenance (backup, restore, clean)
- Updates
- Utilities (health check, PII results, disk usage)

### 2. Organization Script
**File:** `organize_project.sh`

Automated script that organizes the project:
- Moves all markdown files to `documentation/` folder
- Moves all scripts to `deployment_scripts/` folder
- Creates README files for each folder
- Updates .gitignore
- Creates new main README

### 3. Organized Structure

```
orbit_node/
├── datalegos.sh                 # Main management script (NEW)
├── organize_project.sh          # Organization script (NEW)
├── README.md                    # Main README (will be updated)
│
├── documentation/               # All documentation (NEW)
│   ├── README.md               # Documentation index
│   ├── QUICK_START_SERVER.md
│   ├── SERVER_DEPLOYMENT_GUIDE.md
│   ├── DEPLOYMENT_FLOW.md
│   ├── ARCHITECTURE.md
│   ├── PIPELINE_GUIDE.md
│   └── ... (all other .md files)
│
├── deployment_scripts/          # All scripts (NEW)
│   ├── README.md               # Scripts index
│   ├── datalegos.sh            # Copy of main script
│   ├── migrate_structure.sh
│   └── docker-quickstart.sh
│
├── config/                      # Configuration files
├── docker/                      # Docker setup
├── scripts/                     # Pipeline scripts
├── graph_builder/               # Graph modules
├── utils/                       # Utilities
└── standards/                   # Standards
```

## 🚀 How to Use

### Option 1: Quick Start (No Organization)

Just use the new management script immediately:

```bash
# Make script executable
chmod +x datalegos.sh

# View all commands
./datalegos.sh help

# Setup and start
./datalegos.sh install
./datalegos.sh start
./datalegos.sh logs
```

### Option 2: Organize First, Then Use

Organize the project structure first:

```bash
# Make scripts executable
chmod +x organize_project.sh
chmod +x datalegos.sh

# Run organization
./organize_project.sh

# Review changes
ls -la documentation/
ls -la deployment_scripts/

# Replace README if satisfied
mv README.md README_OLD.md
mv README_NEW.md README.md

# Use management script
./datalegos.sh help
./datalegos.sh install
./datalegos.sh start
```

## 📋 Management Script Commands

### Setup & Build
```bash
./datalegos.sh setup              # Initial setup (create .env, directories)
./datalegos.sh build              # Build Docker images
./datalegos.sh install            # Complete installation (setup + build)
```

### Service Management
```bash
./datalegos.sh start              # Start all services
./datalegos.sh stop               # Stop all services
./datalegos.sh restart            # Restart all services
./datalegos.sh status             # Show service status
./datalegos.sh logs               # View pipeline logs (follow mode)
./datalegos.sh logs-neo4j         # View Neo4j logs
./datalegos.sh logs-all           # View all logs
```

### Pipeline Operations
```bash
./datalegos.sh run                # Run complete pipeline
./datalegos.sh run-step collect_data      # Run specific step
./datalegos.sh run-step extract_data
./datalegos.sh run-step build_node_graph
./datalegos.sh run-step aggregate_data
./datalegos.sh run-step detect_pii
./datalegos.sh run-step build_core_graph
```

### Production Mode
```bash
./datalegos.sh prod-start         # Start in production mode
./datalegos.sh prod-stop          # Stop production services
./datalegos.sh prod-restart       # Restart production services
./datalegos.sh prod-logs          # View production logs
```

### Access
```bash
./datalegos.sh shell              # Access pipeline container shell
./datalegos.sh shell-neo4j        # Access Neo4j container shell
./datalegos.sh neo4j              # Open Neo4j browser
```

### Maintenance
```bash
./datalegos.sh backup             # Backup Neo4j databases
./datalegos.sh restore <date>     # Restore from backup (YYYYMMDD_HHMMSS)
./datalegos.sh clean-data         # Clean old data (30+ days)
./datalegos.sh clean-docker       # Clean Docker resources
./datalegos.sh health             # Check system health
```

### Updates
```bash
./datalegos.sh update             # Update application (git pull + rebuild)
./datalegos.sh update-images      # Update Docker images only
```

### Utilities
```bash
./datalegos.sh check-pii          # View latest PII scan results
./datalegos.sh check-pipeline     # Check pipeline completion status
./datalegos.sh disk-usage         # Show disk usage
./datalegos.sh reset              # Reset everything (WARNING: deletes all data)
```

## 🎨 Benefits of New Structure

### 1. Single Script Management
- ✅ One script for all operations
- ✅ No need to remember multiple commands
- ✅ Consistent interface
- ✅ Easy to use and maintain

### 2. Organized Documentation
- ✅ All docs in one folder
- ✅ Easy to find information
- ✅ Clear documentation index
- ✅ Professional structure

### 3. Organized Scripts
- ✅ All scripts in one folder
- ✅ Clear purpose for each script
- ✅ Easy to manage and update
- ✅ Scripts README for reference

### 4. Clean Project Root
- ✅ No scattered markdown files
- ✅ No scattered scripts
- ✅ Clear project structure
- ✅ Professional appearance

## 📖 Quick Reference

### First Time Setup
```bash
# 1. Make scripts executable
chmod +x datalegos.sh organize_project.sh

# 2. Organize project (optional)
./organize_project.sh

# 3. Setup and install
./datalegos.sh install

# 4. Edit configuration
nano .env

# 5. Start services
./datalegos.sh start

# 6. View logs
./datalegos.sh logs
```

### Daily Operations
```bash
# Start services
./datalegos.sh start

# Run pipeline
./datalegos.sh run

# Check status
./datalegos.sh status

# View logs
./datalegos.sh logs

# Stop services
./datalegos.sh stop
```

### Maintenance
```bash
# Backup databases
./datalegos.sh backup

# Clean old data
./datalegos.sh clean-data

# Check health
./datalegos.sh health

# Update application
./datalegos.sh update
```

## 🔄 Migration Path

### Path 1: Use Immediately (Recommended)
```bash
chmod +x datalegos.sh
./datalegos.sh help
./datalegos.sh install
```

### Path 2: Organize Then Use
```bash
chmod +x organize_project.sh datalegos.sh
./organize_project.sh
mv README.md README_OLD.md
mv README_NEW.md README.md
./datalegos.sh install
```

## 📁 Before vs After

### Before
```
orbit_node/
├── README.md
├── ARCHITECTURE.md
├── DEPLOYMENT_FLOW.md
├── SERVER_DEPLOYMENT_GUIDE.md
├── QUICK_START_SERVER.md
├── ... (20+ markdown files scattered)
├── docker-quickstart.sh
├── migrate_structure.sh
├── Makefile
├── config/
├── docker/
├── scripts/
└── ...
```

### After
```
orbit_node/
├── datalegos.sh                 # Single management script
├── README.md                    # Clean main README
├── documentation/               # All docs organized
│   ├── README.md
│   └── ... (all .md files)
├── deployment_scripts/          # All scripts organized
│   ├── README.md
│   └── ... (all .sh files)
├── config/
├── docker/
├── scripts/
└── ...
```

## 🎯 Key Features

### Management Script Features
- ✅ Colored output for better readability
- ✅ Error handling and validation
- ✅ Confirmation prompts for destructive operations
- ✅ Comprehensive help system
- ✅ All operations in one place
- ✅ Easy to extend and customize

### Organization Features
- ✅ Automated organization script
- ✅ Safe to run (doesn't delete data)
- ✅ Creates README files for each folder
- ✅ Updates .gitignore
- ✅ Creates new main README
- ✅ Preserves all files

## 🚨 Important Notes

### About the Management Script
1. **Location**: Keep `datalegos.sh` in project root
2. **Permissions**: Make it executable (`chmod +x datalegos.sh`)
3. **Usage**: Run from project root directory
4. **Help**: Always available with `./datalegos.sh help`

### About Organization
1. **Optional**: You can use the management script without organizing
2. **Safe**: Organization script doesn't delete anything
3. **Reversible**: Keep old files until you're satisfied
4. **Git**: Review changes before committing

### About Makefile
1. **Replaced**: The new script replaces Makefile functionality
2. **Keep**: You can keep Makefile if you prefer
3. **Compatible**: Both can coexist
4. **Recommended**: Use the new script for consistency

## 📞 Support

### Getting Help
```bash
# Show all commands
./datalegos.sh help

# Check status
./datalegos.sh status

# Check health
./datalegos.sh health

# View logs
./datalegos.sh logs
```

### Documentation
- Main README: `README.md` (or `README_NEW.md` before organizing)
- All documentation: `documentation/` folder
- Scripts documentation: `deployment_scripts/` folder

### Troubleshooting
```bash
# Check logs
./datalegos.sh logs

# Check health
./datalegos.sh health

# Check status
./datalegos.sh status

# Access shell for debugging
./datalegos.sh shell
```

## ✅ Checklist

### Before Starting
- [ ] Review this guide
- [ ] Decide: organize now or later?
- [ ] Backup important data (if any)

### Setup
- [ ] Make scripts executable
- [ ] Run organization (optional)
- [ ] Review new structure
- [ ] Update README (optional)

### Installation
- [ ] Run `./datalegos.sh install`
- [ ] Edit `.env` file
- [ ] Start services
- [ ] Verify everything works

### Verification
- [ ] Check status: `./datalegos.sh status`
- [ ] View logs: `./datalegos.sh logs`
- [ ] Run pipeline: `./datalegos.sh run`
- [ ] Check health: `./datalegos.sh health`

## 🎉 Summary

You now have:
1. ✅ Single management script (`datalegos.sh`)
2. ✅ Organization script (`organize_project.sh`)
3. ✅ Clean folder structure plan
4. ✅ Comprehensive documentation
5. ✅ Easy-to-use interface

**Ready to deploy!**

```bash
# Quick start
chmod +x datalegos.sh
./datalegos.sh install
./datalegos.sh start
./datalegos.sh logs
```

---

**Created:** 2026-02-17  
**Purpose:** Single script management and organized structure  
**Status:** Ready to use
