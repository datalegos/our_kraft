# Deployment Scripts Visual Guide

Quick visual reference for all deployment scripts.

## 📊 Script Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT SCRIPTS                        │
└─────────────────────────────────────────────────────────────┘

                    FIRST TIME SETUP
                          │
                          ▼
                  ┌───────────────┐
                  │ start_all.sh  │
                  │               │
                  │ • Check deps  │
                  │ • Create .env │
                  │ • Build images│
                  │ • Start all   │
                  └───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Services     │
                  │   Running     │
                  └───────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ logs.sh  │     │status.sh │     │run_pipe  │
  │          │     │          │     │line.sh   │
  │View logs │     │Check     │     │          │
  │          │     │status    │     │Execute   │
  └──────────┘     └──────────┘     └──────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │   stop.sh     │
                  │               │
                  │ Stop services │
                  └───────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ start.sh │     │cleanup.sh│     │ clear.sh │
  │          │     │          │     │          │
  │Restart   │     │Remove    │     │Remove    │
  │services  │     │containers│     │everything│
  │          │     │& images  │     │+ volumes │
  └──────────┘     └──────────┘     └──────────┘
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ start_all.sh  │
                  │               │
                  │ Start fresh   │
                  └───────────────┘
```

## 🎯 Script Decision Tree

```
Need to deploy?
│
├─ First time? ──────────────────────────► start_all.sh
│
├─ Services stopped? ─────────────────────► start.sh
│
├─ Want to view logs? ────────────────────► logs.sh
│
├─ Check if running? ─────────────────────► status.sh
│
├─ Run pipeline? ─────────────────────────► run_pipeline.sh
│
├─ Need to stop? ─────────────────────────► stop.sh
│
├─ Need to backup? ───────────────────────► backup.sh
│
├─ Rebuild images? ───────────────────────► cleanup.sh → start_all.sh
│
└─ Complete reset? ───────────────────────► clear.sh → start_all.sh
```

## 📋 Script Comparison Matrix

```
┌──────────────┬─────────┬──────────┬─────────┬─────────┬─────────┐
│   Script     │ Stops   │ Removes  │ Removes │ Removes │  Time   │
│              │Services │Container │ Images  │ Volumes │         │
├──────────────┼─────────┼──────────┼─────────┼─────────┼─────────┤
│ start_all.sh │    -    │    -     │    -    │    -    │ 5-10min │
│ start.sh     │    -    │    -     │    -    │    -    │ 10 sec  │
│ stop.sh      │    ✓    │    ✗     │    ✗    │    ✗    │ 5 sec   │
│ cleanup.sh   │    ✓    │    ✓     │    ✓    │    ✗    │ 30 sec  │
│ clear.sh     │    ✓    │    ✓     │    ✓    │    ✓    │ 30 sec  │
│ logs.sh      │    -    │    -     │    -    │    -    │ instant │
│ status.sh    │    -    │    -     │    -    │    -    │ instant │
│ backup.sh    │    -    │    -     │    -    │    -    │ 1-2 min │
│ run_pipe.sh  │    -    │    -     │    -    │    -    │ varies  │
└──────────────┴─────────┴──────────┴─────────┴─────────┴─────────┘

Legend:
  ✓ = Yes
  ✗ = No
  - = Not applicable
```

## 🔄 Common Workflows

### Workflow 1: First Time Deployment
```
1. start_all.sh
   └─► Edit .env
       └─► Services start
           └─► logs.sh (monitor)
               └─► status.sh (verify)
                   └─► run_pipeline.sh
```

### Workflow 2: Daily Operations
```
Morning:
  start.sh → status.sh → logs.sh

During Day:
  run_pipeline.sh → logs.sh → status.sh

Evening:
  stop.sh
```

### Workflow 3: Rebuild Images
```
cleanup.sh → start_all.sh → status.sh → logs.sh
```

### Workflow 4: Complete Reset
```
backup.sh → clear.sh → start_all.sh → status.sh
```

### Workflow 5: Troubleshooting
```
logs.sh → status.sh → cleanup.sh → start_all.sh
```

## 📊 Data Preservation Matrix

```
┌──────────────┬─────────────┬─────────────┬─────────────┐
│   Script     │   Neo4j     │   Shared    │   Docker    │
│              │   Volumes   │    Data     │   Images    │
├──────────────┼─────────────┼─────────────┼─────────────┤
│ stop.sh      │  Preserved  │  Preserved  │  Preserved  │
│ cleanup.sh   │  Preserved  │  Preserved  │  Removed    │
│ clear.sh     │  Removed    │  Preserved  │  Removed    │
└──────────────┴─────────────┴─────────────┴─────────────┘

Neo4j Volumes = Database data (node_kg, core)
Shared Data = ../orbit_node_shared_data/
Docker Images = Built container images
```

## 🎨 Color-Coded Usage Guide

```
🟢 SAFE - No data loss
   • start_all.sh
   • start.sh
   • stop.sh
   • logs.sh
   • status.sh
   • backup.sh
   • run_pipeline.sh

🟡 CAUTION - Removes containers/images (data preserved)
   • cleanup.sh

🔴 DANGER - Deletes Neo4j data
   • clear.sh
```

## 📖 Quick Command Reference

```bash
# ============================================================================
# SETUP & START
# ============================================================================
./start_all.sh              # Complete setup (first time)
./start.sh                  # Start existing services

# ============================================================================
# MONITORING
# ============================================================================
./logs.sh                   # Pipeline logs
./logs.sh neo4j             # Neo4j logs
./logs.sh all               # All logs
./status.sh                 # Check status

# ============================================================================
# OPERATIONS
# ============================================================================
./run_pipeline.sh           # Run complete pipeline
./run_pipeline.sh <step>    # Run specific step
./backup.sh                 # Backup databases

# ============================================================================
# STOP & CLEAN
# ============================================================================
./stop.sh                   # Stop services
./cleanup.sh                # Remove containers/images (🟡 CAUTION)
./clear.sh                  # Remove everything (🔴 DANGER)
```

## 🔍 Script Details

### start_all.sh
```
Purpose: Complete setup and start
When: First time, after cleanup/clear
Time: 5-10 minutes
Steps:
  1. Check prerequisites
  2. Create .env (if needed)
  3. Create directories
  4. Build Docker images
  5. Start services
  6. Wait for Neo4j
  7. Show status
```

### start.sh
```
Purpose: Start existing services
When: Daily operations, after stop
Time: 10 seconds
Steps:
  1. Start Docker services
  2. Show status
```

### stop.sh
```
Purpose: Stop running services
When: End of day, maintenance
Time: 5 seconds
Steps:
  1. Stop Docker services
  2. Show message
```

### cleanup.sh
```
Purpose: Remove containers and images
When: Before rebuild, free space
Time: 30 seconds
Preserves: Volumes, shared data
Steps:
  1. Confirm action
  2. Stop services
  3. Remove containers
  4. Remove images
```

### clear.sh
```
Purpose: Remove everything
When: Complete reset needed
Time: 30 seconds
⚠️ Deletes: Neo4j volumes
Preserves: Shared data
Steps:
  1. Confirm with 'DELETE EVERYTHING'
  2. Stop services
  3. Remove containers
  4. Remove images
  5. Remove volumes
```

### logs.sh
```
Purpose: View service logs
When: Monitoring, debugging
Usage:
  ./logs.sh           # Pipeline logs
  ./logs.sh neo4j     # Neo4j logs
  ./logs.sh all       # All logs
```

### status.sh
```
Purpose: Check system status
When: Verification, monitoring
Shows:
  • Docker service status
  • Pipeline completion
  • Step completion
  • Disk usage
```

### backup.sh
```
Purpose: Backup Neo4j databases
When: Before updates, regularly
Time: 1-2 minutes
Creates:
  • node_kg backup
  • core backup
Location: ../backups/
```

### run_pipeline.sh
```
Purpose: Execute pipeline
When: Data processing needed
Usage:
  ./run_pipeline.sh              # Complete
  ./run_pipeline.sh collect_data # Single step
Steps:
  • collect_data
  • extract_data
  • build_node_graph
  • aggregate_data
  • detect_pii
  • build_core_graph
```

## 🎯 Best Practices

### Daily Operations
```bash
# Morning
./start.sh && ./status.sh

# Work
./run_pipeline.sh
./logs.sh

# Evening
./stop.sh
```

### Weekly Maintenance
```bash
# Backup
./backup.sh

# Check status
./status.sh
```

### Before Updates
```bash
# Backup first
./backup.sh

# Clean up
./cleanup.sh

# Rebuild
./start_all.sh
```

### Troubleshooting
```bash
# Check logs
./logs.sh

# Check status
./status.sh

# If needed, clean restart
./cleanup.sh
./start_all.sh
```

## 📞 Quick Help

```bash
# View script documentation
cat deployment_scripts/README.md

# View main documentation
cat documentation/README.md

# Quick start guide
cat documentation/QUICK_START_SERVER.md

# Complete deployment guide
cat documentation/SERVER_DEPLOYMENT_GUIDE.md
```

---

**Print this page for quick reference!**
