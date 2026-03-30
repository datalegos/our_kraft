# Complete Data Pipeline Guide

## Pipeline Stages

### Day 0: Data Collection
Collect raw data from Wazuh Manager and Indexer
```bash
python scripts/main.py
```
Output: `collected_data/{timestamp}/`

### Day 1: Data Staging & Extraction
Extract and normalize node data from raw collections
```bash
python scripts/extract_nodes.py
```
Output: `extracted_data/{timestamp}/nodes/`

### Day 2: Detailed Graph Building
Build detailed Neo4j graph with all nodes and relationships
```bash
python scripts/build_graph.py
```
Output: Neo4j database (default)

### Day 3: Data Aggregation

Two aggregation approaches available:

**V1: Internal Analytics** (detailed, for your organization)
```bash
python scripts/aggregate_data.py
```
Output: `aggregated_data/{timestamp}/`
- Detailed breakdowns
- Version tracking
- Vendor information
- Multi-version analysis

**V2: Core Graph** (privacy-preserving, for consortium)
```bash
python scripts/aggregate_data_v2.py
```
Output: `aggregated_data_core/{timestamp}/`
- NO PII (hostnames, agent IDs)
- NO per-host data
- Only aggregated counts
- Privacy-compliant

Example aggregations:
- OS: Linux-1, Windows-3, macOS-1
- Software: python3-libs-3, openssl-5, curl-4
- Vulnerabilities: Critical-2, High-27, Medium-16

### Day 4: PII/PCI Detection (Presidio)
Scan aggregated data for sensitive information
```bash
python scripts/detect_pii.py
```
Output: `pii_scan_results/{timestamp}/`
- Scans all JSON files for PII/PCI
- Uses Microsoft Presidio
- Detects: names, emails, credit cards, IPs, etc.
- Generates compliance report
- Exit code 0 = safe, 1 = PII detected

**Installation:**
```bash
pip install -r requirements_presidio.txt
python -m spacy download en_core_web_lg
```

### Day 5: Core Graph Building (Future)
Build high-level analytics graph
```bash
python scripts/build_core_graph.py
```
Output: Neo4j database (core)

## Quick Start

### Internal Analytics Pipeline
```bash
python scripts/main.py
python scripts/extract_nodes.py
python scripts/build_graph.py
python scripts/aggregate_data.py        # V1: Detailed
```

### Core Graph Pipeline (Privacy-Preserving)
```bash
python scripts/main.py
python scripts/extract_nodes.py
python scripts/build_graph.py
python scripts/aggregate_data_v2.py     # V2: Privacy-compliant
python scripts/detect_pii.py            # Verify no PII
# python scripts/build_core_graph.py    # Future: Build Core Graph
```

## Configuration Files

- `config/config.yaml` - Wazuh connection settings
- `config/neo4j_config.yaml` - Neo4j connection settings
- `config/graph_config.yaml` - Graph schema and relationships
- `config/paths_config.yaml` - Data directory paths
- `config/aggregation_config.yaml` - Aggregation settings

## Output Directories

- `collected_data/` - Raw Wazuh data
- `extracted_data/` - Normalized node data
- `aggregated_data/` - V1: Detailed aggregations (internal)
- `aggregated_data_core/` - V2: Privacy-preserving (Core Graph)
- `logs/` - Application logs

## Session Tracking

Each stage creates a `.current_session` file pointing to the latest run:
- `extracted_data/.current_session`
- `aggregated_data/.current_session`

This allows scripts to automatically use the latest data.
