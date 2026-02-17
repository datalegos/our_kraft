# Organized Folder Structure

## Current Issues
- Data directories mixed with code (aggregated_data, collected_data, etc.)
- Documentation files scattered in root
- Multiple README files without clear hierarchy
- __pycache__ directories not in .gitignore

## Proposed Organized Structure

```
orbit_node/                          # Project root
│
├── .github/                         # GitHub workflows (future)
│   └── workflows/
│
├── config/                          # Configuration files
│   ├── aggregation_config.yaml
│   ├── graph_config.yaml
│   ├── neo4j_config.yaml
│   └── paths_config.yaml
│
├── docker/                          # Docker deployment
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml      # Production override
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── healthcheck.sh
│
├── docs/                            # Documentation
│   ├── architecture/
│   │   ├── ARCHITECTURE.md
│   │   ├── ARCHITECTURE_DIAGRAM.md
│   │   └── FOLDER_STRUCTURE.md
│   ├── deployment/
│   │   ├── SERVER_DEPLOYMENT.md     # NEW: Server deployment guide
│   │   ├── DOCKER_DEPLOYMENT.md     # Renamed from README_DOCKER.md
│   │   └── docker-quickstart.sh
│   ├── features/
│   │   ├── AGGREGATION.md
│   │   ├── DAY0_NODE_GRAPH.md
│   │   ├── PII_DETECTION.md
│   │   └── PRIVACY_AGGREGATION.md
│   ├── implementation/
│   │   ├── AGGREGATION_COMPARISON.md
│   │   ├── AGGREGATION_IMPLEMENTATION.md
│   │   ├── DOCKER_IMPLEMENTATION.md
│   │   ├── PRIVACY_IMPLEMENTATION.md
│   │   └── CHANGES_SUMMARY.md
│   ├── guides/
│   │   ├── PIPELINE_GUIDE.md
│   │   ├── INSTALL_PRESIDIO.md
│   │   └── PII_SCAN_ANALYSIS.md
│   ├── reference/
│   │   ├── ENDPOINTS_REFERENCE.md
│   │   └── wazuh_endpoints_structure.csv
│   └── README.md                    # Main documentation index
│
├── graph_builder/                   # Graph building modules
│   ├── __init__.py
│   ├── constraints_manager.py
│   ├── graph_builder.py
│   ├── neo4j_connection.py
│   ├── node_inserter.py
│   ├── relationship_manager.py
│   ├── scanevent_manager.py
│   └── README.md
│
├── scripts/                         # Pipeline scripts
│   ├── orchestrator.py              # Main orchestrator
│   ├── main.py                      # Data collection
│   ├── extract_nodes.py             # Data extraction
│   ├── build_graph.py               # Node graph builder
│   ├── aggregate_data_v2.py         # Aggregation
│   ├── detect_pii.py                # PII detection
│   └── debug/                       # Debug utilities
│
├── standards/                       # Engineering standards
│   ├── backend.md
│   ├── ci-cd.md
│   ├── maintainability.md
│   ├── python_packages.md
│   └── tdd.md
│
├── steering/                        # Kiro steering files
│   ├── code-standards.md
│   └── maintainability.md
│
├── tests/                           # Test suite (future)
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── csv_converter.py
│   ├── data_collector.py
│   ├── vulnerability_detector.py
│   ├── wazuh_indexer_client.py
│   └── wazuh_manager_client.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Makefile
├── README.md                        # Main project README
├── requirements.txt
└── requirements_presidio.txt

# Data directories (NOT in git, created at runtime)
../orbit_node_shared_data/           # Sibling directory
├── collected_data/
├── extracted_data/
├── aggregated_data/
├── aggregated_data_core/
├── pii_scan_results/
├── logs/
└── pipeline/
```

## Migration Steps

### 1. Move Documentation Files
```bash
# Create new doc structure
mkdir -p docs/{architecture,deployment,features,implementation,guides,reference}

# Move architecture docs
mv ARCHITECTURE.md docs/architecture/
mv ARCHITECTURE_DIAGRAM.md docs/architecture/
mv FOLDER_STRUCTURE.md docs/architecture/

# Move deployment docs
mv README_DOCKER.md docs/deployment/DOCKER_DEPLOYMENT.md
mv docker-quickstart.sh docs/deployment/
mv DOCKER_IMPLEMENTATION_SUMMARY.md docs/implementation/DOCKER_IMPLEMENTATION.md

# Move feature docs
mv docs/AGGREGATION.md docs/features/
mv DAY0_NODE_GRAPH_CREATION.md docs/features/DAY0_NODE_GRAPH.md
mv docs/PII_DETECTION.md docs/features/
mv docs/PRIVACY_PRESERVING_AGGREGATION.md docs/features/PRIVACY_AGGREGATION.md

# Move implementation docs
mv AGGREGATION_COMPARISON.md docs/implementation/
mv AGGREGATION_IMPLEMENTATION.md docs/implementation/
mv PRIVACY_IMPLEMENTATION_SUMMARY.md docs/implementation/PRIVACY_IMPLEMENTATION.md
mv CHANGES_SUMMARY.md docs/implementation/

# Move guides
mv PIPELINE_GUIDE.md docs/guides/
mv INSTALL_PRESIDIO.md docs/guides/
mv PII_SCAN_ANALYSIS.md docs/guides/

# Move reference docs
mv ENDPOINTS_REFERENCE.md docs/reference/
mv wazuh_endpoints_structure.csv docs/reference/

# Clean up old docs
rm -f IMPLEMENTATION_COMPLETE.md PII_DETECTION_COMPLETE.md
rm -f README_AGGREGATION.md  # Merge into main README
```

### 2. Remove Data Directories from Git
```bash
# These should be in ../orbit_node_shared_data
git rm -r --cached aggregated_data/
git rm -r --cached aggregated_data_core/
git rm -r --cached collected_data/
git rm -r --cached extracted_data/
git rm -r --cached logs/
git rm -r --cached pii_scan_results/

# Add to .gitignore
echo "" >> .gitignore
echo "# Data directories (stored in ../orbit_node_shared_data)" >> .gitignore
echo "aggregated_data/" >> .gitignore
echo "aggregated_data_core/" >> .gitignore
echo "collected_data/" >> .gitignore
echo "extracted_data/" >> .gitignore
echo "logs/" >> .gitignore
echo "pii_scan_results/" >> .gitignore
```

### 3. Update .gitignore
```bash
# Add Python cache
echo "" >> .gitignore
echo "# Python cache" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.pyo" >> .gitignore
echo "*.pyd" >> .gitignore
echo ".Python" >> .gitignore
```

### 4. Create Tests Directory
```bash
mkdir -p tests/{unit,integration,fixtures}
touch tests/__init__.py
```

## Benefits of New Structure

1. **Clear Separation**
   - Code vs. Data vs. Documentation
   - Development vs. Deployment vs. Standards

2. **Better Navigation**
   - Grouped documentation by purpose
   - Easy to find deployment guides
   - Clear reference materials

3. **Git-Friendly**
   - No data in repository
   - Clean commit history
   - Smaller repo size

4. **Professional**
   - Industry-standard structure
   - Easy onboarding for new developers
   - Clear project organization

5. **Scalable**
   - Room for tests
   - Room for CI/CD
   - Room for additional features
