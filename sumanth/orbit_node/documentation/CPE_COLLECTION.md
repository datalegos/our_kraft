# CPE Collection - Common Platform Enumeration

## Overview

CPE (Common Platform Enumeration) is a standardized method for describing and identifying software applications, operating systems, and hardware devices. The NJS Pipeline now collects CPE identifiers from Wazuh agents to provide standardized platform identification.

## What is CPE?

CPE is a structured naming scheme for information technology systems, software, and packages maintained by NIST (National Institute of Standards and Technology). CPE identifiers are used to:

- Uniquely identify software and hardware platforms
- Match vulnerabilities to affected systems (CVE to CPE mapping)
- Standardize asset inventory across different tools
- Enable automated vulnerability management

### CPE Format

CPE identifiers follow this format:
```
cpe:/a:vendor:product:version:update:edition:language
```

Example:
```
cpe:/a:microsoft:windows_10:1909::pro
cpe:/a:apache:http_server:2.4.41
cpe:/a:python:python:3.9.0
```

## Data Collection

### Source

CPE data is collected from the Wazuh Manager API using the syscollector packages endpoint:
- **Endpoint**: `GET /syscollector/{agent_id}/packages`
- **Source**: Wazuh Manager
- **Data**: CPE identifiers are included in package information

### Collection Process

1. **Fetch Packages**: Retrieve all packages for each agent
2. **Extract CPE**: Filter packages that have CPE identifiers
3. **Enrich Data**: Include package metadata with CPE
4. **Save**: Store per-agent CPE data

### Collected Fields

For each package with a CPE identifier, we collect:

```json
{
  "agent_id": "000",
  "cpe": "cpe:/a:vendor:product:version",
  "package_name": "package-name",
  "package_version": "1.2.3",
  "package_architecture": "x86_64",
  "package_vendor": "Vendor Name",
  "package_description": "Package description",
  "scan_time": "2026-02-17T10:30:00Z"
}
```

## Configuration

### Enable CPE Collection

In `config/config.yaml`:

```yaml
collection:
  enabled_collectors:
    - agents
    - host
    - packages
    - hardware
    - cpe          # ← Add this
    - fim
    - vulnerabilities

collectors:
  cpe:
    source: "manager"
    limit: 1000
    max_retries: 3
    retry_delay: 5
```

### Path Configuration

In `config/paths_config.yaml`:

```yaml
paths:
  data_sources:
    cpe:
      directory: "cpe"
      file_pattern: "Syscollector_CPE_{agent_id}.json"
      description: "CPE (Common Platform Enumeration) identifiers"
      required: false
```

## Output Structure

### Directory Layout

```
njs_shared_data/data/collected/YYYYMMDD_HHMMSS/
├── cpe/
│   ├── agent_000/
│   │   └── Syscollector_CPE_000.json
│   ├── agent_001/
│   │   └── Syscollector_CPE_001.json
│   ├── agent_002/
│   │   └── Syscollector_CPE_002.json
│   └── CPE_Summary.json
```

### Per-Agent File Format

`Syscollector_CPE_000.json`:
```json
{
  "data": {
    "affected_items": [
      {
        "agent_id": "000",
        "cpe": "cpe:/a:microsoft:windows_10:1909::pro",
        "package_name": "Windows 10 Pro",
        "package_version": "1909",
        "package_architecture": "x86_64",
        "package_vendor": "Microsoft",
        "package_description": "Windows 10 Professional Edition",
        "scan_time": "2026-02-17T10:30:00Z"
      },
      {
        "agent_id": "000",
        "cpe": "cpe:/a:python:python:3.9.0",
        "package_name": "python3",
        "package_version": "3.9.0",
        "package_architecture": "x86_64",
        "package_vendor": "Python Software Foundation",
        "package_description": "Python programming language",
        "scan_time": "2026-02-17T10:30:00Z"
      }
    ],
    "total_affected_items": 2,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "CPE information extracted from packages for agent 000",
  "error": 0,
  "source": "manager"
}
```

### Summary File Format

`CPE_Summary.json`:
```json
{
  "000": {
    "data": {
      "affected_items": [...],
      "total_affected_items": 45
    },
    "message": "CPE information extracted from packages for agent 000",
    "error": 0,
    "source": "manager"
  },
  "001": {
    "data": {
      "affected_items": [...],
      "total_affected_items": 38
    },
    "message": "CPE information extracted from packages for agent 001",
    "error": 0,
    "source": "manager"
  }
}
```

## Usage

### Run Collection

```bash
# Run complete pipeline (includes CPE)
make start

# Or run collection script directly
poetry run python scripts/main.py
```

### Collection Output

```
======================================================================
COLLECTING CPE (COMMON PLATFORM ENUMERATION)
======================================================================
Collecting CPE information for 5 agents...
  ✓ Agent 000: Found 45 packages with CPE identifiers
  ✓ Agent 001: Found 38 packages with CPE identifiers
  ✓ Agent 002: Found 52 packages with CPE identifiers
  ✓ Agent 003: Found 41 packages with CPE identifiers
  ✓ Agent 004: Found 47 packages with CPE identifiers
  ✓ Successfully collected CPE for all 5 agents
Saved CPE data to njs_shared_data/data/collected/20260217_103000/cpe/CPE_Summary.json
```

### View Collected Data

```bash
# View CPE data for specific agent
cat ../njs_shared_data/data/collected/*/cpe/agent_000/Syscollector_CPE_000.json

# View summary
cat ../njs_shared_data/data/collected/*/cpe/CPE_Summary.json

# Count CPE identifiers per agent
jq '.data.total_affected_items' ../njs_shared_data/data/collected/*/cpe/agent_*/Syscollector_CPE_*.json
```

## Use Cases

### 1. Vulnerability Mapping

CPE identifiers enable precise vulnerability matching:
- CVE databases reference CPE identifiers
- Match vulnerabilities to specific software versions
- Automated vulnerability assessment

### 2. Asset Inventory

Standardized platform identification:
- Consistent naming across different systems
- Integration with external asset management tools
- Compliance reporting

### 3. Security Analysis

Enhanced security posture analysis:
- Identify outdated software versions
- Track software lifecycle
- Risk assessment based on platform exposure

### 4. Compliance

Regulatory compliance requirements:
- Software inventory for audits
- License management
- Security baseline verification

## Integration with Pipeline

### Data Flow

```
1. Collect Packages (with CPE)
   ↓
2. Extract CPE Identifiers
   ↓
3. Save to data/collected/*/cpe/
   ↓
4. Extract Data (normalize CPE)
   ↓
5. Build Node Graph (CPE as property)
   ↓
6. Aggregate (CPE-based grouping)
   ↓
7. Vulnerability Matching (CVE-CPE mapping)
```

### Node Graph Integration

CPE data can be added to the Node Graph as:
- **Software Node Property**: Add CPE to Software nodes
- **Platform Node**: Create dedicated CPE/Platform nodes
- **Relationship**: Link Software to CPE identifiers

Example Cypher query:
```cypher
// Add CPE to Software nodes
MATCH (s:Software {agent_id: '000', name: 'python3'})
SET s.cpe = 'cpe:/a:python:python:3.9.0'
```

### Vulnerability Matching

Use CPE for precise vulnerability matching:
```cypher
// Match vulnerabilities by CPE
MATCH (v:Vulnerability)
WHERE v.cpe = 'cpe:/a:python:python:3.9.0'
RETURN v.cve, v.severity, v.description
```

## API Reference

### CPECollector Class

```python
from utils.data_collector import CPECollector

# Initialize
cpe_collector = CPECollector(manager_client, config)

# Collect CPE for agents
result = cpe_collector.collect(agent_ids=['000', '001', '002'])

# Result structure
{
  '000': {
    'data': {
      'affected_items': [...],
      'total_affected_items': 45
    },
    'message': 'CPE information extracted...',
    'error': 0,
    'source': 'manager'
  }
}
```

### WazuhManagerClient Method

```python
# Get CPE data (via packages endpoint)
cpe_data = manager_client.get_cpe(agent_id='000', limit=1000)
```

## Troubleshooting

### No CPE Data Collected

**Issue**: CPE collection returns 0 items

**Possible Causes**:
1. Packages don't have CPE identifiers
2. Wazuh version doesn't support CPE
3. Syscollector not configured properly

**Solution**:
```bash
# Check if packages have CPE field
curl -k -u wazuh-wui:password \
  "https://wazuh-server:55000/syscollector/000/packages?pretty=true" \
  | jq '.data.affected_items[] | select(.cpe != null)'
```

### Collection Fails

**Issue**: CPE collection fails with error

**Solution**:
- Check Wazuh Manager connection
- Verify agent IDs are valid
- Check logs: `../njs_shared_data/logs/collect_data.log`

### Incomplete Data

**Issue**: Some agents missing CPE data

**Solution**:
- Retry collection (automatic with retry logic)
- Check agent connectivity
- Verify syscollector is running on agents

## Best Practices

1. **Regular Collection**: Collect CPE data regularly to track software changes
2. **Validation**: Verify CPE format matches NIST standards
3. **Enrichment**: Use CPE for vulnerability matching and risk assessment
4. **Monitoring**: Track CPE changes over time for security analysis
5. **Integration**: Link CPE data with CVE databases for comprehensive vulnerability management

## References

- [NIST CPE Dictionary](https://nvd.nist.gov/products/cpe)
- [CPE Specification](https://cpe.mitre.org/specification/)
- [Wazuh Syscollector Documentation](https://documentation.wazuh.com/current/user-manual/capabilities/syscollector.html)
- [CVE to CPE Mapping](https://nvd.nist.gov/vuln/search)

---

**Feature Added:** February 17, 2026  
**Version:** 1.0.0  
**Status:** Active
