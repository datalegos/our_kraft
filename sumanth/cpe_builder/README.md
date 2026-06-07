# CPE Enricher

## Overview
Standalone CPE enrichment tool for the ORBIT Node Pipeline. Enriches software packages with CPE (Common Platform Enumeration) strings for accurate vulnerability linking.

## Purpose
- Wazuh doesn't provide CPEs for software packages
- CPEs are needed to link software to CVEs in the Core graph
- This tool bridges the gap by constructing and validating CPEs

## How It Works

### Two-Layer Matching Strategy

**Layer 1: Core Graph Match (Primary)**
- Queries existing CVEs in Core graph
- Extracts CPEs from CVE nodes
- Matches against software name/version
- Status: `MATCHED_CORE`

**Layer 2: NVD API Match (Fallback)**
- Queries official NVD CPE dictionary
- Uses nvdlib Python library
- Rate limited: 5 requests per 30 seconds
- Status: `MATCHED_NVD`

**Unverified (Last Resort)**
- Uses constructed CPE if no match found
- Still useful for future matching
- Status: `UNVERIFIED`

## Installation

```bash
# Install dependencies
pip install nvdlib neo4j python-dotenv pyyaml

# Or add to pyproject.toml
poetry add nvdlib neo4j python-dotenv pyyaml
```

## Configuration

### Environment Variables (.env)
```bash
# Neo4j Core Graph Connection
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# NVD API Key (optional - for higher rate limits)
NVD_API_KEY=
```

### Vendor Normalization Map (vendor_map.yaml)
Add new vendors as banks onboard with different software:

```yaml
vendor_mappings:
  "Microsoft Corporation": "microsoft"
  "Google LLC": "google"
  # Add more as needed
```

## Usage

### Standalone Mode

```bash
# Run with sample data
python cpe_builder/cpe_enricher.py
```

### As Module (Future Pipeline Integration)

```python
from cpe_builder.cpe_enricher import CPEEnricher

# Initialize
enricher = CPEEnricher()

# Load software from extractor output
with open('2_extractors/output/software_nodes.json') as f:
    software_list = json.load(f)

# Enrich
enriched = enricher.enrich_batch(software_list)

# Save results
enricher.save_results(enriched)
enricher.close()
```

## Input Format

```json
[
  {
    "name": "Google Chrome",
    "version": "145.0.7632.116",
    "vendor": "Google LLC",
    "agent_id": "001"
  }
]
```

## Output Format

```json
[
  {
    "name": "Google Chrome",
    "version": "145.0.7632.116",
    "vendor": "Google LLC",
    "agent_id": "001",
    "normalized_vendor": "google",
    "normalized_product": "chrome",
    "constructed_cpe": "cpe:2.3:a:google:chrome:145.0.7632.116:*:*:*:*:*:*:*",
    "matched_cpe": "cpe:2.3:a:google:chrome:145.0.7632.116:*:*:*:*:*:*:*",
    "match_status": "MATCHED_CORE",
    "matched_cve_ids": ["CVE-2024-XXXX"],
    "enriched_at": "2026-02-23T17:44:52Z"
  }
]
```

## Output Files

Results are saved to:
```
cpe_builder/output/cpe_enrichment_results_{timestamp}.json
```

## Match Status Values

- `MATCHED_CORE` - Matched against Core graph CVEs (highest confidence)
- `MATCHED_NVD` - Matched against NVD API (high confidence)
- `UNVERIFIED` - No match found, using constructed CPE (medium confidence)
- `ERROR` - Enrichment failed
- `CORE_UNAVAILABLE` - Neo4j connection failed, NVD-only mode
- `CORE_ERROR` - Core graph query failed
- `NVD_ERROR` - NVD API query failed

## Normalization Rules

### Vendor Normalization
- Lookup in vendor_map.yaml first
- Fallback: lowercase, remove special chars, replace spaces with underscores

### Product Normalization
- Lowercase everything
- Remove architecture: `(x64)`, `(x86)`, `(64-bit)`
- Remove `(User)` suffix
- Strip vendor name if present
- Remove version numbers embedded in name
- Replace spaces with underscores
- Remove special characters

### Examples

| Input | Normalized Product |
|-------|-------------------|
| `Google Chrome` | `chrome` |
| `Microsoft Edge` | `edge` |
| `Microsoft Visual C++ 2022 X64 Minimum Runtime` | `visual_c++_2022` |
| `Notepad++ (64-bit x64)` | `notepad++` |
| `Neo4j Desktop 2 2.1.1` | `neo4j_desktop` |

## Rate Limiting

**NVD API (without key):**
- 5 requests per 30 seconds
- Script adds 6-second delay between requests
- For higher limits, add NVD_API_KEY to .env

## Logging

Structured JSON logging to stdout:
```json
{
  "timestamp": "2026-02-23T17:44:52Z",
  "level": "INFO",
  "module": "cpe_enricher",
  "message": "Enriched: Google Chrome → MATCHED_CORE → cpe:2.3:a:google:chrome:145.0.7632.116:*:*:*:*:*:*:*"
}
```

## Pipeline Integration (TODO)

This tool will be integrated as **Step 2.5** in the pipeline:

```
1_collectors → 2_extractors → [2.5_cpe_enricher] → 3_graph → 4_aggregation → 5_privacy
```

**Integration points:**
- Input: `2_extractors/output/software_nodes.json`
- Output: `cpe_builder/output/enriched_software.json`
- Graph builder will read enriched data and create CPE nodes

## Troubleshooting

### Neo4j Connection Failed
- Check NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env
- Script falls back to NVD-only mode automatically

### NVD Rate Limit Exceeded
- Add NVD_API_KEY to .env for higher limits
- Or increase sleep time between requests

### No Matches Found
- Check vendor_map.yaml for vendor normalization
- Review product normalization rules
- Add custom mappings as needed

## Future Enhancements

1. **Fuzzy matching threshold configuration**
2. **Custom CPE construction rules per vendor**
3. **Batch processing with progress bar**
4. **Cache NVD results to reduce API calls**
5. **Manual review queue for unmatched software**
6. **Integration with pipeline orchestrator**

## License

Proprietary - ORBIT Node Pipeline
