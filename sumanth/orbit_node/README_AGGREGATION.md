# Data Aggregation Module - Quick Start

## Overview

Two aggregation options for different use cases:

| Version | Purpose | Privacy | Output |
|---------|---------|---------|--------|
| **V1** | Internal analytics | ⚠️ May contain PII | `aggregated_data/` |
| **V2** | Core Graph (consortium) | ✅ Privacy-preserving | `aggregated_data_core/` |

## Quick Start

### For Internal Analytics
```bash
python scripts/aggregate_data.py
```

### For Core Graph (Recommended for Consortium)
```bash
python scripts/aggregate_data_v2.py
```

## What's the Difference?

### V1: Internal Analytics
```json
{
  "host": {
    "os_version_distribution": {
      "Windows-10.0.26200.7623": 2,
      "Linux-2023": 1
    }
  }
}
```
- ✅ Detailed versions
- ✅ Vendor information
- ⚠️ May enable fingerprinting
- ⚠️ Not privacy-preserving

### V2: Core Graph
```json
{
  "exposure_surface": {
    "os_platforms": {
      "os_type_distribution": {
        "Windows": 3,
        "Linux": 1
      }
    }
  }
}
```
- ✅ Generic types only
- ✅ NO PII
- ✅ Privacy-preserving
- ✅ Consortium-safe

## Complete Pipeline

### Internal Use
```bash
python scripts/main.py                  # Collect from Wazuh
python scripts/extract_nodes.py         # Extract nodes
python scripts/build_graph.py           # Build Node KG
python scripts/aggregate_data.py        # V1: Internal
```

### Consortium Use
```bash
python scripts/main.py                  # Collect from Wazuh
python scripts/extract_nodes.py         # Extract nodes
python scripts/build_graph.py           # Build Node KG
python scripts/aggregate_data_v2.py     # V2: Privacy-preserving
python scripts/detect_pii.py            # Verify no PII (future)
python scripts/build_core_graph.py      # Build Core Graph (future)
```

## Output Structure

### V1 Output
```
aggregated_data/
└── {timestamp}/
    ├── complete_aggregation.json
    ├── host_aggregation.json
    ├── software_aggregation.json
    ├── vulnerability_aggregation.json
    ├── hardware_aggregation.json
    ├── asset_aggregation.json
    ├── assetgroup_aggregation.json
    └── summary_report.txt
```

### V2 Output
```
aggregated_data_core/
└── {timestamp}/
    ├── core_aggregation.json       # Complete (all 3 layers)
    ├── exposure_surface.json       # Layer 1: Technology
    ├── sensitivity_surface.json    # Layer 2: Risk
    ├── outcome_metrics.json        # Layer 3: Health
    └── summary_report.txt
```

## V2 Three-Layer Architecture

### Layer 1: Exposure Surface
What technology exists (aggregated)
```json
{
  "software_packages": {
    "package_distribution": {
      "python3-libs": 3,
      "openssl": 5
    }
  },
  "os_platforms": {
    "os_type_distribution": {
      "Linux": 1,
      "Windows": 3
    }
  }
}
```

### Layer 2: Sensitivity Surface
What vulnerabilities exist (aggregated)
```json
{
  "vulnerability_exposure": {
    "severity_distribution": {
      "Critical": 2,
      "High": 27,
      "Medium": 16
    },
    "risk_score": 3.57
  }
}
```

### Layer 3: Outcome Metrics
How healthy are assets (aggregated)
```json
{
  "asset_health": {
    "total_assets": 5,
    "health_score": 20.0,
    "active_percentage": 20.0
  }
}
```

## Privacy Verification

V2 includes automatic privacy verification:

```
PRIVACY VERIFICATION:
  Contains PII: False
  Privacy Compliant: True
```

## Documentation

| Document | Description |
|----------|-------------|
| `docs/PRIVACY_PRESERVING_AGGREGATION.md` | Complete V2 documentation |
| `AGGREGATION_COMPARISON.md` | V1 vs V2 comparison |
| `PRIVACY_IMPLEMENTATION_SUMMARY.md` | Implementation details |
| `docs/ARCHITECTURE_DIAGRAM.md` | Visual architecture |
| `IMPLEMENTATION_COMPLETE.md` | Summary and status |

## Configuration

Edit `config/aggregation_config.yaml` to customize:
- Which aggregations to run
- Top N limits
- Output formats
- Future: PII detection settings
- Future: Core graph settings

## When to Use Which?

### Use V1 When:
- Building internal dashboards
- Need detailed version tracking
- Want vendor information
- Analyzing your own data only
- Not sharing with external parties

### Use V2 When:
- Submitting to Core Graph
- Sharing with consortium
- Privacy compliance required
- GDPR/CCPA compliance needed
- Multi-tenant environments

## Key Principles

### Node KG (Local Reality)
- Detailed operational data
- Contains PII (internal use only)
- Full detail for investigations

### Core Graph (Strategic Abstraction)
- Aggregated intelligence
- NO PII (consortium-safe)
- Strategic insights only

### Privacy Boundary
- Mathematically enforced
- Automated verification
- Complete audit trail

## Next Steps

### Phase 4: PII Detection
```bash
python scripts/detect_pii.py
```
- Scan for accidental PII
- Defense-in-depth verification
- Audit trail generation

### Phase 5: Core Graph Building
```bash
python scripts/build_core_graph.py
```
- Build high-level analytics graph
- Separate Neo4j database ("core")
- Link to MERU (CVE, CWE, CAPEC, ATT&CK)

## Support

For questions or issues:
1. Check documentation in `docs/`
2. Review comparison in `AGGREGATION_COMPARISON.md`
3. See examples in output folders

## Status

- ✅ V1: Complete and tested
- ✅ V2: Complete and tested
- ✅ Privacy: Verified
- ✅ Documentation: Complete
- ⏳ Phase 4: PII Detection (next)
- ⏳ Phase 5: Core Graph (next)
