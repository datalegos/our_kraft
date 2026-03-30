# Privacy-Preserving Aggregation for Core Graph

## Problem Statement

The original aggregation (`aggregate_data.py`) violated Core Graph privacy principles by storing:

❌ **What Core Should NOT Store:**
- Hostnames (e.g., "wazuh.manager", "vishnu", "naveen")
- Agent IDs (e.g., "000", "001", "002")
- Per-host software lists
- Per-host vulnerability mappings
- Vendor-specific information that could identify organizations
- Any PII or execution logs

## Solution: Privacy-Preserving Aggregation v2

The new aggregation (`aggregate_data_v2.py`) follows strict Core Graph principles:

✅ **What Core SHOULD Store:**
- Aggregated counts only (no instances)
- Technology exposure surface
- Sensitivity surface (risk amplifiers)
- Outcome metrics (effectiveness feedback)

## Three-Layer Architecture

### Layer 1: Technology Exposure Surface

**Purpose**: What technology exists in the consortium (aggregated, anonymized)

**Data Stored**:
```json
{
  "software_packages": {
    "total_instances": 726,
    "unique_packages": 583,
    "package_distribution": {
      "python3-libs": 3,
      "openssl": 5,
      "curl": 4
    },
    "format_distribution": {
      "rpm": 116,
      "deb": 45,
      "pkg": 278
    }
  },
  "os_platforms": {
    "total_instances": 5,
    "os_type_distribution": {
      "Linux": 1,
      "Windows": 3,
      "macOS": 1
    }
  }
}
```

**Privacy Guarantees**:
- NO hostnames
- NO agent IDs
- NO per-host mappings
- Only aggregated counts

### Layer 2: Sensitivity Surface (Risk Amplifier)

**Purpose**: What vulnerabilities exist (aggregated, no per-host mapping)

**Data Stored**:
```json
{
  "vulnerability_exposure": {
    "total_vulnerabilities": 53,
    "unique_cves": 40,
    "severity_distribution": {
      "Critical": 2,
      "High": 27,
      "Medium": 16,
      "Low": 2
    },
    "risk_score": 3.57,
    "critical_exposure": 2,
    "high_exposure": 27
  }
}
```

**Privacy Guarantees**:
- NO per-host vulnerability lists
- NO mapping of CVE to specific assets
- Only aggregated severity counts
- Risk score calculated from aggregates

### Layer 3: Outcome Metrics (Effectiveness Feedback)

**Purpose**: Aggregate metrics about asset health (no identifiers)

**Data Stored**:
```json
{
  "asset_health": {
    "total_assets": 5,
    "status_distribution": {
      "active": 1,
      "disconnected": 4
    },
    "health_score": 20.0,
    "active_percentage": 20.0
  }
}
```

**Privacy Guarantees**:
- NO asset names
- NO asset IDs
- Only aggregated health metrics
- Percentage-based scores

## Comparison: Old vs New

### Old Aggregation (aggregate_data.py)

```json
{
  "host": {
    "total_hosts": 5,
    "os_distribution": {
      "Linux": 1,
      "Windows": 3
    },
    "os_version_distribution": {
      "Linux-2023": 1,
      "Windows-10.0.26200.7623": 2
    }
  }
}
```

**Problems**:
- ❌ Stores specific OS versions (fingerprinting risk)
- ❌ Could be correlated with other data sources
- ❌ Reveals organizational technology choices

### New Aggregation (aggregate_data_v2.py)

```json
{
  "exposure_surface": {
    "os_platforms": {
      "total_instances": 5,
      "os_type_distribution": {
        "Linux": 1,
        "Windows": 3,
        "macOS": 1
      }
    }
  }
}
```

**Benefits**:
- ✅ Generic OS types only (no versions)
- ✅ Cannot be fingerprinted
- ✅ Privacy-preserving aggregates
- ✅ Still useful for strategic analysis

## Usage

### Run Privacy-Preserving Aggregation

```bash
python scripts/aggregate_data_v2.py
```

### Output Structure

```
aggregated_data_core/
├── .current_session
└── {timestamp}/
    ├── core_aggregation.json       # Complete aggregation
    ├── exposure_surface.json       # Layer 1
    ├── sensitivity_surface.json    # Layer 2
    ├── outcome_metrics.json        # Layer 3
    └── summary_report.txt          # Human-readable
```

### Verification

Every aggregation includes privacy verification:

```
PRIVACY VERIFICATION:
  Contains PII: False
  Privacy Compliant: True
```

## Integration with Core Graph

The privacy-preserving aggregation is designed to feed directly into the Core Graph:

```
Node KG (Detailed)          Core Graph (Strategic)
==================          ======================
Asset: "vishnu"       →     OS Type: "Windows" (count: 3)
Agent ID: "001"       →     (NO agent IDs stored)
Hostname: "vishnu"    →     (NO hostnames stored)
Software: [list]      →     Package: "python3" (count: 3)
Vulnerabilities: []   →     Severity: "Critical" (count: 2)
```

## Mathematical Privacy Enforcement

The aggregation enforces privacy through:

1. **K-Anonymity**: All data is aggregated (k ≥ total instances)
2. **No Identifiers**: Zero PII stored
3. **No Correlation**: Cannot map back to individual assets
4. **Aggregate-Only**: Core reasons over distributions, not instances

## Next Steps

### Phase 4: PII Detection (Presidio)

Even though v2 aggregation is privacy-preserving, we'll add Presidio scanning as defense-in-depth:

```bash
python scripts/detect_pii.py
```

This will:
- Scan aggregated data for any accidental PII
- Flag any privacy violations
- Provide audit trail

### Phase 5: Core Graph Building

```bash
python scripts/build_core_graph.py
```

This will:
- Create high-level analytics graph
- Use separate Neo4j database ("core")
- Store only aggregated nodes
- Link to MERU (CVE, CWE, CAPEC, ATT&CK)

## Benefits

1. **Privacy**: Mathematically enforced, no PII
2. **Compliance**: GDPR, CCPA, HIPAA compatible
3. **Security**: Cannot be used for reconnaissance
4. **Utility**: Still provides strategic insights
5. **Scalability**: Much smaller data footprint
6. **Trust**: Consortium members can verify no PII

## Validation Checklist

Before sending data to Core Graph, verify:

- [ ] No hostnames present
- [ ] No IP addresses present
- [ ] No agent IDs present
- [ ] No file paths present
- [ ] No per-host data present
- [ ] No vendor-specific identifiers
- [ ] Only aggregated counts
- [ ] Privacy metadata confirms compliance
