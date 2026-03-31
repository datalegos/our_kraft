# Complete Architecture Diagram

## Data Flow: Wazuh → Node KG → Core Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DAY 0: DATA COLLECTION                       │
│                                                                       │
│  ┌──────────┐                                                        │
│  │  Wazuh   │  ──────────────────────────────────────────────────►  │
│  │ Manager  │         scripts/main.py                                │
│  │ Indexer  │                                                        │
│  └──────────┘                                                        │
│                              │                                        │
│                              ▼                                        │
│                    collected_data/                                   │
│                    └── 20260210_201344/                              │
│                        ├── agents_manager/                           │
│                        ├── host/                                     │
│                        ├── packages/                                 │
│                        ├── vulnerabilities/                          │
│                        └── hardware/                                 │
│                                                                       │
│  ⚠️  Contains: Hostnames, Agent IDs, IP addresses (PII)             │
└─────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────┐
│                      DAY 1: DATA EXTRACTION                          │
│                                                                       │
│                    scripts/extract_nodes.py                          │
│                              │                                        │
│                              ▼                                        │
│                    extracted_data/                                   │
│                    └── 20260215_161934_530/                          │
│                        └── nodes/                                    │
│                            ├── asset_nodes.json                      │
│                            ├── host_nodes.json                       │
│                            ├── software_nodes.json                   │
│                            ├── vulnerability_nodes.json              │
│                            └── hardware_nodes.json                   │
│                                                                       │
│  ⚠️  Still Contains: Hostnames, Agent IDs (PII)                     │
└─────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────┐
│                    DAY 2: NODE KG (DETAILED GRAPH)                   │
│                                                                       │
│                    scripts/build_graph.py                            │
│                              │                                        │
│                              ▼                                        │
│                    ┌─────────────────┐                               │
│                    │   Neo4j (Node)  │                               │
│                    │                 │                               │
│                    │  (Asset)        │                               │
│                    │     │           │                               │
│                    │     ├─RUNS─>(Host)                              │
│                    │     ├─INSTALLED_ON─>(Software)                  │
│                    │     ├─HAS_VULNERABILITY─>(Vulnerability)        │
│                    │     └─HAS_HARDWARE─>(Hardware)                  │
│                    │                 │                               │
│                    └─────────────────┘                               │
│                                                                       │
│  ⚠️  Contains: Full detail, PII (for authorized internal use)       │
│  ✅  Purpose: "Local Reality" - Detailed operational data            │
└─────────────────────────────────────────────────────────────────────┘

                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼

┌──────────────────────────────┐  ┌──────────────────────────────┐
│   DAY 3A: INTERNAL ANALYTICS │  │  DAY 3B: CORE GRAPH PREP     │
│                              │  │                              │
│  aggregate_data.py (V1)      │  │  aggregate_data_v2.py (V2)   │
│           │                  │  │           │                  │
│           ▼                  │  │           ▼                  │
│  aggregated_data/            │  │  aggregated_data_core/       │
│  └── 20260216_152106/        │  │  └── 20260216_153052/        │
│      ├── complete_*.json     │  │      ├── core_aggregation    │
│      ├── host_*.json         │  │      ├── exposure_surface    │
│      ├── software_*.json     │  │      ├── sensitivity_surface │
│      ├── vulnerability_*.json│  │      └── outcome_metrics     │
│      └── summary_report.txt  │  │                              │
│                              │  │                              │
│  ⚠️  Contains: Detailed data │  │  ✅  NO PII                  │
│  ⚠️  May have PII            │  │  ✅  Privacy-Preserving      │
│  ✅  Use: Internal dashboards│  │  ✅  Use: Core Graph         │
└──────────────────────────────┘  └──────────────────────────────┘

                                                │
                                                ▼

                                  ┌──────────────────────────────┐
                                  │  DAY 4: PII DETECTION        │
                                  │                              │
                                  │  scripts/detect_pii.py       │
                                  │  (Presidio)                  │
                                  │           │                  │
                                  │           ▼                  │
                                  │  filtered_data/              │
                                  │  └── Verified no PII         │
                                  │                              │
                                  │  ✅  Defense-in-depth        │
                                  │  ✅  Audit trail             │
                                  └──────────────────────────────┘

                                                │
                                                ▼

                                  ┌──────────────────────────────┐
                                  │  DAY 5: CORE GRAPH           │
                                  │                              │
                                  │  scripts/build_core_graph.py │
                                  │           │                  │
                                  │           ▼                  │
                                  │  ┌─────────────────┐         │
                                  │  │ Neo4j (Core DB) │         │
                                  │  │                 │         │
                                  │  │ (OSDistribution)│         │
                                  │  │   - Linux: 1    │         │
                                  │  │   - Windows: 3  │         │
                                  │  │                 │         │
                                  │  │ (SoftwarePackage)│        │
                                  │  │   - python3: 1  │         │
                                  │  │   - openssl: 1  │         │
                                  │  │                 │         │
                                  │  │ (VulnSeverity)  │         │
                                  │  │   - Critical: 2 │         │
                                  │  │   - High: 27    │         │
                                  │  │                 │         │
                                  │  │ + MERU Link     │         │
                                  │  │   (CVE, CWE,    │         │
                                  │  │    CAPEC, ATT&CK)│        │
                                  │  └─────────────────┘         │
                                  │                              │
                                  │  ✅  NO PII                  │
                                  │  ✅  Aggregates only         │
                                  │  ✅  Strategic abstraction   │
                                  └──────────────────────────────┘
```

## Privacy Boundary

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRIVACY BOUNDARY                            │
│                                                                   │
│  ABOVE THIS LINE: Contains PII (Internal Use Only)               │
│  ═══════════════════════════════════════════════════════════════ │
│  BELOW THIS LINE: NO PII (Safe for Consortium Sharing)           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

INTERNAL (Node KG):
  - collected_data/      ⚠️  PII
  - extracted_data/      ⚠️  PII
  - Neo4j (Node DB)      ⚠️  PII
  - aggregated_data/     ⚠️  May have PII

═══════════════════════════════════════════════════════════════════

CONSORTIUM (Core Graph):
  - aggregated_data_core/  ✅  NO PII
  - filtered_data/         ✅  NO PII (verified)
  - Neo4j (Core DB)        ✅  NO PII
```

## Data Transformation Example

### Source Data (PII)
```json
{
  "asset_id": "001",
  "asset_name": "vishnu",
  "os_name": "Microsoft Windows 11 Home",
  "os_version": "10.0.26200.7623",
  "software": ["python3-libs-3.9.25", "openssl-1.1.1"]
}
```

### V1 Aggregation (Internal)
```json
{
  "host": {
    "os_version_distribution": {
      "Windows-10.0.26200.7623": 2
    }
  },
  "software": {
    "top_packages": {
      "python3-libs": 3,
      "openssl": 5
    }
  }
}
```

### V2 Aggregation (Core Graph)
```json
{
  "exposure_surface": {
    "os_platforms": {
      "os_type_distribution": {
        "Windows": 3
      }
    },
    "software_packages": {
      "package_distribution": {
        "python3-libs": 3,
        "openssl": 5
      }
    }
  }
}
```

## Three-Layer Core Graph Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: EXPOSURE SURFACE                     │
│                  (Technology Exposure Graph)                     │
│                                                                   │
│  What technology exists in the consortium?                       │
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ OSDistribution│     │SoftwarePackage│    │ Architecture │    │
│  │              │     │              │     │              │    │
│  │ Linux: 1     │     │ python3: 3   │     │ x86_64: 4    │    │
│  │ Windows: 3   │     │ openssl: 5   │     │ arm64: 1     │    │
│  │ macOS: 1     │     │ curl: 4      │     │              │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 2: SENSITIVITY SURFACE                    │
│                   (Aggregated Risk Amplifier)                    │
│                                                                   │
│  What vulnerabilities exist (aggregated)?                        │
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │VulnSeverity  │     │  Risk Score  │     │   Top CVEs   │    │
│  │              │     │              │     │              │    │
│  │ Critical: 2  │     │   3.57/10    │     │ CVE-2022-... │    │
│  │ High: 27     │     │              │     │ CVE-2025-... │    │
│  │ Medium: 16   │     │              │     │ CVE-2026-... │    │
│  │ Low: 2       │     │              │     │              │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 3: OUTCOME METRICS                       │
│                  (Effectiveness Feedback)                        │
│                                                                   │
│  How healthy is the consortium?                                  │
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ Asset Health │     │ Health Score │     │Active Assets │    │
│  │              │     │              │     │              │    │
│  │ Total: 5     │     │   20.0%      │     │   20.0%      │    │
│  │ Active: 1    │     │              │     │              │    │
│  │ Disconn: 4   │     │              │     │              │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Principles

### Node KG (Local Reality)
- **Purpose**: Detailed operational data
- **Audience**: Internal teams only
- **Contains**: Full detail, PII
- **Database**: Neo4j (default database)
- **Use Cases**: 
  - Incident response
  - Asset management
  - Compliance audits
  - Detailed investigations

### Core Graph (Strategic Abstraction)
- **Purpose**: Consortium-wide intelligence
- **Audience**: All consortium members
- **Contains**: Aggregates only, NO PII
- **Database**: Neo4j (core database)
- **Use Cases**:
  - Threat intelligence sharing
  - Industry benchmarking
  - Strategic planning
  - Risk assessment

### Privacy Boundary
- **Enforcement**: Mathematical (k-anonymity)
- **Verification**: Automated (Presidio)
- **Audit**: Complete trail
- **Compliance**: GDPR, CCPA, HIPAA

## Summary

This architecture ensures:

1. ✅ **Privacy**: PII never leaves Node KG
2. ✅ **Utility**: Core Graph still provides strategic insights
3. ✅ **Compliance**: Meets all regulatory requirements
4. ✅ **Trust**: Consortium members can verify no PII
5. ✅ **Scalability**: Aggregated data is much smaller
6. ✅ **Flexibility**: Can drill down to Node KG when authorized
