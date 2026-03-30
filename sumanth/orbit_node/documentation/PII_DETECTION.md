# PII/PCI Detection with Microsoft Presidio

## Overview

The PII detection module uses Microsoft Presidio to scan aggregated data for sensitive information before sending to Core Graph. This provides defense-in-depth verification that no PII/PCI data leaked through the aggregation process.

## Installation

### Step 1: Install Presidio
```bash
pip install -r requirements_presidio.txt
```

### Step 2: Download spaCy Language Model
```bash
python -m spacy download en_core_web_lg
```

## Usage

### Basic Usage (Scan Latest Aggregated Data)
```bash
python scripts/detect_pii.py
```

This will:
1. Find the latest folder in `aggregated_data_core/`
2. Scan all JSON files for PII/PCI
3. Generate detailed report in `pii_scan_results/`

### Advanced Usage

**Scan Specific Directory:**
```bash
python scripts/detect_pii.py --input aggregated_data_core/20260216_153052
```

**Custom Output Directory:**
```bash
python scripts/detect_pii.py --output my_scan_results
```

**Adjust Confidence Threshold:**
```bash
python scripts/detect_pii.py --confidence 0.7
```
- Lower threshold (0.3): More sensitive, more false positives
- Higher threshold (0.7): Less sensitive, fewer false positives
- Default: 0.5 (balanced)

## What Gets Detected

### Personal Information
- PERSON - Names of individuals
- EMAIL_ADDRESS - Email addresses
- PHONE_NUMBER - Phone numbers
- US_SSN - Social Security Numbers
- US_PASSPORT - Passport numbers
- US_DRIVER_LICENSE - Driver's license numbers

### Financial Information (PCI)
- CREDIT_CARD - Credit card numbers
- IBAN_CODE - International Bank Account Numbers
- US_BANK_NUMBER - Bank account numbers
- CRYPTO - Cryptocurrency addresses

### Network Information
- IP_ADDRESS - IPv4/IPv6 addresses
- URL - Web URLs
- DOMAIN_NAME - Domain names

### Location Information
- LOCATION - Physical locations
- US_ZIP_CODE - ZIP codes

### Medical Information
- MEDICAL_LICENSE - Medical license numbers

### Other Identifiers
- DATE_TIME - Dates and times (can be PII in context)
- NRP - National Registry of Persons
- AU_ABN - Australian Business Number
- AU_ACN - Australian Company Number
- AU_TFN - Australian Tax File Number
- AU_MEDICARE - Australian Medicare numbers

## Output Structure

```
pii_scan_results/
└── {timestamp}/
    ├── pii_scan_results.json       # Complete scan results (JSON)
    ├── pii_scan_summary.txt        # Human-readable summary
    └── detailed_findings.json      # Detailed findings (if any PII found)
```

## Example Output

### Clean Scan (No PII)
```
================================================================================
PII/PCI DETECTION SCAN REPORT
================================================================================
Scan Timestamp: 2026-02-16T15:45:30.123456
Directory Scanned: aggregated_data_core/20260216_153052
================================================================================

SCAN SUMMARY
--------------------------------------------------------------------------------
Total Files Scanned: 5
Files with PII/PCI: 0
Total PII/PCI Findings: 0

✅ PRIVACY STATUS: COMPLIANT
✅ SAFE FOR CORE GRAPH: YES
✅ NO PII/PCI DETECTED

FILE-BY-FILE SUMMARY
--------------------------------------------------------------------------------
  core_aggregation.json: ✅ CLEAN (0 findings)
  exposure_surface.json: ✅ CLEAN (0 findings)
  sensitivity_surface.json: ✅ CLEAN (0 findings)
  outcome_metrics.json: ✅ CLEAN (0 findings)
  summary_report.txt: ✅ CLEAN (0 findings)

================================================================================
RECOMMENDATIONS
================================================================================

✅ Data is safe to send to Core Graph
✅ No PII/PCI detected
✅ Privacy compliance verified

Next Steps:
  1. Proceed with Core Graph submission
  2. Run: python scripts/build_core_graph.py

================================================================================
```

### PII Detected
```
================================================================================
PII/PCI DETECTION SCAN REPORT
================================================================================

SCAN SUMMARY
--------------------------------------------------------------------------------
Total Files Scanned: 5
Files with PII/PCI: 2
Total PII/PCI Findings: 5

❌ PRIVACY STATUS: NON-COMPLIANT
❌ SAFE FOR CORE GRAPH: NO
❌ PII/PCI DETECTED - REVIEW REQUIRED

DETECTED ENTITY TYPES
--------------------------------------------------------------------------------
  PERSON: 3
  EMAIL_ADDRESS: 1
  IP_ADDRESS: 1

FILE-BY-FILE SUMMARY
--------------------------------------------------------------------------------
  core_aggregation.json: ❌ PII FOUND (3 findings)
    - PERSON: 2
    - EMAIL_ADDRESS: 1
  exposure_surface.json: ❌ PII FOUND (2 findings)
    - PERSON: 1
    - IP_ADDRESS: 1
  sensitivity_surface.json: ✅ CLEAN (0 findings)

DETAILED FINDINGS
--------------------------------------------------------------------------------

Finding #1:
  Entity Type: PERSON
  Confidence: 0.85
  Location: core_aggregation.json.metadata.source_path
  Text: vishnu

Finding #2:
  Entity Type: EMAIL_ADDRESS
  Confidence: 0.92
  Location: core_aggregation.json.contact
  Text: admin@example.com

================================================================================
RECOMMENDATIONS
================================================================================

❌ DO NOT send this data to Core Graph
❌ PII/PCI detected - must be removed or anonymized

Required Actions:
  1. Review detailed findings in detailed_findings.json
  2. Remove or anonymize detected PII/PCI
  3. Re-run aggregation with privacy fixes
  4. Re-scan with: python scripts/detect_pii.py

================================================================================
```

## Integration with Pipeline

### Complete Pipeline with PII Detection
```bash
# Step 1: Collect data
python scripts/main.py

# Step 2: Extract nodes
python scripts/extract_nodes.py

# Step 3: Build Node KG
python scripts/build_graph.py

# Step 4: Create privacy-preserving aggregates
python scripts/aggregate_data_v2.py

# Step 5: Scan for PII/PCI (defense-in-depth)
python scripts/detect_pii.py

# Step 6: If clean, build Core Graph
python scripts/build_core_graph.py
```

## How It Works

### 1. Recursive JSON Scanning
The scanner recursively traverses all JSON structures:
- Dictionary keys and values
- List items
- Nested objects
- String values

### 2. NLP Analysis
Uses spaCy's NLP engine to:
- Tokenize text
- Identify named entities
- Apply pattern matching
- Calculate confidence scores

### 3. Entity Recognition
Presidio's recognizers detect:
- Pattern-based entities (e.g., credit cards, SSNs)
- Context-based entities (e.g., names, locations)
- Custom entities (configurable)

### 4. Confidence Scoring
Each detection includes a confidence score (0.0 to 1.0):
- 0.0-0.3: Low confidence (likely false positive)
- 0.3-0.7: Medium confidence (review recommended)
- 0.7-1.0: High confidence (likely true positive)

## Configuration

Edit `config/aggregation_config.yaml` to customize:

```yaml
pii_detection:
  enabled: true
  confidence_threshold: 0.5
  entities_to_detect:
    - PERSON
    - EMAIL_ADDRESS
    - CREDIT_CARD
    # ... add more entity types
```

## False Positives

Presidio may flag non-PII data as PII:

### Common False Positives
- **PERSON**: Generic words like "Windows", "Linux" (rare)
- **DATE_TIME**: Version numbers like "2023.1.0"
- **IP_ADDRESS**: Version numbers like "10.0.1"

### Handling False Positives
1. Review `detailed_findings.json`
2. Check context of each finding
3. Adjust confidence threshold if needed
4. Add custom filters (advanced)

## Exit Codes

- `0`: No PII detected (safe for Core Graph)
- `1`: PII detected OR scan error (DO NOT send to Core Graph)

## Automation

### CI/CD Integration
```bash
#!/bin/bash
# Run aggregation and PII scan
python scripts/aggregate_data_v2.py
python scripts/detect_pii.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ PII scan passed - proceeding to Core Graph"
    python scripts/build_core_graph.py
else
    echo "❌ PII scan failed - blocking Core Graph submission"
    exit 1
fi
```

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
if [ -d "aggregated_data_core" ]; then
    python scripts/detect_pii.py
    if [ $? -ne 0 ]; then
        echo "❌ PII detected in aggregated data - commit blocked"
        exit 1
    fi
fi
```

## Performance

### Scan Times (Approximate)
- Small dataset (5 files, <1MB): ~5-10 seconds
- Medium dataset (20 files, <10MB): ~30-60 seconds
- Large dataset (100 files, <100MB): ~5-10 minutes

### Optimization Tips
1. Use higher confidence threshold (0.7) for faster scans
2. Scan only specific files if needed
3. Run in parallel for multiple directories

## Troubleshooting

### Error: "Presidio not installed"
```bash
pip install -r requirements_presidio.txt
```

### Error: "spaCy model not found"
```bash
python -m spacy download en_core_web_lg
```

### Error: "No aggregated_data_core folder"
```bash
# Run aggregation first
python scripts/aggregate_data_v2.py
```

### High False Positive Rate
```bash
# Increase confidence threshold
python scripts/detect_pii.py --confidence 0.7
```

## Best Practices

1. **Always scan before Core Graph submission**
2. **Review detailed findings** if PII is detected
3. **Use appropriate confidence threshold** (0.5 default, 0.7 for production)
4. **Automate in CI/CD** to prevent accidental PII leaks
5. **Keep audit trail** of all scans

## Security Considerations

- Scan results may contain PII (in detailed findings)
- Store scan results securely
- Limit access to scan results
- Delete scan results after review
- Never commit scan results to version control

## Next Steps

After successful PII scan:
1. Verify `✅ SAFE FOR CORE GRAPH: YES`
2. Proceed to Phase 5: Core Graph Building
3. Run: `python scripts/build_core_graph.py`
