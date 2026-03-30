# Installing and Testing Presidio PII Detection

## Quick Installation

### Step 1: Install Presidio and Dependencies
```bash
pip install -r requirements_presidio.txt
```

This installs:
- `presidio-analyzer` - PII detection engine
- `presidio-anonymizer` - PII anonymization (for future use)
- `spacy` - NLP engine

### Step 2: Download spaCy Language Model
```bash
python -m spacy download en_core_web_lg
```

This downloads the large English language model (~800MB) needed for accurate entity recognition.

### Step 3: Verify Installation
```bash
python -c "from presidio_analyzer import AnalyzerEngine; print('✅ Presidio installed successfully')"
```

## Test the PII Scanner

### Test 1: Scan Your Aggregated Data
```bash
python scripts/detect_pii.py
```

Expected output:
```
2026-02-16 15:45:30,123 - INFO - Initializing Presidio Analyzer...
2026-02-16 15:45:32,456 - INFO - Presidio Analyzer initialized successfully
2026-02-16 15:45:32,457 - INFO - Using current session: 20260216_153052
2026-02-16 15:45:32,458 - INFO - Scanning directory: aggregated_data_core\20260216_153052
2026-02-16 15:45:32,459 - INFO - Found 5 JSON files to scan
2026-02-16 15:45:32,460 - INFO - Scanning file: core_aggregation.json
...
================================================================================
SCAN COMPLETE
================================================================================
Total Files Scanned: 5
Files with PII/PCI: 0
Total PII/PCI Findings: 0

✅ PRIVACY STATUS: COMPLIANT
✅ SAFE FOR CORE GRAPH: YES
✅ NO PII/PCI DETECTED

Results saved to: pii_scan_results\20260216_154532
================================================================================
```

### Test 2: Scan Specific Directory
```bash
python scripts/detect_pii.py --input aggregated_data_core/20260216_153052
```

### Test 3: Adjust Confidence Threshold
```bash
# More sensitive (may have false positives)
python scripts/detect_pii.py --confidence 0.3

# Less sensitive (fewer false positives)
python scripts/detect_pii.py --confidence 0.7
```

## Check the Results

### View Summary Report
```bash
# Windows
type pii_scan_results\{timestamp}\pii_scan_summary.txt

# Linux/Mac
cat pii_scan_results/{timestamp}/pii_scan_summary.txt
```

### View JSON Results
```bash
# Windows
type pii_scan_results\{timestamp}\pii_scan_results.json

# Linux/Mac
cat pii_scan_results/{timestamp}/pii_scan_results.json
```

## Expected Results

### For Privacy-Preserving Aggregation (V2)
You should see:
```
✅ PRIVACY STATUS: COMPLIANT
✅ SAFE FOR CORE GRAPH: YES
✅ NO PII/PCI DETECTED
```

All files should show:
```
FILE-BY-FILE SUMMARY
--------------------------------------------------------------------------------
  core_aggregation.json: ✅ CLEAN (0 findings)
  exposure_surface.json: ✅ CLEAN (0 findings)
  sensitivity_surface.json: ✅ CLEAN (0 findings)
  outcome_metrics.json: ✅ CLEAN (0 findings)
  summary_report.txt: ✅ CLEAN (0 findings)
```

### If PII is Detected
You'll see:
```
❌ PRIVACY STATUS: NON-COMPLIANT
❌ SAFE FOR CORE GRAPH: NO
❌ PII/PCI DETECTED - REVIEW REQUIRED
```

And detailed findings showing:
- Entity type (PERSON, EMAIL_ADDRESS, etc.)
- Confidence score
- Location in JSON
- Detected text

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'presidio_analyzer'"
**Solution:**
```bash
pip install presidio-analyzer presidio-anonymizer
```

### Issue: "Can't find model 'en_core_web_lg'"
**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Issue: "aggregated_data_core folder not found"
**Solution:**
```bash
# Run aggregation first
python scripts/aggregate_data_v2.py
```

### Issue: Scan is very slow
**Solution:**
- Use smaller confidence threshold
- Scan specific files only
- Use smaller spaCy model (en_core_web_sm) for testing

### Issue: Too many false positives
**Solution:**
```bash
# Increase confidence threshold
python scripts/detect_pii.py --confidence 0.7
```

## Integration Test

Run the complete pipeline:

```bash
# Step 1: Create privacy-preserving aggregates
python scripts/aggregate_data_v2.py

# Step 2: Scan for PII
python scripts/detect_pii.py

# Step 3: Check exit code
echo Exit Code: %ERRORLEVEL%  # Windows
echo Exit Code: $?             # Linux/Mac
```

Expected:
- Exit Code: 0 (no PII detected)
- Summary shows "✅ SAFE FOR CORE GRAPH: YES"

## Performance Benchmarks

On typical hardware:

| Dataset Size | Files | Time | Memory |
|--------------|-------|------|--------|
| Small | 5 files, <1MB | ~5-10s | ~500MB |
| Medium | 20 files, <10MB | ~30-60s | ~800MB |
| Large | 100 files, <100MB | ~5-10min | ~1.5GB |

## Next Steps

After successful installation and testing:

1. ✅ Presidio installed and working
2. ✅ PII scan passes (no PII detected)
3. ➡️ Proceed to Phase 5: Core Graph Building
4. ➡️ Run: `python scripts/build_core_graph.py` (coming next)

## Alternative: Docker Installation

If you prefer Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_presidio.txt .
RUN pip install -r requirements_presidio.txt
RUN python -m spacy download en_core_web_lg

# Copy scripts
COPY scripts/ scripts/
COPY aggregated_data_core/ aggregated_data_core/

# Run scan
CMD ["python", "scripts/detect_pii.py"]
```

Build and run:
```bash
docker build -t pii-scanner .
docker run -v $(pwd)/aggregated_data_core:/app/aggregated_data_core pii-scanner
```

## Support

For issues:
1. Check `docs/PII_DETECTION.md` for detailed documentation
2. Review Presidio documentation: https://microsoft.github.io/presidio/
3. Check spaCy documentation: https://spacy.io/
