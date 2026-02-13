#!/bin/bash

# Complete PCI Data Discovery and Analysis Pipeline

echo "=========================================="
echo "    PCI Data Discovery & Analysis Pipeline"
echo "=========================================="
echo "This script will:"
echo "1. Extract data from databases"
echo "2. Analyze extracted data for PCI/PII content"
echo "3. Generate compliance reports"
echo "=========================================="
echo ""

# Step 1: Run Database Extraction
echo "STEP 1: Database Data Extraction"
echo "================================="
./scripts/run_database_extractor.sh

if [ $? -ne 0 ]; then
    echo "[ERROR] Database extraction failed. Stopping pipeline."
    exit 1
fi

echo ""
echo "STEP 2: PCI Data Analysis"
echo "========================="
./scripts/run_presidio_analysis.sh

if [ $? -ne 0 ]; then
    echo "[ERROR] PCI analysis failed. Stopping pipeline."
    exit 1
fi

echo ""
echo "=========================================="
echo "    PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "- Extracted Data: data/extracted_data/"
echo "- PCI Analysis: data/pci_analysis_results/"
echo "- Logs: logs/"
echo ""
echo "Next Steps:"
echo "1. Review the HTML dashboard for executive summary"
echo "2. Check detailed JSON reports for specific findings"
echo "3. Use CSV exports for further analysis"
echo "=========================================="