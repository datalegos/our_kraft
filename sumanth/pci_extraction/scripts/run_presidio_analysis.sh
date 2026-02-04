#!/bin/bash

# Run Presidio PCI Analysis

echo "=========================================="
echo "    Presidio PCI Data Analysis"
echo "=========================================="

# Find Python command
PYTHON_CMD=""
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "[ERROR] Python is not installed or not in PATH"
    exit 1
fi

echo "[INFO] Using Python: $PYTHON_CMD"

# Check if config file exists
CONFIG_FILE="config/presidio_config.yml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Check if extracted data exists
SOURCE_DIR="./data/extracted_data"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "[ERROR] Source directory '$SOURCE_DIR' not found!"
    echo "Please run the database extractor first"
    exit 1
fi

# Count JSON files
JSON_COUNT=$(find "$SOURCE_DIR" -name "*.json" | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "[ERROR] No JSON files found in '$SOURCE_DIR'"
    echo "Please run the database extractor first"
    exit 1
fi

echo "[INFO] Found $JSON_COUNT JSON files to analyze"

# Check if Presidio is installed
echo "[INFO] Checking Presidio installation..."
$PYTHON_CMD -c "import presidio_analyzer, presidio_anonymizer" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] Presidio not installed. Installing..."
    
    # Find pip command
    PIP_CMD=""
    if command -v pip &> /dev/null; then
        PIP_CMD="pip"
    elif command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    else
        echo "[ERROR] pip is not installed or not in PATH"
        exit 1
    fi
    
    echo "[INFO] Installing Presidio requirements..."
    $PIP_CMD install -r requirements/presidio_requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install Presidio requirements"
        exit 1
    fi
    
    # Download spaCy model
    echo "[INFO] Downloading spaCy model..."
    $PYTHON_CMD -m spacy download en_core_web_sm
fi

echo "[INFO] Starting PCI analysis..."
echo "[INFO] Configuration: $CONFIG_FILE"
echo "[INFO] Source directory: $SOURCE_DIR"
echo "[INFO] Timestamp: $(date)"
echo ""

# Run the analyzer from src folder
$PYTHON_CMD src/pci_analyzer.py -c "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "    Analysis Completed Successfully"
    echo "=========================================="
    echo "Check the data/pci_analysis_results/ directory for:"
    echo "  - Detailed PII findings (JSON)"
    echo "  - Compliance report (JSON)"
    echo "  - Analysis summary (CSV)"
    echo "  - Dashboard report (HTML)"
    echo ""
    echo "Check the logs/ directory for detailed logs"
else
    echo ""
    echo "=========================================="
    echo "    Analysis Failed"
    echo "=========================================="
    echo "Check the logs for error details"
    exit 1
fi