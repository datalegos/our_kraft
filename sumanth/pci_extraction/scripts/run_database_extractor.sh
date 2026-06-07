#!/bin/bash

# Database Extractor Runner Script

echo "=========================================="
echo "    Database Discovery & Data Extractor"
echo "=========================================="

# Find available Python command
PYTHON_CMD=""
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py"
else
    echo "[ERROR] Python is not installed or not in PATH"
    echo "Please install Python or add it to your PATH"
    exit 1
fi

echo "[INFO] Using Python command: $PYTHON_CMD"
echo "[INFO] Python version: $($PYTHON_CMD --version)"

# Check if config file exists
CONFIG_FILE="config/extraction_config.yml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Configuration file '$CONFIG_FILE' not found!"
    echo "Please ensure the config file exists"
    exit 1
fi

# Check if dependencies are installed
echo "[INFO] Checking dependencies..."
$PYTHON_CMD -c "import yaml, psycopg2, mysql.connector" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] Some dependencies are missing"
    echo "[INFO] Installing dependencies..."
    
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
    
    echo "[INFO] Installing with: $PIP_CMD"
    $PIP_CMD install -r requirements/database_requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
fi

echo "[INFO] Starting database extraction..."
echo "[INFO] Configuration: $CONFIG_FILE"
echo "[INFO] Timestamp: $(date)"
echo ""

# Run the extractor from src folder
$PYTHON_CMD src/database_extractor.py -c "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "    Extraction Completed Successfully"
    echo "=========================================="
    echo "Check the data/extracted_data/ directory for results"
    echo "Check the logs/ directory for detailed logs"
else
    echo ""
    echo "=========================================="
    echo "    Extraction Failed"
    echo "=========================================="
    echo "Check the logs for error details"
    exit 1
fi