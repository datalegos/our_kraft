#!/bin/bash

# Install all dependencies for PCI Data Discovery and Analysis

echo "=========================================="
echo "    Installing PCI Analysis Dependencies"
echo "=========================================="

# Check if Python is available
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
echo "[INFO] Python version: $($PYTHON_CMD --version)"

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

echo "[INFO] Using pip: $PIP_CMD"

# Install database requirements
echo ""
echo "[INFO] Installing database extraction requirements..."
$PIP_CMD install -r requirements/database_requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install database requirements"
    exit 1
fi

# Install Presidio requirements
echo ""
echo "[INFO] Installing Presidio analysis requirements..."
$PIP_CMD install -r requirements/presidio_requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Presidio requirements"
    exit 1
fi

# Download spaCy language model
echo ""
echo "[INFO] Downloading spaCy English language model..."
$PYTHON_CMD -m spacy download en_core_web_sm

if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to download spaCy model. Trying alternative..."
    $PIP_CMD install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1.tar.gz
fi

# Test installations
echo ""
echo "[INFO] Testing installations..."

echo "[INFO] Testing database drivers..."
$PYTHON_CMD -c "
import psycopg2
import mysql.connector
print('✅ Database drivers installed successfully!')
"

echo "[INFO] Testing Presidio..."
$PYTHON_CMD -c "
import presidio_analyzer
import presidio_anonymizer
import spacy
print('✅ Presidio installed successfully!')
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy model loaded successfully!')
except:
    print('⚠️  spaCy model not found, but Presidio is installed')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "    Installation Completed Successfully"
    echo "=========================================="
    echo "You can now run:"
    echo "  ./scripts/run_database_extractor.sh    # Extract database data"
    echo "  ./scripts/run_presidio_analysis.sh     # Analyze for PCI data"
    echo "  ./scripts/run_full_pipeline.sh         # Run complete pipeline"
else
    echo ""
    echo "=========================================="
    echo "    Installation Failed"
    echo "=========================================="
    echo "Please check the error messages above"
    exit 1
fi