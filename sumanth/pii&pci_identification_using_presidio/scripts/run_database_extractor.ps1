# PowerShell Database Extractor Runner Script

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "    Database Discovery & Data Extractor" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue

# Activate conda orbit environment
Write-Host "[INFO] Activating conda orbit environment..." -ForegroundColor Yellow
try {
    & conda activate orbit
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate orbit environment"
    }
    Write-Host "[SUCCESS] Activated orbit environment" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Failed to activate orbit conda environment" -ForegroundColor Red
    Write-Host "Please create the environment: conda create -n orbit python=3.9" -ForegroundColor Yellow
    exit 1
}

# Use python from orbit environment
$pythonCmd = "python"
Write-Host "[INFO] Using Python from orbit environment" -ForegroundColor Green

# Check if config file exists
$configFile = "config/extraction_config.yml"
if (-not (Test-Path $configFile)) {
    Write-Host "[ERROR] Configuration file '$configFile' not found!" -ForegroundColor Red
    exit 1
}

# Check if dependencies are installed
Write-Host "[INFO] Checking dependencies..." -ForegroundColor Yellow
try {
    & $pythonCmd -c "import yaml, psycopg2, mysql.connector" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Dependencies missing"
    }
    Write-Host "[INFO] All dependencies are available" -ForegroundColor Green
}
catch {
    Write-Host "[WARNING] Some dependencies are missing" -ForegroundColor Yellow
    Write-Host "[INFO] Installing dependencies in orbit environment..." -ForegroundColor Yellow
    
    & pip install -r requirements/database_requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[SUCCESS] Dependencies installed successfully" -ForegroundColor Green
}

Write-Host "[INFO] Starting database extraction..." -ForegroundColor Yellow
Write-Host "[INFO] Configuration: $configFile" -ForegroundColor Cyan
Write-Host "[INFO] Environment: orbit" -ForegroundColor Cyan
Write-Host "[INFO] Timestamp: $(Get-Date)" -ForegroundColor Cyan
Write-Host ""

# Run the extractor from src folder
try {
    & $pythonCmd src/database_extractor.py -c $configFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "==========================================" -ForegroundColor Blue
        Write-Host "    Extraction Completed Successfully" -ForegroundColor Blue
        Write-Host "==========================================" -ForegroundColor Blue
        Write-Host "Check the data/extracted_data/ directory for results" -ForegroundColor Green
        Write-Host "Check the logs/ directory for detailed logs" -ForegroundColor Green
    }
    else {
        throw "Extraction failed"
    }
}
catch {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "    Extraction Failed" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "Check the logs for error details" -ForegroundColor Red
    exit 1
}