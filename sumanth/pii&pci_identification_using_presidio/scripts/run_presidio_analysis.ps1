# PowerShell script to run Presidio PCI Analysis

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "    Presidio PCI Data Analysis" -ForegroundColor Blue
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
$configFile = "config/presidio_config.yml"
if (-not (Test-Path $configFile)) {
    Write-Host "[ERROR] Configuration file '$configFile' not found!" -ForegroundColor Red
    exit 1
}

# Check if extracted data exists
$sourceDir = "./data/extracted_data"
if (-not (Test-Path $sourceDir)) {
    Write-Host "[ERROR] Source directory '$sourceDir' not found!" -ForegroundColor Red
    Write-Host "Please run the database extractor first" -ForegroundColor Yellow
    exit 1
}

# Count JSON files
$jsonFiles = Get-ChildItem -Path $sourceDir -Filter "*.json"
if ($jsonFiles.Count -eq 0) {
    Write-Host "[ERROR] No JSON files found in '$sourceDir'" -ForegroundColor Red
    Write-Host "Please run the database extractor first" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Found $($jsonFiles.Count) JSON files to analyze" -ForegroundColor Green

# Check if Presidio is installed in orbit environment
Write-Host "[INFO] Checking Presidio installation in orbit environment..." -ForegroundColor Yellow
try {
    & $pythonCmd -c "import presidio_analyzer, presidio_anonymizer" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Presidio not installed"
    }
    Write-Host "[INFO] Presidio is available in orbit environment" -ForegroundColor Green
}
catch {
    Write-Host "[WARNING] Presidio not installed in orbit environment. Installing..." -ForegroundColor Yellow
    
    # Clean install in orbit environment
    Write-Host "[INFO] Installing Presidio requirements in orbit environment..." -ForegroundColor Yellow
    & pip install -r requirements/presidio_requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install Presidio requirements" -ForegroundColor Red
        exit 1
    }
    
    # Download spaCy model
    Write-Host "[INFO] Downloading spaCy model..." -ForegroundColor Yellow
    & $pythonCmd -m spacy download en_core_web_sm
    
    Write-Host "[SUCCESS] Presidio installation completed in orbit environment" -ForegroundColor Green
}

Write-Host "[INFO] Starting PCI analysis..." -ForegroundColor Yellow
Write-Host "[INFO] Configuration: $configFile" -ForegroundColor Cyan
Write-Host "[INFO] Source directory: $sourceDir" -ForegroundColor Cyan
Write-Host "[INFO] Environment: orbit" -ForegroundColor Cyan
Write-Host "[INFO] Timestamp: $(Get-Date)" -ForegroundColor Cyan
Write-Host ""

# Run the analyzer from src folder
try {
    & $pythonCmd src/pci_analyzer.py -c $configFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "==========================================" -ForegroundColor Blue
        Write-Host "    Analysis Completed Successfully" -ForegroundColor Blue
        Write-Host "==========================================" -ForegroundColor Blue
        Write-Host "Check the data/pci_analysis_results/ directory for:" -ForegroundColor Green
        Write-Host "  - Detailed PII findings (JSON)" -ForegroundColor Green
        Write-Host "  - Compliance report (JSON)" -ForegroundColor Green
        Write-Host "  - Analysis summary (CSV)" -ForegroundColor Green
        Write-Host "  - Dashboard report (HTML)" -ForegroundColor Green
        Write-Host ""
        Write-Host "Check the logs/ directory for detailed logs" -ForegroundColor Green
        
        # Show quick summary of results
        $resultsDir = "data/pci_analysis_results"
        if (Test-Path $resultsDir) {
            $resultFiles = Get-ChildItem -Path $resultsDir
            Write-Host ""
            Write-Host "Generated files:" -ForegroundColor Cyan
            foreach ($file in $resultFiles) {
                Write-Host "  - $($file.Name)" -ForegroundColor Gray
            }
        }
    }
    else {
        throw "Analysis failed"
    }
}
catch {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "    Analysis Failed" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "Check the logs for error details" -ForegroundColor Red
    
    # Try to show recent log file if it exists
    $logDir = "logs"
    if (Test-Path $logDir) {
        $latestLog = Get-ChildItem $logDir -Filter "presidio_analysis_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestLog) {
            Write-Host ""
            Write-Host "Latest log file: $($latestLog.FullName)" -ForegroundColor Yellow
            Write-Host "Last few lines:" -ForegroundColor Yellow
            Get-Content $latestLog.FullName -Tail 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        }
    }
    
    exit 1
}