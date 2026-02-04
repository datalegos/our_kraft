# Setup Orbit Conda Environment for PCI Analysis

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "    Setting up Orbit Environment" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue

# Check if conda is available
try {
    & conda --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Conda not found"
    }
    Write-Host "[INFO] Conda is available" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Conda is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Create orbit environment if it doesn't exist
Write-Host "[INFO] Checking for orbit environment..." -ForegroundColor Yellow
$envExists = & conda env list | Select-String "orbit"
if (-not $envExists) {
    Write-Host "[INFO] Creating orbit environment..." -ForegroundColor Yellow
    & conda create -n orbit python=3.9 -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create orbit environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] Created orbit environment" -ForegroundColor Green
} else {
    Write-Host "[INFO] Orbit environment already exists" -ForegroundColor Green
}

# Activate orbit environment
Write-Host "[INFO] Activating orbit environment..." -ForegroundColor Yellow
try {
    & conda activate orbit
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate orbit environment"
    }
    Write-Host "[SUCCESS] Activated orbit environment" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Failed to activate orbit environment" -ForegroundColor Red
    exit 1
}

# Install basic requirements in orbit environment
Write-Host "[INFO] Installing basic requirements in orbit environment..." -ForegroundColor Yellow
& pip install -r requirements/database_requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Database requirements installed in orbit environment" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Some database requirements failed to install" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Blue
Write-Host "    Orbit Environment Setup Complete" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue
Write-Host "You can now run:" -ForegroundColor Green
Write-Host "  .\scripts\run_database_extractor.ps1" -ForegroundColor Cyan
Write-Host "  .\scripts\run_presidio_analysis.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Presidio will be installed automatically when needed" -ForegroundColor Yellow