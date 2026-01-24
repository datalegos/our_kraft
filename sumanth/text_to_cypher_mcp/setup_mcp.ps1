#!/usr/bin/env pwsh
# Setup script for Text-to-Cypher MCP Server

Write-Host "🚀 Setting up Text-to-Cypher MCP Server for Cursor" -ForegroundColor Cyan

# Check if Node.js is installed
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
    
    # Check if version is 18+
    $versionNumber = [int]($nodeVersion -replace 'v(\d+)\..*', '$1')
    if ($versionNumber -lt 18) {
        Write-Host "❌ Node.js 18+ required. Current version: $nodeVersion" -ForegroundColor Red
        Write-Host "Please upgrade Node.js to version 18 or higher" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+ first" -ForegroundColor Red
    Write-Host "Download from: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Install dependencies
Write-Host "📦 Installing MCP SDK dependencies..." -ForegroundColor Yellow
try {
    npm install
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Test the MCP server
Write-Host "🧪 Testing MCP server..." -ForegroundColor Yellow
try {
    # Check if the application is running
    $response = Invoke-RestMethod -Uri "http://localhost:8081/health" -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Host "✅ Text-to-Cypher application is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Text-to-Cypher application is not healthy" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Text-to-Cypher application is not running" -ForegroundColor Red
    Write-Host "Please start the application with: docker-compose up -d" -ForegroundColor Yellow
}

Write-Host "`n🎯 Setup Complete!" -ForegroundColor Cyan
Write-Host "📋 Next Steps:" -ForegroundColor White
Write-Host "1. Make sure your text-to-cypher application is running: docker-compose up -d" -ForegroundColor Gray
Write-Host "2. The MCP configuration is already created in .kiro/settings/mcp.json" -ForegroundColor Gray
Write-Host "3. Restart Cursor to load the new MCP server" -ForegroundColor Gray
Write-Host "4. You can now use these tools in Cursor:" -ForegroundColor Gray
Write-Host "   • generate_cypher - Convert text to Cypher queries" -ForegroundColor Green
Write-Host "   • execute_cypher - Execute Cypher queries" -ForegroundColor Green
Write-Host "   • get_database_schema - Get Neo4j schema" -ForegroundColor Green
Write-Host "   • health_check - Check application status" -ForegroundColor Green
Write-Host "   • get_config - View application configuration" -ForegroundColor Green

Write-Host "`n💡 Example usage in Cursor:" -ForegroundColor Cyan
Write-Host "Ask: 'Use the text-to-cypher MCP to find all users in the database'" -ForegroundColor White