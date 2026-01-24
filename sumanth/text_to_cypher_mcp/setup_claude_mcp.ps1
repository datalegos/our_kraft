#!/usr/bin/env pwsh
# Setup Text-to-Cypher MCP for Claude Desktop

Write-Host "🎯 Setting up Text-to-Cypher MCP for Claude Desktop" -ForegroundColor Cyan

# Get current directory
$currentPath = Get-Location
$mcpServerPath = Join-Path $currentPath "text-to-cypher-mcp-server.js"

# Check if MCP server file exists
if (-not (Test-Path $mcpServerPath)) {
    Write-Host "❌ MCP server file not found: $mcpServerPath" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
npm install

# Find Claude config directory
$claudeConfigDir = "$env:APPDATA\Claude"
$claudeConfigFile = Join-Path $claudeConfigDir "claude_desktop_config.json"

# Create directory if it doesn't exist
if (-not (Test-Path $claudeConfigDir)) {
    New-Item -ItemType Directory -Path $claudeConfigDir -Force
    Write-Host "📁 Created Claude config directory: $claudeConfigDir" -ForegroundColor Green
}

# Create MCP configuration
$mcpConfig = @{
    mcpServers = @{
        "text-to-cypher" = @{
            command = "node"
            args = @($mcpServerPath.Replace('\', '/'))
            env = @{}
        }
    }
} | ConvertTo-Json -Depth 10

# Write configuration
$mcpConfig | Out-File -FilePath $claudeConfigFile -Encoding UTF8
Write-Host "✅ Claude MCP configuration created: $claudeConfigFile" -ForegroundColor Green

# Test application
Write-Host "🧪 Testing text-to-cypher application..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8081/health" -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Host "✅ Text-to-cypher application is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Text-to-cypher application is not healthy" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Text-to-cypher application is not running" -ForegroundColor Red
    Write-Host "Please start it with: docker-compose up -d" -ForegroundColor Yellow
}

Write-Host "`n🎉 Setup Complete!" -ForegroundColor Cyan
Write-Host "📋 Next Steps:" -ForegroundColor White
Write-Host "1. Restart Claude Desktop completely" -ForegroundColor Gray
Write-Host "2. Start a new conversation" -ForegroundColor Gray
Write-Host "3. Ask Claude to use the text-to-cypher MCP server" -ForegroundColor Gray

Write-Host "`n💬 Example prompts to try:" -ForegroundColor Cyan
Write-Host "• 'Use the text-to-cypher MCP to check if it's working'" -ForegroundColor White
Write-Host "• 'Get the database schema using the MCP server'" -ForegroundColor White
Write-Host "• 'Find all users in the database using text-to-cypher'" -ForegroundColor White

Write-Host "`n📍 Configuration file location:" -ForegroundColor Yellow
Write-Host $claudeConfigFile -ForegroundColor White