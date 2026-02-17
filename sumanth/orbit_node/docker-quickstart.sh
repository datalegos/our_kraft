#!/bin/bash
# DataLegos Docker Quick Start Script
# Automates initial setup and deployment

set -e

echo "=================================================================="
echo "DataLegos Pipeline - Docker Quick Start"
echo "=================================================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo "✅ Docker Compose found: $(docker-compose --version)"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env with your configuration:"
    echo "   - WAZUH_API_URL"
    echo "   - WAZUH_API_USERNAME"
    echo "   - WAZUH_API_PASSWORD"
    echo "   - NEO4J_PASSWORD"
    echo "   - BANK_ID"
    echo ""
    read -p "Press Enter after editing .env, or Ctrl+C to exit..."
else
    echo "✅ .env file already exists"
fi

# Create shared data directory
echo ""
echo "Creating shared data directory..."
mkdir -p ../orbit_node_shared_data/{collected_data,extracted_data,aggregated_data_core,pii_scan_results,logs,pipeline}
echo "✅ Created: ../orbit_node_shared_data/"
echo ""

# Build Docker images
echo "Building Docker images (this may take a few minutes)..."
cd docker
docker-compose build
echo "✅ Docker images built"
echo ""

# Start services
echo "Starting services..."
docker-compose up -d
echo "✅ Services started"
echo ""

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
sleep 10

# Check service status
echo ""
echo "Service Status:"
docker-compose ps
echo ""

# Show next steps
echo "=================================================================="
echo "✅ Setup Complete!"
echo "=================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. View logs:"
echo "   docker-compose logs -f pipeline"
echo ""
echo "2. Run pipeline:"
echo "   docker-compose exec pipeline python /app/scripts/orchestrator.py"
echo ""
echo "3. Or use Makefile commands:"
echo "   make logs    # View logs"
echo "   make run     # Run pipeline"
echo "   make status  # Check status"
echo ""
echo "4. Access Neo4j browser:"
echo "   http://localhost:7474"
echo "   Username: neo4j"
echo "   Password: (from your .env file)"
echo ""
echo "For more information, see README_DOCKER.md"
echo "=================================================================="
