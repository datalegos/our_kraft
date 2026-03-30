#!/bin/bash
# Start Services Script
# Starts existing Docker services

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="${PROJECT_ROOT}/docker"

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Starting NJS Services${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

print_info "Starting services..."

cd "${DOCKER_DIR}"
docker compose up -d

echo ""
print_info "Services started"
echo ""

# Show status
docker compose ps

echo ""
echo -e "${BLUE}Access Points:${NC}"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - View logs: ./logs.sh"
echo ""
