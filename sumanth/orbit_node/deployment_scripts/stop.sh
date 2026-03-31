#!/bin/bash
# Stop Services Script
# Stops running Docker services without removing containers

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="${PROJECT_ROOT}/docker"

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Stopping NJS Services${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

print_info "Stopping services..."

cd "${DOCKER_DIR}"
docker compose stop

echo ""
print_info "Services stopped"
echo ""

print_warning "Containers are stopped but not removed"
print_info "To start again: ./start.sh"
echo ""
