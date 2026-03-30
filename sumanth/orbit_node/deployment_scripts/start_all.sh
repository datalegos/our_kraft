#!/bin/bash
# Complete Setup and Start Script
# Creates all Docker setup and starts all services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="${PROJECT_ROOT}/docker"
SHARED_DATA_DIR="${PROJECT_ROOT}/../njs_shared_data"

print_header "NJS - Complete Setup and Start"

# Step 1: Check prerequisites
print_info "Step 1: Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi

print_success "Prerequisites OK"

# Step 2: Create .env if not exists
print_info "Step 2: Checking environment configuration..."

if [ ! -f "${PROJECT_ROOT}/.env" ]; then
    print_warning ".env file not found"
    print_info "Creating .env from template..."
    cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    print_success ".env file created"
    echo ""
    print_warning "IMPORTANT: Edit .env file with your configuration before continuing!"
    echo ""
    echo "Required settings:"
    echo "  - WAZUH_API_URL"
    echo "  - WAZUH_API_USERNAME"
    echo "  - WAZUH_API_PASSWORD"
    echo "  - NEO4J_PASSWORD"
    echo "  - BANK_ID"
    echo ""
    read -p "Press Enter after editing .env file..."
else
    print_success ".env file exists"
fi

# Step 3: Create shared data directories
print_info "Step 3: Creating shared data directories..."

mkdir -p "${SHARED_DATA_DIR}"/{data/{collected,extracted,aggregated,aggregated_core,pii_scan_results},logs,pipeline,config}
print_success "Shared data directories created"

# Step 4: Create backup directory
print_info "Step 4: Creating backup directory..."
mkdir -p "${PROJECT_ROOT}/backups"
print_success "Backup directory created"

# Step 5: Build Docker images
print_info "Step 5: Building Docker images..."
print_warning "This may take several minutes..."

cd "${DOCKER_DIR}"
docker compose build --no-cache

print_success "Docker images built"

# Step 6: Start services
print_info "Step 6: Starting services..."

docker compose up -d

print_success "Services started"

# Step 7: Wait for services to be ready
print_info "Step 7: Waiting for services to be ready..."

echo -n "Waiting for Neo4j"
for i in {1..30}; do
    if docker compose exec -T neo4j cypher-shell -u neo4j -p "$(grep NEO4J_PASSWORD ${PROJECT_ROOT}/.env | cut -d '=' -f2)" "RETURN 1" &> /dev/null; then
        echo ""
        print_success "Neo4j is ready"
        break
    fi
    echo -n "."
    sleep 2
done

# Step 8: Show status
print_info "Step 8: Checking service status..."
echo ""
docker compose ps

# Final summary
print_header "Setup Complete!"

echo -e "${GREEN}✓${NC} Docker images built"
echo -e "${GREEN}✓${NC} Services started"
echo -e "${GREEN}✓${NC} Neo4j ready"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. View logs: cd deployment_scripts && ./logs.sh"
echo "  2. Access Neo4j: http://localhost:7474"
echo "  3. Check status: ./status.sh"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "  - Start services: ./start.sh"
echo "  - Stop services: ./stop.sh"
echo "  - View logs: ./logs.sh"
echo "  - Check status: ./status.sh"
echo "  - Backup: ./backup.sh"
echo ""
