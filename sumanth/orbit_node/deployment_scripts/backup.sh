#!/bin/bash
# Backup Script
# Backs up Neo4j databases

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
BACKUP_DIR="${PROJECT_ROOT}/backups"

DATE=$(date +%Y%m%d_%H%M%S)

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}NJS Backup${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

print_info "Creating backup: $DATE"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

cd "${DOCKER_DIR}"

# Backup node_kg
print_info "Backing up node_kg database..."
docker compose exec -T neo4j neo4j-admin database dump node_kg \
    --to=/var/lib/neo4j/import/node_kg_${DATE}.dump || true

# Backup core
print_info "Backing up core database..."
docker compose exec -T neo4j neo4j-admin database dump core \
    --to=/var/lib/neo4j/import/core_${DATE}.dump || true

# Copy to host
print_info "Copying backups to host..."
docker cp orbit-neo4j:/var/lib/neo4j/import/node_kg_${DATE}.dump "${BACKUP_DIR}/" || true
docker cp orbit-neo4j:/var/lib/neo4j/import/core_${DATE}.dump "${BACKUP_DIR}/" || true

echo ""
print_info "Backup complete!"
echo ""
echo "Backup files:"
ls -lh "${BACKUP_DIR}/"*${DATE}*
echo ""
echo "Backup location: ${BACKUP_DIR}/"
echo ""
