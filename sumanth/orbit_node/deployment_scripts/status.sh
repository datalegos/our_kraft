#!/bin/bash
# Status Check Script
# Shows status of all services and pipeline

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="${PROJECT_ROOT}/docker"
SHARED_DATA_DIR="${PROJECT_ROOT}/../orbit_node_shared_data"

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}NJS Status${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Docker services status
echo -e "${BLUE}Docker Services:${NC}"
cd "${DOCKER_DIR}"
docker compose ps
echo ""

# Pipeline status
echo -e "${BLUE}Pipeline Status:${NC}"
if [ -f "${SHARED_DATA_DIR}/pipeline/.done" ]; then
    echo -e "${GREEN}✓${NC} Pipeline completed"
    cat "${SHARED_DATA_DIR}/pipeline/.done"
else
    echo -e "${RED}✗${NC} Pipeline not completed"
fi
echo ""

# Step completion status
echo -e "${BLUE}Step Completion:${NC}"
steps=("collected_data" "extracted_data" "aggregated_data_core" "pii_scan_results")

for step in "${steps[@]}"; do
    if [ -f "${SHARED_DATA_DIR}/${step}/.done" ]; then
        echo -e "  ${GREEN}✓${NC} $step"
    else
        echo -e "  ${RED}✗${NC} $step"
    fi
done
echo ""

# Disk usage
echo -e "${BLUE}Disk Usage:${NC}"
du -sh "${SHARED_DATA_DIR}" 2>/dev/null || echo "Shared data directory not found"
echo ""
