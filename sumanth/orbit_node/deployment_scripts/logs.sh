#!/bin/bash
# View Logs Script
# Shows logs from Docker services

# Colors
BLUE='\033[0;34m'
NC='\033[0m'

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="${PROJECT_ROOT}/docker"

SERVICE=${1:-pipeline}

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}NJS Logs - $SERVICE${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Press Ctrl+C to exit"
echo ""

cd "${DOCKER_DIR}"

if [ "$SERVICE" = "all" ]; then
    docker compose logs -f
else
    docker compose logs -f "$SERVICE"
fi
