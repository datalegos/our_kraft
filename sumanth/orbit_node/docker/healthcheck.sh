#!/bin/bash

# DataLegos Pipeline Health Check
# Returns 0 if healthy, 1 if unhealthy

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

# Check if shared data path is mounted
if [ ! -d "${SHARED_DATA_PATH:-/shared_data}" ]; then
    echo "ERROR: Shared data path not mounted"
    exit 1
fi

# Check if .env file exists
if [ ! -f /app/.env ]; then
    echo "ERROR: .env file not found"
    exit 1
fi

# Check if orchestrator script exists
if [ ! -f /app/scripts/orchestrator.py ]; then
    echo "ERROR: orchestrator.py not found"
    exit 1
fi

# If pipeline is running, check its status
if pgrep -f "orchestrator.py" > /dev/null; then
    echo "Pipeline is running"
    exit 0
fi

# If pipeline completed, check for success marker
if [ -f "${SHARED_DATA_PATH:-/shared_data}/pipeline/.done" ]; then
    echo "Pipeline completed successfully"
    exit 0
fi

# If we're in run-once mode and nothing is running, that's OK
if [ "${PIPELINE_MODE:-run-once}" = "run-once" ]; then
    echo "Pipeline in run-once mode (idle)"
    exit 0
fi

# Default: healthy
echo "Container healthy"
exit 0
