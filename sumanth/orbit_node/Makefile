# NJS Pipeline Makefile
# Convenience commands for Docker operations

.PHONY: help setup install build start stop restart logs status backup clean

# Default target
help:
	@echo "NJS Pipeline - Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initial setup (copy .env.example, create dirs)"
	@echo "  make install        - Install Poetry dependencies locally"
	@echo "  make build          - Build Docker images"
	@echo ""
	@echo "Operations:"
	@echo "  make start          - Start all services"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View pipeline logs"
	@echo "  make logs-neo4j     - View Neo4j logs"
	@echo ""
	@echo "Pipeline:"
	@echo "  make run            - Run complete pipeline"
	@echo "  make shell          - Access pipeline container shell"
	@echo ""
	@echo "Maintenance:"
	@echo "  make status         - Show service status"
	@echo "  make backup         - Backup Neo4j databases"
	@echo "  make clean          - Remove old data (7+ days)"
	@echo "  make reset          - Stop and remove all containers/volumes"

# Setup
setup:
	@echo "Setting up NJS Pipeline..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file - please edit with your configuration"; \
	else \
		echo "⚠️  .env already exists"; \
	fi
	@mkdir -p ../njs_shared_data/{data/{collected,extracted,aggregated,aggregated_core,pii_scan_results},logs,pipeline,config}
	@echo "✅ Created shared data directories"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env with your configuration"
	@echo "  2. Run: make build"
	@echo "  3. Run: make start"

# Install Poetry dependencies locally
install:
	@echo "Installing Poetry dependencies..."
	@if ! command -v poetry &> /dev/null; then \
		echo "Poetry not found. Installing..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
	fi
	@poetry install
	@poetry run python -m spacy download en_core_web_lg
	@echo "✅ Dependencies installed"

# Build
build:
	@echo "Building Docker images..."
	cd docker && docker compose build --no-cache

# Start services
start:
	@echo "Starting services..."
	cd docker && docker compose up -d
	@echo "✅ Services started"
	@echo "View logs: make logs"

# Stop services
stop:
	@echo "Stopping services..."
	cd docker && docker compose down
	@echo "✅ Services stopped"

# Restart services
restart: stop start

# View logs
logs:
	cd docker && docker compose logs -f pipeline

logs-neo4j:
	cd docker && docker compose logs -f neo4j

# Run pipeline
run:
	@echo "Running complete pipeline..."
	cd docker && docker compose exec pipeline poetry run python /app/scripts/orchestrator.py

# Shell access
shell:
	cd docker && docker compose exec pipeline /bin/bash

# Status
status:
	@echo "Service Status:"
	cd docker && docker compose ps
	@echo ""
	@echo "Pipeline Status:"
	@if [ -f ../njs_shared_data/pipeline/.done ]; then \
		echo "✅ Pipeline completed"; \
		cat ../njs_shared_data/pipeline/.done; \
	else \
		echo "⏳ Pipeline not completed or running"; \
	fi

# Backup Neo4j
backup:
	@echo "Backing up Neo4j databases..."
	@mkdir -p ./backups
	cd docker && docker compose exec neo4j neo4j-admin database dump node_kg --to=/var/lib/neo4j/import/node_kg_backup.dump || true
	cd docker && docker compose exec neo4j neo4j-admin database dump core --to=/var/lib/neo4j/import/core_backup.dump || true
	docker cp orbit-neo4j:/var/lib/neo4j/import/node_kg_backup.dump ./backups/ || true
	docker cp orbit-neo4j:/var/lib/neo4j/import/core_backup.dump ./backups/ || true
	@echo "✅ Backups saved to ./backups/"

# Clean old data
clean:
	@echo "Cleaning old data (7+ days)..."
	@find ../njs_shared_data/data/collected/ -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
	@find ../njs_shared_data/data/pii_scan_results/ -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Reset everything
reset:
	@echo "⚠️  WARNING: This will remove all containers and volumes!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read confirm
	cd docker && docker compose down -v
	@echo "✅ Reset complete"
