# DataLegos Pipeline Makefile
# Convenience commands for Docker operations

.PHONY: help setup build up down logs shell clean backup

# Default target
help:
	@echo "DataLegos Pipeline - Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initial setup (copy .env.example, create dirs)"
	@echo "  make build          - Build Docker images"
	@echo ""
	@echo "Operations:"
	@echo "  make up             - Start all services"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View pipeline logs"
	@echo "  make logs-neo4j     - View Neo4j logs"
	@echo ""
	@echo "Pipeline:"
	@echo "  make run            - Run complete pipeline"
	@echo "  make run-collect    - Run collect_data step"
	@echo "  make run-extract    - Run extract_data step"
	@echo "  make run-node       - Run build_node_graph step"
	@echo "  make run-aggregate  - Run aggregate_data step"
	@echo "  make run-pii        - Run detect_pii step"
	@echo "  make run-core       - Run build_core_graph step"
	@echo ""
	@echo "Access:"
	@echo "  make shell          - Access pipeline container shell"
	@echo "  make shell-neo4j    - Access Neo4j container shell"
	@echo "  make neo4j          - Open Neo4j browser"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove old data (7+ days)"
	@echo "  make backup         - Backup Neo4j databases"
	@echo "  make reset          - Stop and remove all containers/volumes"

# Setup
setup:
	@echo "Setting up DataLegos Pipeline..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file - please edit with your configuration"; \
	else \
		echo "⚠️  .env already exists"; \
	fi
	@mkdir -p ../orbit_node_shared_data/{collected_data,extracted_data,aggregated_data_core,pii_scan_results,logs,pipeline}
	@echo "✅ Created shared data directories"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env with your configuration"
	@echo "  2. Run: make build"
	@echo "  3. Run: make up"

# Build
build:
	@echo "Building Docker images..."
	cd docker && docker-compose build --no-cache

# Start services
up:
	@echo "Starting services..."
	cd docker && docker-compose up -d
	@echo "✅ Services started"
	@echo "View logs: make logs"

# Stop services
down:
	@echo "Stopping services..."
	cd docker && docker-compose down
	@echo "✅ Services stopped"

# Restart services
restart: down up

# View logs
logs:
	cd docker && docker-compose logs -f pipeline

logs-neo4j:
	cd docker && docker-compose logs -f neo4j

# Run pipeline
run:
	@echo "Running complete pipeline..."
	cd docker && docker-compose exec pipeline python /app/scripts/orchestrator.py

# Run individual steps
run-collect:
	@echo "Running collect_data step..."
	cd docker && docker-compose exec pipeline python /app/scripts/orchestrator.py --step collect_data

run-extract:
	@echo "Running extract_data step..."
	cd docker && docker-compose exec pipeline python /app/scripts/orchestrator.py --step extract_data

run-node:
	@echo "Running build_node_graph step..."
	cd docker && docker-compose exec pipeline python /app/scripts/orchestrator.py --step build_node_graph

run-aggregate:
	@echo "Running aggregate_data step..."
	cd docker && docker-compose exec pipeline python /app/scripts/orchestrator.py --step aggregate_data

run-pii:
	@echo "Running detect_pii step..."
	cd docker && docker-compose exec pipeline python /app/scripts/orchestrator.py --step detect_pii

run-core:
	@echo "Running build_core_graph step..."
	cd docker && docker-compose exec pipeline python /app/scripts/orchestrator.py --step build_core_graph

# Shell access
shell:
	cd docker && docker-compose exec pipeline /bin/bash

shell-neo4j:
	cd docker && docker-compose exec neo4j /bin/bash

# Neo4j browser
neo4j:
	@echo "Opening Neo4j browser..."
	@echo "URL: http://localhost:7474"
	@echo "Username: neo4j"
	@echo "Password: (from your .env file)"
	@command -v open >/dev/null 2>&1 && open http://localhost:7474 || \
	command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:7474 || \
	echo "Please open http://localhost:7474 in your browser"

# Clean old data
clean:
	@echo "Cleaning old data (7+ days)..."
	@find ../orbit_node_shared_data/collected_data/ -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
	@find ../orbit_node_shared_data/pii_scan_results/ -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Backup Neo4j
backup:
	@echo "Backing up Neo4j databases..."
	@mkdir -p ./backups
	cd docker && docker-compose exec neo4j neo4j-admin database dump node_kg --to=/var/lib/neo4j/import/node_kg_backup.dump || true
	cd docker && docker-compose exec neo4j neo4j-admin database dump core --to=/var/lib/neo4j/import/core_backup.dump || true
	docker cp datalegos_neo4j:/var/lib/neo4j/import/node_kg_backup.dump ./backups/ || true
	docker cp datalegos_neo4j:/var/lib/neo4j/import/core_backup.dump ./backups/ || true
	@echo "✅ Backups saved to ./backups/"

# Reset everything
reset:
	@echo "⚠️  WARNING: This will remove all containers and volumes!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read confirm
	cd docker && docker-compose down -v
	@echo "✅ Reset complete"

# Status
status:
	@echo "Service Status:"
	cd docker && docker-compose ps
	@echo ""
	@echo "Pipeline Status:"
	@if [ -f ../orbit_node_shared_data/pipeline/.done ]; then \
		echo "✅ Pipeline completed"; \
		cat ../orbit_node_shared_data/pipeline/.done; \
	else \
		echo "⏳ Pipeline not completed or running"; \
	fi
