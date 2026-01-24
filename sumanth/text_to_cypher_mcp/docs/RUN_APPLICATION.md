# 🚀 How to Run the Text-to-Cypher Application

This guide will help you run the complete text-to-Cypher conversion system using Docker Compose.

## 📋 Prerequisites

- Docker and Docker Compose installed
- The `mcp/neo4j-cypher:latest` image (which you already have)

## 🏃‍♂️ Quick Start

### Option 1: Use the Startup Script (Windows)

```bash
# Simply run the startup script
start.bat
```

### Option 2: Manual Steps

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Wait for services to be ready (30-60 seconds):**
   ```bash
   # Check if services are running
   docker-compose ps
   ```

3. **Test the application:**
   ```bash
   python example_usage.py
   ```

## 🔧 Service Architecture

The application consists of 3 services:

| Service | Port | Purpose |
|---------|------|---------|
| **neo4j** | 7474, 7687 | Graph database |
| **mcp-server** | 8000 | MCP server for text-to-Cypher conversion |
| **text-to-cypher** | 8081 | REST API wrapper |

## 🌐 Service URLs

- **Neo4j Web UI**: http://localhost:7474 (neo4j/password)
- **MCP Server**: http://localhost:8000/api/mcp/
- **Text-to-Cypher API**: http://localhost:8081
- **API Documentation**: http://localhost:8081 (GET request)

## 🧪 Testing the Application

### 1. Health Check
```bash
curl http://localhost:8081/health
```

### 2. List Available MCP Tools
```bash
curl http://localhost:8081/debug/tools
```

### 3. Convert Text to Cypher
```bash
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Find all users who bought products"}'
```

### 4. Run Comprehensive Tests
```bash
python example_usage.py
```

## 📝 API Usage Examples

### Basic Text-to-Cypher Conversion

**Request:**
```json
POST http://localhost:8081/generate
Content-Type: application/json

{
  "text": "Show me all movies from 2023"
}
```

**Response:**
```json
{
  "input_text": "Show me all movies from 2023",
  "cypher_query": "MATCH (m:Movie) WHERE m.year = 2023 RETURN m",
  "status": "success"
}
```

### Python Example
```python
import requests

response = requests.post(
    "http://localhost:8081/generate",
    json={"text": "Find users who bought expensive products"}
)

result = response.json()
print(f"Generated Cypher: {result['cypher_query']}")
```

## 🔍 Troubleshooting

### Services Won't Start
```bash
# Check service logs
docker-compose logs neo4j
docker-compose logs mcp-server
docker-compose logs text-to-cypher

# Restart services
docker-compose restart
```

### MCP Server Connection Issues
```bash
# Test MCP server directly
curl -X POST http://localhost:8000/api/mcp/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }'
```

### Neo4j Connection Issues
1. Open http://localhost:7474
2. Login with: `neo4j` / `password`
3. Run a test query: `RETURN 1`

### Application Returns Errors
1. Check if all services are healthy:
   ```bash
   docker-compose ps
   ```

2. Check available MCP tools:
   ```bash
   curl http://localhost:8081/debug/tools
   ```

3. View application logs:
   ```bash
   docker-compose logs text-to-cypher
   ```

## 🛑 Stopping the Application

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears Neo4j data)
docker-compose down -v
```

## 🔧 Configuration

The application uses these key configurations:

- **MCP Server URL**: `http://mcp-server:8000/api/mcp/`
- **Neo4j Connection**: `bolt://neo4j:7687` (neo4j/password)
- **API Port**: 8081

You can modify these in the `docker-compose.yml` file if needed.

## 📊 Expected Behavior

1. **Startup**: Services start in order (Neo4j → MCP Server → API)
2. **Health Checks**: All services should report healthy status
3. **Tool Discovery**: MCP server should expose text-to-Cypher tools
4. **Conversion**: English text gets converted to valid Cypher queries

If everything is working correctly, you should see successful Cypher query generation when running the test script!