# Text-to-Cypher Converter

A simple Flask service that converts English text to Cypher queries using existing MCP servers.

## Features

- ✅ Clean, minimal implementation
- ✅ Uses existing MCP servers (no custom server creation needed)
- ✅ No hardcoded fallbacks or complex logic
- ✅ Environment variable configuration
- ✅ Simple REST API

## Quick Start

### 1. Set Environment Variables

```bash
# Required: MCP Server URL
export MCP_SERVER_URL="http://your-mcp-server:8080/mcp/"

# Optional: Configuration
export MCP_TIMEOUT="30"
export PORT="8081"
export HOST="0.0.0.0"
export DEBUG="false"
```

### 2. Install Dependencies

```bash
cd mcp-wrapper
pip install -r requirements.txt
```

### 3. Run the Service

```bash
python app.py
```

## API Usage

### Convert Text to Cypher

**POST** `/generate`

```json
{
  "text": "Find all users who bought products"
}
```

**Response:**
```json
{
  "input_text": "Find all users who bought products",
  "cypher_query": "MATCH (u:User)-[:PURCHASED]->(p:Product) RETURN u, p",
  "status": "success"
}
```

### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "service": "text-to-cypher"
}
```

## Example Usage

```python
import requests

# Convert English to Cypher
response = requests.post(
    "http://localhost:8081/generate",
    json={"text": "Show me all movies from 2023"}
)

result = response.json()
print(f"Cypher: {result['cypher_query']}")
```

## MCP Server Requirements

Your MCP server should provide a tool named `generate_cypher` that accepts:
- `query`: The English text to convert

Example MCP tool call:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "generate_cypher",
    "arguments": {
      "query": "Find all users"
    }
  }
}
```

## Configuration

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_URL` | `http://localhost:8080/mcp/` | MCP server endpoint |
| `MCP_TIMEOUT` | `30` | Request timeout in seconds |
| `PORT` | `8081` | Service port |
| `HOST` | `0.0.0.0` | Service host |
| `DEBUG` | `false` | Enable debug mode |

## Testing

Run the example script to test the service:

```bash
python example_usage.py
```

This will test various English queries and show the generated Cypher queries.