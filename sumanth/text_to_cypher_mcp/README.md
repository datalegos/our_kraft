# Cypher Pipeline System

A configurable microservices system that converts plain English text to Cypher queries and analyzes character counts.

## 🏗️ Architecture

- **Neo4j Database** - Graph database for schema reference
- **MCP Wrapper** - Converts English text to Cypher queries
- **Character Counter** - Analyzes text character metrics
- **Client** - Python client with retry logic and health checks

## ⚙️ Configuration System

All hardcoded values have been moved to JSON configuration files in the `config/` directory:

### Configuration Files

| File | Purpose | Environment Variable |
|------|---------|---------------------|
| `config/mcp-wrapper-config.json` | MCP wrapper service settings | `CONFIG_PATH` |
| `config/char-counter-config.json` | Character counter settings | `CONFIG_PATH` |
| `config/client-config.json` | Client configuration | `CLIENT_CONFIG_PATH` |

### MCP Wrapper Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8081,
    "debug": true
  },
  "services": {
    "cypher_generator": {
      "url": "http://cypher-generator:8000/mcp/",
      "timeout": 10
    },
    "character_counter": {
      "url": "http://char-counter:5000/count",
      "timeout": 10
    }
  },
  "cypher_generation": {
    "default_limit": 10,
    "max_limit": 100,
    "patterns": {
      "find_keywords": ["find", "get", "show"],
      "create_keywords": ["create", "add", "insert"],
      // ... more patterns
    },
    "node_types": {
      "person": ["user", "person", "people"],
      "product": ["product", "item", "goods"],
      // ... more node types
    }
  }
}
```

### Character Counter Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  },
  "analysis": {
    "include_spaces": true,
    "include_words": true,
    "include_lines": true,
    "custom_separators": {
      "word_separators": [" ", "\t", "\n"],
      "sentence_separators": [".", "!", "?"]
    }
  },
  "limits": {
    "max_text_length": 10000
  }
}
```

### Client Configuration

```json
{
  "services": {
    "mcp_wrapper": {
      "base_url": "http://localhost:8081",
      "timeout": 30,
      "retry_attempts": 3
    },
    "character_counter": {
      "base_url": "http://localhost:5000",
      "timeout": 10,
      "retry_attempts": 2
    }
  },
  "logging": {
    "level": "INFO",
    "enable_debug": false
  }
}
```

## 🚀 Quick Start

### 1. Start the System
```bash
docker-compose up --build -d
```

### 2. Test with Python Client
```bash
python client.py
```

### 3. Test with Postman
- **Complete Pipeline**: `POST http://localhost:8081/process`
- **Generate Cypher**: `POST http://localhost:8081/generate`
- **Count Characters**: `POST http://localhost:5000/count`

## 🔧 Customization

### Modify Cypher Generation Patterns

Edit `config/mcp-wrapper-config.json`:

```json
{
  "cypher_generation": {
    "patterns": {
      "find_keywords": ["find", "search", "locate", "discover"],
      "create_keywords": ["create", "make", "build", "generate"]
    },
    "node_types": {
      "customer": ["customer", "client", "buyer"],
      "invoice": ["invoice", "bill", "receipt"]
    }
  }
}
```

### Adjust Character Analysis

Edit `config/char-counter-config.json`:

```json
{
  "analysis": {
    "include_sentences": true,
    "include_paragraphs": true,
    "custom_separators": {
      "sentence_separators": [".", "!", "?", ";"]
    }
  }
}
```

### Configure Client Behavior

Edit `config/client-config.json`:

```json
{
  "services": {
    "mcp_wrapper": {
      "timeout": 60,
      "retry_attempts": 5,
      "retry_delay": 2
    }
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

## 📡 API Endpoints

### MCP Wrapper Service (Port 8081)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process` | POST | Complete pipeline |
| `/generate` | POST | Generate Cypher only |
| `/health` | GET | Health check |
| `/config` | GET | View configuration |

### Character Counter Service (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/count` | POST | Analyze text |
| `/health` | GET | Health check |
| `/config` | GET | View configuration |

## 🔍 Configuration Validation

The system validates configuration on startup and provides detailed error messages for:
- Missing configuration files
- Invalid JSON syntax
- Missing required fields
- Invalid data types

## 📊 Example Requests

### Complete Pipeline
```bash
curl -X POST http://localhost:8081/process \
  -H "Content-Type: application/json" \
  -d '{"question": "Find all users who bought products"}'
```

### Response
```json
{
  "question": "Find all users who bought products",
  "cypher_query": "// Generated from: Find all users who bought products\nMATCH (p:Person) RETURN p LIMIT 10",
  "character_analysis": {
    "total_characters": 67,
    "words": 11,
    "lines": 2
  },
  "metadata": {
    "node_type": "person",
    "operation": "find"
  }
}
```

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `/app/config/mcp-wrapper-config.json` | MCP wrapper config path |
| `CLIENT_CONFIG_PATH` | `config/client-config.json` | Client config path |
| `LOG_LEVEL` | `INFO` | Logging level |

## 🔄 Configuration Hot Reload

To apply configuration changes:
1. Edit the JSON files in `config/`
2. Restart the services: `docker-compose restart`

## 📝 Logging

All services support configurable logging:
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Format**: Timestamp, service name, level, message
- **Output**: Console (captured by Docker)

View logs:
```bash
docker-compose logs -f mcp-wrapper
docker-compose logs -f char-counter
```

## 🎯 Benefits of Configuration-Driven Design

✅ **No Hardcoded Values** - All settings in JSON files  
✅ **Easy Customization** - Modify behavior without code changes  
✅ **Environment Specific** - Different configs for dev/prod  
✅ **Validation** - Startup validation with clear error messages  
✅ **Documentation** - Self-documenting configuration structure  
✅ **Extensibility** - Easy to add new patterns and behaviors  

Your system is now fully configurable and production-ready! 🎉