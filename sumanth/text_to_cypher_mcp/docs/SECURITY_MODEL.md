# 🔒 Secure Text-to-Cypher Architecture

## Security-First Design

You asked the right question! Instead of giving LLMs direct database access, we implement a **secure isolation model**.

## 🏗️ Architecture Overview

```
[User Question] 
    ↓
[Our App] ← Gets schema via MCP (with our credentials)
    ↓
[OpenAI/LLM] ← Receives: Question + Schema (NO DB ACCESS)
    ↓
[Generated Cypher] 
    ↓
[Security Validation] ← Blocks dangerous operations
    ↓
[MCP Server] ← Executes with our credentials
    ↓
[Results] → [User]
```

## 🛡️ Security Layers

### Layer 1: LLM Isolation
- ✅ **LLM has ZERO database access**
- ✅ **Only receives**: Question text + Schema info
- ✅ **Only returns**: Cypher query string
- ❌ **Cannot**: Execute queries, access data, modify DB

### Layer 2: Query Validation
```python
# Blocked operations
BLOCKED = ['DELETE', 'CREATE', 'MERGE', 'SET', 'REMOVE', 'DROP']

# Allowed operations  
ALLOWED = ['MATCH', 'RETURN', 'WHERE', 'ORDER BY', 'LIMIT', 'COUNT']

# Additional checks
- Max query length: 1000 chars
- No multiple statements (;)
- No admin commands (SHOW USERS, CALL dbms)
- Must contain read-only operations
```

### Layer 3: MCP Execution
- ✅ **Our credentials** connect to database
- ✅ **Our MCP server** executes validated queries
- ✅ **Controlled data access** via read-only tools
- ✅ **Audit trail** of all executed queries

## 🔐 What This Prevents

### ❌ Direct LLM Database Access
```
BAD: [LLM] → [Database] (Direct connection)
GOOD: [LLM] → [Our App] → [MCP] → [Database]
```

### ❌ Malicious Query Injection
```python
# This gets blocked:
"DELETE ALL NODES; DROP DATABASE;"

# This gets allowed:
"MATCH (u:User) RETURN u.name LIMIT 10"
```

### ❌ Data Exfiltration
- LLM never sees actual data
- Only sees schema structure
- Cannot execute queries directly
- All results go through our app

### ❌ Privilege Escalation
- No admin commands allowed
- No schema modifications
- No user management
- Read-only operations only

## 🎯 Benefits

### For Security
1. **Zero Trust**: LLM has no database privileges
2. **Validation**: Every query security-checked
3. **Audit**: All operations logged
4. **Control**: We own the database connection

### For Functionality
1. **AI-Powered**: Real natural language understanding
2. **Schema-Aware**: Uses your actual database structure
3. **Flexible**: Handles complex queries
4. **Fallback**: Works even without OpenAI

## 🚀 Usage Examples

### Secure Text-to-Cypher
```bash
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Find users who bought products in 2023",
    "execute": true
  }'
```

**What happens:**
1. Our app gets schema via MCP (secure)
2. Sends question + schema to OpenAI (no DB access)
3. OpenAI returns Cypher query
4. We validate the query (security check)
5. We execute via MCP (our credentials)
6. Return results to user

### Security Validation
```bash
curl -X POST http://localhost:8081/execute \
  -H "Content-Type: application/json" \
  -d '{
    "cypher": "DELETE ALL NODES"
  }'
```

**Response:**
```json
{
  "error": "Operation 'DELETE' is not allowed for security reasons.",
  "security_note": "Query blocked by security validation",
  "status": "failed"
}
```

## 🔧 Configuration

### Required: MCP Server (Your DB Credentials)
```env
MCP_SERVER_URL=http://neo4j-mcp:8000
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
```

### Optional: OpenAI (For AI Features)
```env
OPENAI_API_KEY=sk-your-openai-key
```

### Security Settings
```env
MAX_QUERY_LENGTH=1000
```

## 📊 Security vs Functionality Matrix

| Feature | Security Level | Functionality |
|---------|---------------|---------------|
| **LLM Isolation** | 🔒 Maximum | ✅ AI-powered queries |
| **Query Validation** | 🔒 High | ✅ Safe operations only |
| **MCP Execution** | 🔒 Controlled | ✅ Full Neo4j access |
| **Audit Logging** | 🔒 Complete | ✅ Full traceability |

## 🎉 Result

You get **AI-powered text-to-Cypher conversion** with **enterprise-grade security**:

- ✅ LLMs generate smart queries
- ✅ Zero database access for LLMs  
- ✅ Your credentials stay secure
- ✅ All queries validated
- ✅ Complete audit trail

This is the **secure way** to do AI-powered database querying!