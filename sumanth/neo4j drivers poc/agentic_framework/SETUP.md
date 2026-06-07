# Setup Guide - Two-Agent Agentic Neo4j System

## Step-by-Step Setup Instructions

### 1. Prerequisites

**Python Environment:**
```bash
# Install required packages
pip install langchain langchain-openai langchain-community neo4j python-dotenv
```

**If you get import errors, try:**
```bash
pip install --upgrade langchain langchain-openai langchain-community
```

**Neo4j Database:**
- Install Neo4j Desktop or use Neo4j Cloud
- Create a database instance
- Default connection: `bolt://localhost:7687`
- Username: `neo4j`
- Set your password

### 2. Configuration

**Update Configuration File:**
Edit `../simple_ai_system/ai_agent_config.json`:

```json
{
  "llm": {
    "model": "gpt-4",
    "api_key": "your-openai-api-key-here",
    "temperature": 0.1
  },
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "username": "neo4j", 
    "password": "your-neo4j-password-here"
  }
}
```

**Get OpenAI API Key:**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and paste into the config file

### 3. Database Setup (Optional)

If you want to load the student placement data:

```bash
# Navigate to demo folder
cd ../demo

# Run the standalone data loader
python standalone_data_loader.py
```

This will load ~45,000 student records for testing.

### 4. Run the System

```bash
# Navigate to agentic framework
cd agentic_framework

# Run the two-agent system
python langchain_neo4j_agent.py
```

### 5. Test the System

**Query Agent Examples:**
```
🎤 You: How many students are in the database?
🎤 You: Show me students with CGPA above 8.5
🎤 You: Get the database schema
```

**Analytics Agent Examples:**
```
🎤 You: Analyze placement success factors
🎤 You: What trends do you see in the data?
🎤 You: Generate insights about student performance
```

## Troubleshooting

### Common Issues

**1. Configuration File Not Found**
```
❌ Configuration file not found
```
**Solution:** Ensure `ai_agent_config.json` exists in `../simple_ai_system/` directory

**2. OpenAI API Key Error**
```
❌ Invalid API key
```
**Solution:** Check your OpenAI API key in the config file

**3. Neo4j Connection Error**
```
❌ Failed to connect to Neo4j
```
**Solution:** 
- Ensure Neo4j is running
- Check URI, username, and password in config
- Test connection: `bolt://localhost:7687`

**4. Import Errors**
```
❌ Missing dependencies
```
**Solution:** Install required packages:
```bash
pip install langchain langchain-openai neo4j
```

### Verification Steps

**1. Test Neo4j Connection:**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your-password"))
with driver.session() as session:
    result = session.run("RETURN 'Hello Neo4j' as message")
    print(result.single()["message"])
driver.close()
```

**2. Test OpenAI API:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key="your-api-key", model="gpt-4")
response = llm.invoke("Hello, world!")
print(response.content)
```

## System Architecture

```
Project Structure:
├── agentic_framework/
│   ├── langchain_neo4j_agent.py    # Main system
│   ├── router.py                   # AI-powered router
│   ├── agents/
│   │   ├── query_agent.py          # Query specialist
│   │   └── analytics_agent.py      # Analytics specialist
│   ├── tools/
│   │   └── neo4j_tools.py          # LangChain tools
│   ├── README.md                   # Documentation
│   └── agentic_config.json         # System config
└── simple_ai_system/
    ├── ai_agent_config.json        # Main configuration
    └── services/                   # Neo4j services
```

## Expected Output

When running successfully, you should see:

```
🤖 Two-Agent LangChain Neo4j System Ready
🧠 Model: gpt-4
🎯 Agents: Query Agent + Analytics Agent
🔀 Router: AI-powered (no keyword matching)
💡 Features: Specialized agents, autonomous routing

🚀 Welcome to Two-Agent Agentic Neo4j System!
💡 I have two specialized agents that work together:
   🔍 Query Agent: Direct database queries and data retrieval
   📊 Analytics Agent: Data analysis, insights, and complex reasoning
🧠 AI Router automatically selects the best agent for each request
...
```

## Next Steps

1. **Try Different Queries**: Test both simple queries and complex analysis requests
2. **Observe Routing**: Watch how the AI router selects different agents
3. **Explore Commands**: Use `help`, `agents`, `status` commands
4. **Load Your Data**: Use the data loader to import your own CSV files

The system is now ready for demonstration of true agentic behavior with specialized agents and AI-powered routing!