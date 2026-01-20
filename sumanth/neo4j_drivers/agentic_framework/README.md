# Two-Agent LangChain Agentic Neo4j System

A clean demonstration of multi-agent agentic behavior using LangChain and OpenAI for Neo4j database operations.

## What Makes This Agentic?

🧠 **AI-Powered Agent Routing**: Uses OpenAI LLM to analyze requests and select the optimal specialized agent
🎯 **Specialized Agents**: Two focused agents with different expertise and tool access
🔧 **Autonomous Tool Selection**: Each agent automatically chooses appropriate tools without hardcoded rules
🔄 **Multi-Step Reasoning**: Agents can chain multiple operations to complete complex requests
💭 **Memory & Context**: Each agent maintains conversation history for better understanding
🎨 **Transparent Decision Making**: Shows routing decisions and agent specializations

## Architecture

```
Two-Agent System
├── AI Router (OpenAI LLM)
│   ├── Analyzes user requests
│   ├── Selects optimal agent
│   └── No keyword matching
├── Query Agent (Specialized)
│   ├── Tools: neo4j_query, neo4j_schema
│   ├── Focus: Direct queries, fast data retrieval
│   └── Memory: 8 messages
└── Analytics Agent (Specialized)
    ├── Tools: All 4 tools (query, schema, loader, analytics)
    ├── Focus: Complex analysis, insights, reasoning
    └── Memory: 12 messages
```

## Key Features

- **AI-Powered Routing**: No hardcoded rules - pure AI reasoning for agent selection
- **Specialized Agents**: Each agent optimized for specific types of tasks
- **Tool Distribution**: Query agent has focused tools, Analytics agent has full access
- **Natural Language Interface**: Ask questions in plain English
- **Autonomous Behavior**: Agents decide the best approach for each request
- **Error Recovery**: Handles failures gracefully and suggests alternatives

## Quick Start

### Prerequisites

1. **Python Environment**:
   ```bash
   pip install langchain langchain-openai neo4j
   ```

2. **Neo4j Database**: Running on `bolt://localhost:7687`

3. **OpenAI API Key**: Update `../simple_ai_system/ai_agent_config.json`

### Configuration

Update `../simple_ai_system/ai_agent_config.json`:
```json
{
  "llm": {
    "model": "gpt-4",
    "api_key": "your-openai-api-key",
    "temperature": 0.1
  },
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "your-password"
  }
}
```

### Run the System

```bash
cd agentic_framework
python langchain_neo4j_agent.py
```

## Example Interactions

### Query Agent (Direct Data Retrieval)
```
🎤 You: Show me students with CGPA above 8.5

🧠 AI Router Decision:
   Selected: Query Agent
   Reasoning: AI selected query agent based on request analysis

🔍 Query Agent Processing: Show me students with CGPA above 8.5

🤖 Query Agent Response:
   Specialization: Direct queries and data retrieval

I'll execute a Cypher query to find students with CGPA above 8.5.

[Executes optimized query]

Found 1,247 students with CGPA above 8.5:
- Student 12345: CGPA 9.2, Communication: 4.5, Placed: Yes
- Student 12346: CGPA 8.7, Communication: 4.2, Placed: Yes
...
```

### Analytics Agent (Complex Analysis)
```
🎤 You: What factors most influence placement success?

🧠 AI Router Decision:
   Selected: Analytics Agent
   Reasoning: AI selected analytics agent based on request analysis

📊 Analytics Agent Processing: What factors most influence placement success?

🤖 Analytics Agent Response:
   Specialization: Data analysis and insights

I'll analyze the placement data to identify key success factors. Let me examine correlations and patterns.

[Uses multiple tools: schema analysis, complex queries, statistical analysis]

Based on my analysis of 45,000 student records, here are the key factors:

1. **CGPA (Correlation: 0.73)** - Strongest predictor
   - Students with CGPA > 8.0: 89% placement rate
   - Students with CGPA < 6.0: 23% placement rate

2. **Communication Skills (Correlation: 0.68)** - Critical for success
   - Rating 4.0+: 85% placement rate
   - Rating < 3.0: 31% placement rate

3. **Aptitude Test Score (Correlation: 0.61)** - Technical competency
   - Score > 80: 82% placement rate
   - Score < 60: 28% placement rate
...
```

## Commands

- `help` - Show system capabilities and agent information
- `agents` - Display detailed information about both agents
- `status` - Show system status and configuration
- `clear` - Clear memory from both agents
- `exit` - Quit the system

## Agent Specializations

### 🔍 Query Agent
- **Best for**: Direct queries, specific data retrieval, schema questions
- **Tools**: neo4j_query, neo4j_schema
- **Behavior**: Fast, precise, focused responses
- **Memory**: 8 messages (optimized for quick interactions)
- **Temperature**: 0.0 (deterministic for consistent queries)

### 📊 Analytics Agent  
- **Best for**: Data analysis, insights, complex reasoning, trend identification
- **Tools**: All 4 tools (query, schema, data_loader, analytics)
- **Behavior**: Thorough analysis, multi-step reasoning, comprehensive insights
- **Memory**: 12 messages (more context for complex analysis)
- **Temperature**: 0.2 (slightly creative for analytical insights)

## Why This is Truly Agentic

1. **AI-Powered Routing**: Uses LLM to understand request intent and select optimal agent
2. **No Hardcoded Rules**: Pure AI reasoning, no keyword matching or predefined patterns
3. **Specialized Expertise**: Each agent has distinct capabilities and tool access
4. **Autonomous Decision Making**: Agents choose tools and approaches independently
5. **Adaptive Behavior**: System adjusts complexity based on request type
6. **Transparent Reasoning**: Shows why specific agents and approaches were chosen

## Database Schema

Works with student placement database containing:
- **~45,000 student records**
- **Node types**: Student, Degree, Branch, Skill, PlacementStatus
- **Key properties**: student_id, cgpa, communication_skills, placement_status
- **Relationships**: Students connected to degrees, branches, skills, outcomes

This two-agent system demonstrates how specialized AI agents can work together with intelligent routing to provide both fast data access and deep analytical insights.