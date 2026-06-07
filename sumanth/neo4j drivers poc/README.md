# Neo4j AI Agent Systems

This repository contains two different approaches to AI-powered Neo4j interactions:

## 🔧 Simple AI System (`simple_ai_system/`)

**What it is**: Traditional service-based architecture enhanced with AI
- **Architecture**: Services + LLM calls for decision making
- **Complexity**: Low - Easy to understand and modify
- **Use case**: Straightforward AI-enhanced database operations
- **Framework**: Custom implementation with OpenAI API

**Run it:**
```bash
cd simple_ai_system
python fully_autonomous_neo4j_agent.py
```

## 🤖 True Agentic Framework (`agentic_framework/`)

**What it is**: Proper agentic system using established frameworks
- **Architecture**: True agents with reasoning, planning, tools, and memory
- **Complexity**: High - Production-ready agentic behavior
- **Use case**: Complex autonomous tasks and multi-step reasoning
- **Framework**: **LangChain + OpenAI** (industry standard)

**Run it:**
```bash
cd agentic_framework
python langchain_neo4j_agent.py
```

## 🎯 Key Differences

| Feature | Simple AI System | True Agentic Framework |
|---------|------------------|------------------------|
| **Framework** | Custom | LangChain + OpenAI |
| **Decision Making** | Simple if/else + LLM | Autonomous reasoning |
| **Planning** | Basic action mapping | Multi-step planning |
| **Memory** | None | Conversation memory |
| **Tools** | Direct service calls | LangChain tools |
| **Error Handling** | Basic | Autonomous recovery |
| **Learning** | None | Context learning |
| **Complexity** | Low | High |

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key in ai_agent_config.json
```

## 📊 Demo Data Loader (`demo/`)

Standalone CSV to Neo4j loader without any agent dependencies:

```bash
cd demo
python standalone_data_loader.py
```

## 🔑 Configuration

Update `simple_ai_system/ai_agent_config.json` with your:
- OpenAI API key
- Neo4j connection details

## 🎯 Which One to Use?

- **Simple AI System**: For learning, demos, or straightforward AI-enhanced operations
- **True Agentic Framework**: For production systems requiring autonomous behavior and complex reasoning

Both systems work with the same Neo4j database and can handle the student placement dataset.