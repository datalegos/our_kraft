# Research & Analysis Agent Demo

A demonstration of **Agentic AI** capabilities, showcasing how AI agents can autonomously perform complex multi-step tasks using tools and reasoning.

## 🎯 Two-Agent System

This project now includes **two collaborating agents**:

1. **Research Agent** - Conducts research, searches the web, and generates reports
2. **Review Agent** - Reviews and evaluates the Research Agent's responses for quality and completeness

## 🤖 What is an Agentic AI?

An **Agentic AI** is an AI system that can:
- **Reason and Plan**: Break down complex tasks into steps
- **Use Tools**: Interact with external systems (web search, file systems, APIs)
- **Make Decisions**: Choose which tools to use and when
- **Iterate**: Refine its approach based on results
- **Complete Tasks**: Execute multi-step workflows autonomously

## 🎯 This Demo Agent

This **Research & Analysis Agent** demonstrates:
1. **Query Understanding**: Interprets research questions
2. **Web Search**: Uses DuckDuckGo to find relevant information
3. **Information Synthesis**: Combines multiple sources into coherent insights
4. **Report Generation**: Creates formatted markdown reports
5. **File Management**: Saves reports automatically

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Conda (optional, but recommended)

### Setup

1. **Clone or navigate to this directory**

2. **Create a conda environment** (if using conda):
```bash
conda create -n geopulse python=3.10
conda activate geopulse
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up your API key**:
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_actual_api_key_here
```

### Run the Demo

**Option 1: Multi-Agent System** (Recommended - Shows agent collaboration)
```bash
conda activate agents
python multi_agent_demo.py
```

**Option 2: Research Agent Only**
```bash
conda activate agents
python demo.py
```

**Option 3: Test Script**
```bash
conda activate agents
python test_agent.py
```

## 📊 Use Cases for Agentic AI

### 1. **Research & Intelligence**
- Market research and competitor analysis
- Trend monitoring and reporting
- Academic literature reviews
- News aggregation and summarization

### 2. **Code Development**
- Automated code reviews
- Technical debt analysis
- Documentation generation
- Test case creation

### 3. **Data Analysis**
- Automated ETL pipelines
- Report generation
- Anomaly detection
- Predictive analytics

### 4. **Business Automation**
- Customer support routing
- Invoice processing
- Email triage and responses
- Workflow orchestration

### 5. **Content Creation**
- Multi-source content synthesis
- SEO-optimized article writing
- Social media content planning
- Technical documentation

## 🏗️ Architecture

```
User Query
    ↓
Research Agent
    ↓
┌─────────────────┐
│ 1. Plan Steps   │
│ 2. Search Web   │
│ 3. Synthesize   │
│ 4. Generate     │
│ 5. Save Report  │
└─────────────────┘
    ↓
Final Report
```

## 🔧 How It Works

1. **Agent receives a query** (e.g., "What are AI trends in 2024?")

2. **Agent plans the approach**:
   - Identifies search terms
   - Determines what information is needed
   - Plans the research strategy

3. **Agent uses tools**:
   - Calls `web_search` tool with relevant queries
   - Gathers information from multiple sources

4. **Agent synthesizes**:
   - Combines findings from different sources
   - Identifies key insights and trends
   - Structures information logically

5. **Agent generates output**:
   - Creates a formatted report
   - Saves it to the `reports/` directory

## 📁 Project Structure

```
agent/
├── research_agent.py    # Research Agent - conducts research and generates reports
├── review_agent.py      # Review Agent - evaluates research quality
├── multi_agent_demo.py   # Multi-agent collaboration demo
├── demo.py              # Interactive demo script (Research Agent only)
├── test_agent.py        # Test script for Research Agent
├── clean_report.py      # Utility to clean report files
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # This file
└── reports/             # Generated reports (created automatically)
```

## 🎓 Key Concepts Demonstrated

### Tool Use
Agents can call external functions (tools) to extend their capabilities:
- `web_search`: Searches the internet
- `save_report`: Writes files to disk

### Reasoning
The agent uses an LLM to:
- Understand context
- Plan multi-step workflows
- Make decisions about tool usage

### Autonomy
The agent operates independently:
- No human intervention needed
- Handles errors gracefully
- Iterates until task completion

## 🔮 Extending This Agent

You can enhance this agent by adding:

- **More Tools**:
  - Database queries
  - API integrations
  - Code execution
  - Image generation

- **Better Reasoning**:
  - Multi-agent collaboration
  - Memory and context retention
  - Learning from past queries

- **Specialized Domains**:
  - Financial analysis agent
  - Medical research agent
  - Legal document analysis agent

## 📝 Example Queries

Try these queries in the interactive demo:

- "What are the latest trends in AI agents in 2024?"
- "Compare the top 3 programming languages for AI development"
- "What are the key challenges in deploying LLM agents in production?"
- "Research the impact of AI on software development workflows"
- "What are the best practices for building production-ready AI agents?"

## 🤝 For Your Team Demo

### Talking Points

1. **What makes this "agentic"?**
   - It uses tools autonomously
   - It plans and executes multi-step workflows
   - It makes decisions without constant human input

2. **Real-world applications**:
   - Research automation
   - Code analysis and generation
   - Business process automation
   - Customer service automation

3. **Advantages over simple chatbots**:
   - Can interact with external systems
   - Can perform complex, multi-step tasks
   - Can adapt and iterate
   - Can handle errors and retry

4. **Limitations to discuss**:
   - Requires API keys and costs
   - May need validation for critical tasks
   - Can make mistakes (hallucinations)
   - Requires careful prompt engineering

## 📚 Further Reading

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Agentic AI Patterns](https://www.anthropic.com/research)

## ⚠️ Notes

- This is a **demo** for educational purposes
- Web search uses DuckDuckGo (free, no API key needed)
- Reports are saved in the `reports/` directory
- Make sure you have sufficient OpenAI API credits

## 🐛 Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure you created a `.env` file with your API key

**"No results found"**
- The web search might be rate-limited, try again later

**Import errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`

---

**Built for demonstrating Agentic AI capabilities** 🚀

