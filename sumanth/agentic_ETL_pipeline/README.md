# 🚀 Agentic ETL Pipeline

A fully agentic ETL system using LangChain framework that processes data through three intelligent agents and loads it into a Neo4j graph database.

## 📁 Project Structure

```
agentic_ETL_pipeline/
├── agents/                    # All agent implementations
│   ├── etl_agent.py          # Agent 1: Extract, Transform, Load
│   ├── metrics_agent.py      # Agent 2: Data Quality & Metrics Analysis  
│   ├── neo4j_agent.py        # Agent 3: Neo4j Graph Database Loading
│   └── orchestrator.py       # Central coordinator for all agents
├── config/                    # Configuration files
│   └── config.yaml           # System configuration
├── data/                      # Input data files
│   └── train.csv             # Sample dataset
├── output/                    # Generated output files
│   ├── processed_data.csv    # Cleaned data from Agent 1
│   ├── metrics_report.json   # Analysis from Agent 2
│   └── neo4j_load_report.json # Graph loading results from Agent 3
├── main.py                    # Main entry point
└── requirements.txt           # Python dependencies
```

## 🎯 Agent Pipeline

### **Agent 1: ETL Agent**
- **Purpose**: File analysis, parsing, and data cleaning
- **Framework**: LangChain ReAct with custom tools
- **Capabilities**: 
  - Automatic file type detection (CSV, Excel, JSON)
  - Intelligent data cleaning based on natural language instructions
  - Flexible parsing with error handling

### **Agent 2: Metrics Agent** 
- **Purpose**: Data quality analysis and graph recommendations
- **Framework**: LangChain with sequential tool execution
- **Capabilities**:
  - Data quality metrics (nulls, duplicates, outliers)
  - Statistical analysis (mean, median, distributions)
  - Cardinality analysis for indexing recommendations
  - Relationship detection for graph modeling

### **Agent 3: Neo4j Agent**
- **Purpose**: Intelligent graph database loading
- **Framework**: LangChain with connection management
- **Capabilities**:
  - Metrics-driven schema creation
  - Categorical node extraction and relationships
  - Performance-based labeling and similarity analysis
  - Batch processing for large datasets

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Neo4j
Update `config/config.yaml` with your Neo4j credentials:
```yaml
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j" 
  password: "your-password"
```

### 3. Configure OpenAI API
Update `config/config.yaml` with your OpenAI API key:
```yaml
llm:
  api_key: "your-openai-api-key"
```

### 4. Run the Pipeline
```bash
python main.py
```

## 📊 Expected Results

The pipeline will create:
- **45,000 Student nodes** with numeric properties
- **13 Categorical nodes**: Gender (2), Degree (4), Branch (5), PlacementStatus (2)
- **180,000 Relationships** connecting students to categories
- **Performance labels**: HighPerformer, Placed, HighlySkilled

## 🔧 Configuration

All settings are in `config/config.yaml`:
- **File paths**: Input data and output locations
- **Neo4j connection**: Database credentials and settings
- **LLM settings**: Model, temperature, token limits
- **Agent behavior**: Timeouts, batch sizes, processing options

## 🎯 Key Features

- **Pure Agentic**: All decisions made by LLM agents
- **Loop Prevention**: Sequential execution with clear termination
- **Connection Management**: Robust Neo4j connection handling
- **Field Mapping**: Automatic handling of column name variations
- **Existing Data**: Smart detection and handling of existing database content
- **Batch Processing**: Efficient handling of large datasets

## 📈 Graph Database Schema

```cypher
(:Student {student_id, age, cgpa, coding_skills, ...})
-[:HAS_GENDER]-> (:Gender {name: "Female"})
-[:HAS_DEGREE]-> (:Degree {name: "B.Tech"})  
-[:HAS_BRANCH]-> (:Branch {name: "CSE"})
-[:HAS_PLACEMENT_STATUS]-> (:PlacementStatus {name: "Placed"})
```

## 🛠️ Troubleshooting

- **Rate Limits**: The system respects OpenAI rate limits with proper error handling
- **Connection Issues**: Neo4j agent includes retry logic and connection validation
- **Memory Usage**: Batch processing prevents memory issues with large datasets
- **Data Conflicts**: Automatic detection and handling of existing data

## 📝 License

This project demonstrates agentic ETL capabilities using LangChain framework for educational and development purposes.