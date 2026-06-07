"""
Neo4j Tools for LangChain Agentic Framework
Implements proper LangChain tools for Neo4j operations
"""

from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
import json
import sys
import os

# Add simple_ai_system to path for service imports
# Note: This creates a dependency on simple_ai_system services
# TODO: Consider making agentic_framework fully independent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'simple_ai_system'))

class Neo4jQueryInput(BaseModel):
    """Input schema for Neo4j Query tool"""
    query_or_description: str = Field(description="Either a Cypher query or natural language description of what to find")
    is_cypher: bool = Field(default=False, description="True if input is a Cypher query, False if natural language")

class Neo4jQueryTool(BaseTool):
    """Advanced Neo4j Query Tool with proper input validation"""
    
    name: str = "neo4j_query"
    description: str = """Execute Neo4j database queries or convert natural language to Cypher.
    
    Use this tool to:
    - Execute Cypher queries directly
    - Convert natural language questions to Cypher and execute them
    - Get data from the student placement database
    - Perform analytics and aggregations
    
    Examples:
    - "Show me all students with CGPA > 8.0"
    - "How many students are placed?"
    - "MATCH (s:Student) WHERE s.cgpa > 8.0 RETURN s" (with is_cypher=True)
    """
    args_schema: Type[BaseModel] = Neo4jQueryInput
    query_service: Any = None  # Will be set during initialization
    
    def __init__(self, query_service):
        super().__init__()
        self.query_service = query_service
    
    def _run(self, query_or_description: str, is_cypher: bool = False) -> str:
        """Execute the Neo4j query tool"""
        try:
            if is_cypher:
                # Direct Cypher execution
                result = self.query_service.execute_query(query_or_description)
            else:
                # Natural language to Cypher
                result = self.query_service.process(query_or_description)
            
            # Format result for agent consumption
            if isinstance(result, dict):
                if result.get('status') == 'success':
                    data = result.get('data', result.get('result', []))
                    if isinstance(data, list) and len(data) > 0:
                        return f"Query executed successfully. Found {len(data)} results:\n{json.dumps(data[:10], indent=2)}"
                    else:
                        return f"Query executed successfully. Result: {json.dumps(result, indent=2)}"
                else:
                    return f"Query failed: {result.get('error', 'Unknown error')}"
            else:
                return f"Query result: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            raise ToolException(f"Neo4j query failed: {str(e)}")

class Neo4jSchemaInput(BaseModel):
    """Input schema for Neo4j Schema tool"""
    action: str = Field(default="get_schema", description="Action to perform: 'get_schema', 'analyze_schema', or 'get_constraints'")

class Neo4jSchemaTool(BaseTool):
    """Neo4j Schema Analysis Tool"""
    
    name: str = "neo4j_schema"
    description: str = """Get Neo4j database schema information and structure.
    
    Use this tool to:
    - Get all node types and their properties
    - Get all relationship types
    - Understand database structure
    - Get constraints and indexes
    
    Actions available:
    - 'get_schema': Get complete schema information
    - 'analyze_schema': Get detailed schema analysis
    - 'get_constraints': Get database constraints
    """
    args_schema: Type[BaseModel] = Neo4jSchemaInput
    schema_service: Any = None  # Will be set during initialization
    
    def __init__(self, schema_service):
        super().__init__()
        self.schema_service = schema_service
    
    def _run(self, action: str = "get_schema") -> str:
        """Execute the schema tool"""
        try:
            result = self.schema_service.process(action)
            
            if isinstance(result, dict):
                if result.get('status') == 'success':
                    schema_data = result.get('schema', result.get('data', {}))
                    
                    # Format schema information nicely
                    formatted_result = "Database Schema Information:\n\n"
                    
                    if 'nodes' in schema_data:
                        formatted_result += "Node Types:\n"
                        for node_type, properties in schema_data['nodes'].items():
                            formatted_result += f"  • {node_type}: {properties}\n"
                    
                    if 'relationships' in schema_data:
                        formatted_result += "\nRelationship Types:\n"
                        for rel in schema_data['relationships']:
                            formatted_result += f"  • {rel}\n"
                    
                    return formatted_result
                else:
                    return f"Schema retrieval failed: {result.get('error', 'Unknown error')}"
            else:
                return f"Schema information: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            raise ToolException(f"Schema retrieval failed: {str(e)}")

class Neo4jDataLoaderInput(BaseModel):
    """Input schema for Neo4j Data Loader tool"""
    user_request: str = Field(description="Description of what data to load or analyze")
    csv_path: Optional[str] = Field(default=None, description="Path to CSV file to load (optional)")

class Neo4jDataLoaderTool(BaseTool):
    """Neo4j Data Loading Tool"""
    
    name: str = "neo4j_data_loader"
    description: str = """Load CSV data into Neo4j database with AI-powered analysis.
    
    Use this tool to:
    - Load new CSV data into the database
    - Analyze CSV structure before loading
    - Handle data transformations automatically
    - Create appropriate node types and relationships
    
    Examples:
    - "Load student data from students.csv"
    - "Import the new dataset and analyze its structure"
    - "Load data from /path/to/data.csv"
    """
    args_schema: Type[BaseModel] = Neo4jDataLoaderInput
    data_loader: Any = None  # Will be set during initialization
    
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
    
    def _run(self, user_request: str, csv_path: Optional[str] = None) -> str:
        """Execute the data loader tool"""
        try:
            result = self.data_loader.process(user_request, csv_path=csv_path)
            
            if isinstance(result, dict):
                if result.get('status') == 'success':
                    details = result.get('details', {})
                    message = result.get('message', 'Data loading completed')
                    
                    formatted_result = f"Data Loading Result: {message}\n\n"
                    
                    if details:
                        formatted_result += "Details:\n"
                        for key, value in details.items():
                            formatted_result += f"  • {key}: {value}\n"
                    
                    return formatted_result
                else:
                    return f"Data loading failed: {result.get('message', 'Unknown error')}"
            else:
                return f"Data loading result: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            raise ToolException(f"Data loading failed: {str(e)}")

class Neo4jAnalyticsTool(BaseTool):
    """Advanced Neo4j Analytics Tool"""
    
    name: str = "neo4j_analytics"
    description: str = """Perform advanced analytics on Neo4j data.
    
    Use this tool for:
    - Statistical analysis of student data
    - Placement rate calculations
    - Correlation analysis
    - Trend identification
    - Performance metrics
    
    Examples:
    - "Calculate placement rate by branch"
    - "Analyze correlation between CGPA and placement"
    - "Show distribution of students by skills"
    """
    query_service: Any = None  # Will be set during initialization
    
    def __init__(self, query_service):
        super().__init__()
        self.query_service = query_service
    
    def _run(self, analysis_request: str) -> str:
        """Execute analytics"""
        try:
            # Use the query service to perform analytics
            analytics_prompt = f"Perform this analytics request: {analysis_request}"
            result = self.query_service.process(analytics_prompt)
            
            if isinstance(result, dict) and result.get('status') == 'success':
                return f"Analytics completed: {json.dumps(result.get('data', result), indent=2)}"
            else:
                return f"Analytics result: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            raise ToolException(f"Analytics failed: {str(e)}")

def create_neo4j_tools(query_service, schema_service, data_loader):
    """Factory function to create all Neo4j tools"""
    
    return [
        Neo4jQueryTool(query_service),
        Neo4jSchemaTool(schema_service),
        Neo4jDataLoaderTool(data_loader),
        Neo4jAnalyticsTool(query_service)
    ]