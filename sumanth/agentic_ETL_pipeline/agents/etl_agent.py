"""
Agentic ETL Agent using LangGraph Framework
"""

import os
import yaml
import pandas as pd
import json
from typing import Dict, Any, Optional, List, Annotated
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from datetime import datetime


def get_timestamp() -> str:
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and process config with env vars"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Simple env var substitution
    def substitute(obj):
        if isinstance(obj, dict):
            return {k: substitute(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.getenv(obj[2:-1], obj)
        return obj
    
    return substitute(config)


class AgenticETL:
    """LangGraph-based agentic ETL system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.current_data = None
        
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_llm(self):
        """Create LLM from config"""
        llm_config = self.config['llm']
        return ChatOpenAI(
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens'],
            api_key=llm_config['api_key']
        )
    
    def _create_tools(self):
        """Create LangChain tools for ETL operations"""
        
        @tool
        def analyze_file(file_path: str) -> str:
            """Analyze a file to understand its type, structure and content"""
            try:
                if not os.path.exists(file_path):
                    return f"Error: File {file_path} not found"
                
                file_size = os.path.getsize(file_path)
                max_size = self.config['etl']['max_file_size_mb'] * 1024 * 1024
                
                if file_size > max_size:
                    return f"Error: File too large ({file_size} bytes)"
                
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # Read sample content
                sample_chars = self.config['etl']['sample_content_chars']
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        sample_content = f.read(sample_chars)
                except:
                    sample_content = "Binary file or encoding issue"
                
                return json.dumps({
                    "file_path": file_path,
                    "extension": file_ext,
                    "size_bytes": file_size,
                    "sample_content": sample_content
                })
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def parse_csv(file_path: str, separator: str = ",") -> str:
            """Parse a CSV file with specified separator"""
            try:
                df = pd.read_csv(file_path, sep=separator)
                self.current_data = df
                return json.dumps({
                    "status": "success",
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "sample": df.head(3).to_dict('records')
                })
            except Exception as e:
                return f"Error parsing CSV: {str(e)}"
        
        @tool
        def parse_excel(file_path: str, sheet_name: str = None) -> str:
            """Parse an Excel file, optionally specify sheet name"""
            try:
                if sheet_name and sheet_name != "None":
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                else:
                    df = pd.read_excel(file_path)
                self.current_data = df
                return json.dumps({
                    "status": "success",
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "sample": df.head(3).to_dict('records')
                })
            except Exception as e:
                return f"Error parsing Excel: {str(e)}"
        
        @tool
        def parse_json(file_path: str) -> str:
            """Parse a JSON file"""
            try:
                df = pd.read_json(file_path)
                self.current_data = df
                return json.dumps({
                    "status": "success",
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "sample": df.head(3).to_dict('records')
                })
            except Exception as e:
                return f"Error parsing JSON: {str(e)}"
        
        @tool
        def clean_data(operations: str) -> str:
            """Clean data with any operations. Describe what you want to do in natural language."""
            try:
                if self.current_data is None:
                    return "Error: No data loaded. Parse a file first."
                
                df = self.current_data.copy()
                original_shape = df.shape
                applied_ops = []
                
                # Convert operations to lowercase for flexible matching
                ops_lower = operations.lower()
                
                # Flexible duplicate removal
                if any(word in ops_lower for word in ["duplicate", "duplicates", "duplicate rows", "repeated", "same rows"]):
                    before = len(df)
                    df = df.drop_duplicates()
                    after = len(df)
                    if before != after:
                        applied_ops.append(f"Removed {before - after} duplicate rows")
                
                # Flexible null handling
                if any(word in ops_lower for word in ["null", "nulls", "missing", "na", "nan", "empty values", "blank"]):
                    if any(word in ops_lower for word in ["drop", "remove", "delete", "eliminate"]):
                        # Drop nulls
                        before = len(df)
                        df = df.dropna()
                        after = len(df)
                        if before != after:
                            applied_ops.append(f"Dropped {before - after} rows with null values")
                    elif any(word in ops_lower for word in ["fill", "replace", "mean", "average"]):
                        # Fill with mean
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                            applied_ops.append(f"Filled null values in {len(numeric_cols)} numeric columns with mean")
                    elif any(word in ops_lower for word in ["mode", "most frequent", "common"]):
                        # Fill with mode
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                mode_val = df[col].mode()
                                if not mode_val.empty:
                                    df[col] = df[col].fillna(mode_val[0])
                        applied_ops.append("Filled null values in text columns with mode")
                
                # Flexible string cleaning
                if any(word in ops_lower for word in ["empty string", "blank string", "empty text", ""]):
                    df = df.replace('', pd.NA)
                    applied_ops.append("Converted empty strings to NA")
                
                # Flexible column standardization
                if any(word in ops_lower for word in ["column", "columns", "standardize", "normalize", "clean names", "format names"]):
                    old_cols = list(df.columns)
                    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
                    if old_cols != list(df.columns):
                        applied_ops.append("Standardized column names (lowercase, underscores)")
                
                # Flexible outlier removal
                if any(word in ops_lower for word in ["outlier", "outliers", "extreme values", "anomal"]):
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    outliers_removed = 0
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        before = len(df)
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        outliers_removed += before - len(df)
                    if outliers_removed > 0:
                        applied_ops.append(f"Removed {outliers_removed} outlier rows")
                
                # Apply default operations if no specific operations were detected
                if not applied_ops:
                    default_ops = self.config['etl']['default_operations']
                    for op in default_ops:
                        if op == 'remove_duplicates':
                            before = len(df)
                            df = df.drop_duplicates()
                            after = len(df)
                            if before != after:
                                applied_ops.append(f"Applied default: removed {before - after} duplicates")
                        elif op == 'drop_nulls':
                            before = len(df)
                            df = df.dropna()
                            after = len(df)
                            if before != after:
                                applied_ops.append(f"Applied default: dropped {before - after} null rows")
                
                self.current_data = df
                
                return json.dumps({
                    "status": "success",
                    "original_shape": original_shape,
                    "new_shape": df.shape,
                    "operations_applied": applied_ops,
                    "operations_requested": operations,
                    "sample": df.head(3).to_dict('records')
                })
            except Exception as e:
                return f"Error cleaning data: {str(e)}"
        
        @tool
        def get_data_info(dummy: str = "") -> str:
            """Get information about the currently loaded data"""
            try:
                if self.current_data is None:
                    return "No data loaded"
                
                df = self.current_data
                return json.dumps({
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "null_counts": df.isnull().sum().to_dict(),
                    "sample": df.head(3).to_dict('records')
                })
            except Exception as e:
                return f"Error getting data info: {str(e)}"
        
        return [analyze_file, parse_csv, parse_excel, parse_json, clean_data, get_data_info]
    
    def _create_agent(self):
        """Create LangGraph ReAct agent"""
        
        # Create agent using LangGraph with system message
        agent = create_react_agent(self.llm, self.tools)
        
        return agent
    
    async def process_file(self, file_path: str, instructions: str = "") -> Dict[str, Any]:
        """Process file using agentic approach"""
        try:
            # Create instruction for the agent
            instruction = f"""Process {file_path}:
1. analyze_file
2. parse (csv/excel/json)
3. clean_data
4. Done

{instructions}"""
            
            # Run the agent
            messages = [HumanMessage(content=instruction)]
            result = await self.agent.ainvoke({"messages": messages})
            
            # Extract the final message
            final_message = result["messages"][-1].content
            
            return {
                "status": "success",
                "agent_output": final_message,
                "final_data_shape": self.current_data.shape if self.current_data is not None else None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get the processed data"""
        return self.current_data
    
    def save_processed_data(self, output_path: str) -> bool:
        """Save processed data to file with timestamp"""
        try:
            if self.current_data is None:
                print(f"[{get_timestamp()}] ❌ ETL Agent: No data to save")
                return False
            
            # Create timestamped filename
            base_name, ext = os.path.splitext(output_path)
            timestamp_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
            timestamped_path = f"{base_name}{timestamp_suffix}{ext}"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(timestamped_path), exist_ok=True)
            
            # Add timestamp to data as metadata
            data_with_metadata = self.current_data.copy()
            
            print(f"[{get_timestamp()}] 💾 ETL Agent: Saving processed data to {timestamped_path}")
            
            if ext == '.csv':
                data_with_metadata.to_csv(timestamped_path, index=False)
            elif ext in ['.xlsx', '.xls']:
                data_with_metadata.to_excel(timestamped_path, index=False)
            elif ext == '.json':
                with open(timestamped_path, 'w') as f:
                    data_with_metadata.to_json(f, orient='records', indent=2)
            else:
                # Default to CSV
                data_with_metadata.to_csv(timestamped_path, index=False)
            
            print(f"[{get_timestamp()}] ✅ ETL Agent: Successfully saved {len(data_with_metadata)} records to {timestamped_path}")
            return True
            
        except Exception as e:
            print(f"[{get_timestamp()}] ❌ ETL Agent: Failed to save data - {str(e)}")
            return False


def create_etl_agent(config_path: str = "config/config.yaml"):
    """Create ETL agent using LangGraph agentic framework"""
    return AgenticETL(config_path)