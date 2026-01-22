"""
Agent 2: Metrics & Metadata Agent - Purely Agentic
Purpose: Intelligent metric extraction and pipeline observability
"""

import os
import yaml
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from scipy import stats
from datetime import datetime


def get_timestamp() -> str:
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and process config with env vars"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    def substitute(obj):
        if isinstance(obj, dict):
            return {k: substitute(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.getenv(obj[2:-1], obj)
        return obj
    
    return substitute(config)


class MetricsAgent:
    """Purely agentic metrics analysis"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.current_data = None
        self.metrics_results = {}
        
        # Initialize LLM and agent
        llm_config = self.config['llm']
        self.llm = ChatOpenAI(
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens'],
            api_key=llm_config['api_key']
        )
        
        self.tools = self._create_tools()
        self.agent = create_react_agent(self.llm, self.tools)
    
    def _create_tools(self):
        """Create agentic tools for metrics analysis"""
        
        @tool
        def analyze_data_quality(dummy: str = "") -> str:
            """Analyze data quality: nulls, duplicates, outliers. Call FIRST and ONLY ONCE."""
            df = self.current_data
            total_rows = len(df)
            
            # Quality metrics
            null_rates = {col: round((df[col].isnull().sum() / total_rows) * 100, 2) for col in df.columns}
            duplicate_rate = round((df.duplicated().sum() / total_rows) * 100, 2)
            
            # Outliers for numeric columns
            outlier_info = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                outlier_info[col] = round((len(outliers) / total_rows) * 100, 2)
            
            self.metrics_results["data_quality"] = {
                "null_rates": null_rates,
                "duplicate_rate": duplicate_rate,
                "outlier_rates": outlier_info
            }
            
            return f"✅ DATA QUALITY DONE. Nulls: {max(null_rates.values())}% max, Duplicates: {duplicate_rate}%, Outliers analyzed. → Step 2: calculate_statistics"
        
        @tool
        def calculate_statistics(dummy: str = "") -> str:
            """Calculate statistical metrics for all fields. Call SECOND and ONLY ONCE."""
            df = self.current_data
            
            # Numeric stats
            numeric_stats = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                data = df[col].dropna()
                numeric_stats[col] = {
                    "mean": round(data.mean(), 2),
                    "median": round(data.median(), 2),
                    "std": round(data.std(), 2),
                    "skew": round(stats.skew(data), 2)
                }
            
            # Categorical stats
            categorical_stats = {}
            for col in df.select_dtypes(include=['object']).columns:
                counts = df[col].value_counts()
                categorical_stats[col] = {
                    "unique_count": len(counts),
                    "top_value": str(counts.index[0]),
                    "top_count": int(counts.iloc[0])
                }
            
            self.metrics_results["statistics"] = {
                "numeric": numeric_stats,
                "categorical": categorical_stats
            }
            
            return f"✅ STATISTICS DONE. {len(numeric_stats)} numeric, {len(categorical_stats)} categorical columns analyzed. → Step 3: analyze_cardinality"
        
        @tool
        def analyze_cardinality(dummy: str = "") -> str:
            """Analyze cardinality for indexing recommendations. Call THIRD and ONLY ONCE."""
            df = self.current_data
            total_rows = len(df)
            
            cardinality_info = {}
            index_recommendations = []
            
            for col in df.columns:
                unique_count = df[col].nunique()
                ratio = unique_count / total_rows
                
                cardinality_info[col] = {
                    "unique_count": unique_count,
                    "ratio": round(ratio, 4)
                }
                
                if ratio > 0.7:
                    index_recommendations.append(f"{col}: Primary index (high cardinality)")
                elif 0.05 < ratio <= 0.3:
                    index_recommendations.append(f"{col}: Secondary index (medium cardinality)")
            
            self.metrics_results["cardinality"] = {
                "column_info": cardinality_info,
                "index_recommendations": index_recommendations
            }
            
            return f"✅ CARDINALITY DONE. {len(index_recommendations)} indexing recommendations generated. → Step 4: analyze_relationships"
        
        @tool
        def analyze_relationships(dummy: str = "") -> str:
            """Analyze entity relationships and graph patterns. Call FOURTH and ONLY ONCE."""
            df = self.current_data
            
            # Identify entities
            entities = {}
            for col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                is_id = any(keyword in col.lower() for keyword in ['id', 'key', 'code'])
                
                if is_id and unique_ratio > 0.8:
                    entity_type = "Primary Entity"
                elif is_id:
                    entity_type = "Foreign Key"
                else:
                    entity_type = "Attribute"
                
                entities[col] = {
                    "type": entity_type,
                    "unique_ratio": round(unique_ratio, 3)
                }
            
            # Graph projections
            primary_entities = [col for col, info in entities.items() if info["type"] == "Primary Entity"]
            estimated_nodes = sum(df[col].nunique() for col in primary_entities) if primary_entities else len(df)
            
            self.metrics_results["relationships"] = {
                "entities": entities,
                "graph_projection": {
                    "estimated_nodes": estimated_nodes,
                    "estimated_edges": len(df),
                    "density": "High" if estimated_nodes < 1000 else "Medium"
                }
            }
            
            return f"✅ RELATIONSHIPS DONE. {len(primary_entities)} primary entities identified. Graph density: {self.metrics_results['relationships']['graph_projection']['density']}. → Step 5: generate_insights"
        
        @tool
        def generate_insights(dummy: str = "") -> str:
            """Generate business insights and domain KPIs. Call FIFTH and ONLY ONCE."""
            df = self.current_data
            column_names = ' '.join(df.columns).lower()
            
            # Domain detection
            if any(word in column_names for word in ['student', 'grade', 'placement']):
                domain = "Education"
            elif any(word in column_names for word in ['customer', 'order', 'sales']):
                domain = "E-commerce"
            else:
                domain = "General"
            
            # Key metrics
            completeness = round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
            
            insights = {
                "domain": domain,
                "data_completeness": completeness,
                "total_records": len(df),
                "key_recommendations": [
                    f"Data is {completeness}% complete",
                    f"Domain: {domain} - consider domain-specific KPIs",
                    f"Ready for {self.metrics_results['relationships']['graph_projection']['density'].lower()} density graph processing"
                ]
            }
            
            self.metrics_results["business_insights"] = insights
            
            return f"✅ INSIGHTS DONE. Domain: {domain}, Completeness: {completeness}%. → Step 6: create_final_report"
        
        @tool
        def create_final_report(dummy: str = "") -> str:
            """Create comprehensive final report. Call SIXTH and FINAL step."""
            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_shape": self.current_data.shape,
                "analyses_completed": list(self.metrics_results.keys()),
                "summary": self.metrics_results
            }
            
            return f"✅ FINAL REPORT COMPLETE. All {len(self.metrics_results)} analyses done: {', '.join(self.metrics_results.keys())}. ANALYSIS FINISHED - PROVIDE SUMMARY NOW."
        
        return [analyze_data_quality, calculate_statistics, analyze_cardinality, 
                analyze_relationships, generate_insights, create_final_report]
    
    async def analyze_data(self, data: pd.DataFrame, schema_info: str = "") -> Dict[str, Any]:
        """Pure agentic data analysis"""
        try:
            self.current_data = data
            
            instruction = f"""Analyze {data.shape} dataset. Execute 6 tools once each:
1. analyze_data_quality
2. calculate_statistics  
3. analyze_cardinality
4. analyze_relationships
5. generate_insights
6. create_final_report

Start now."""
            
            messages = [HumanMessage(content=instruction)]
            result = await self.agent.ainvoke({"messages": messages})
            
            return {
                "status": "success",
                "agent_output": result["messages"][-1].content,
                "metrics_results": self.metrics_results,
                "data_shape": data.shape
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_metrics_results(self) -> Dict[str, Any]:
        """Get metrics results"""
        return self.metrics_results
    
    def save_metrics_report(self, output_path: str) -> bool:
        """Save metrics report with timestamp"""
        try:
            # Create timestamped filename
            base_name, ext = os.path.splitext(output_path)
            timestamp_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
            timestamped_path = f"{base_name}{timestamp_suffix}{ext}"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(timestamped_path), exist_ok=True)
            
            # Add timestamp metadata to the report
            report_with_metadata = {
                "generated_at": get_timestamp(),
                "report_version": "1.0",
                **self.metrics_results
            }
            
            print(f"[{get_timestamp()}] 💾 Metrics Agent: Saving metrics report to {timestamped_path}")
            
            with open(timestamped_path, 'w') as f:
                json.dump(report_with_metadata, f, indent=2, default=str)
            
            print(f"[{get_timestamp()}] ✅ Metrics Agent: Successfully saved metrics report with {len(self.metrics_results)} categories")
            return True
            
        except Exception as e:
            print(f"[{get_timestamp()}] ❌ Metrics Agent: Failed to save report - {str(e)}")
            return False


def create_metrics_agent(config_path: str = "config/config.yaml"):
    """Create purely agentic metrics agent"""
    return MetricsAgent(config_path)