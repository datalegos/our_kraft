"""
Neo4j Agent - Direct execution without LLM overhead
Optimized for linear operations that don't require AI decision-making
"""

import os
import yaml
import pandas as pd
import json
from typing import Dict, Any
from neo4j import GraphDatabase
import logging
from datetime import datetime


def get_timestamp() -> str:
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load config directly"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class Neo4jAgent:
    """Direct Neo4j operations without LLM overhead"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.driver = None
        self.load_results = {}
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _connect_to_neo4j(self) -> bool:
        """Connect to Neo4j database"""
        try:
            neo4j_config = self.config['neo4j']
            self.driver = GraphDatabase.driver(
                neo4j_config['uri'], 
                auth=(neo4j_config['username'], neo4j_config['password']),
                max_connection_lifetime=3600,
                max_connection_pool_size=10
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1").single()
            
            print(f"[{get_timestamp()}] ✅ Neo4j Agent: Connected to database")
            self.logger.info("Connected to Neo4j successfully")
            return True
            
        except Exception as e:
            print(f"[{get_timestamp()}] ❌ Neo4j Agent: Connection failed - {e}")
            self.logger.error(f"Neo4j connection failed: {e}")
            return False
    
    def _analyze_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract graph structure from metrics"""
        print(f"[{get_timestamp()}] 📊 Neo4j Agent: Analyzing metrics for graph structure")
        
        # Get categorical fields
        stats = metrics_data.get('statistics', {})
        categorical_fields = list(stats.get('categorical', {}).keys())
        
        # Get primary entities
        cardinality = metrics_data.get('cardinality', {})
        column_info = cardinality.get('column_info', {})
        primary_entities = []
        for col, info in column_info.items():
            if info.get('ratio', 0) >= 0.9:
                primary_entities.append(col)
        
        analysis = {
            "categorical_fields": categorical_fields,
            "primary_entities": primary_entities
        }
        
        print(f"[{get_timestamp()}] 📊 Neo4j Agent: Found {len(categorical_fields)} categorical fields, {len(primary_entities)} primary entities")
        return analysis
    
    def _create_field_mapping(self, df: pd.DataFrame, categorical_fields: list) -> Dict[str, str]:
        """Map metrics fields to actual column names"""
        field_mapping = {}
        df_columns = df.columns.tolist()
        
        for field in categorical_fields:
            # Try different case variations
            possible_names = [
                field,
                field.title(),
                field.upper(),
                field.replace('_', ' ').title().replace(' ', '_'),
            ]
            
            for possible_name in possible_names:
                if possible_name in df_columns:
                    field_mapping[field] = possible_name
                    break
        
        print(f"[{get_timestamp()}] 🔗 Neo4j Agent: Mapped {len(field_mapping)} fields for relationships")
        return field_mapping
    
    def _clear_existing_data(self):
        """Clear existing Neo4j data"""
        print(f"[{get_timestamp()}] 🧹 Neo4j Agent: Clearing existing data")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("Cleared existing Neo4j data")
    
    def _create_categorical_nodes(self, df: pd.DataFrame, field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Create categorical nodes"""
        print(f"[{get_timestamp()}] 🏷️ Neo4j Agent: Creating categorical nodes")
        nodes_created = {}
        
        with self.driver.session() as session:
            for field, column_name in field_mapping.items():
                unique_values = df[column_name].dropna().unique()
                label = field.title().replace('_', '')
                
                for value in unique_values:
                    query = f"MERGE (n:{label} {{name: $value}})"
                    session.run(query, value=str(value))
                
                nodes_created[field] = {
                    "label": label,
                    "column_name": column_name,
                    "count": len(unique_values)
                }
                
                print(f"[{get_timestamp()}] 🏷️ Neo4j Agent: Created {len(unique_values)} {label} nodes")
                self.logger.info(f"Created {len(unique_values)} {label} nodes")
        
        return nodes_created
    
    def _create_student_nodes(self, df: pd.DataFrame, field_mapping: Dict[str, str]) -> int:
        """Create Student nodes with properties"""
        print(f"[{get_timestamp()}] 👥 Neo4j Agent: Creating Student nodes")
        
        # Property columns (exclude categorical fields)
        mapped_columns = list(field_mapping.values())
        property_columns = [col for col in df.columns if col not in mapped_columns]
        
        # Build dynamic property assignments
        property_assignments = []
        for col in property_columns:
            property_assignments.append(f"{col}: student.{col}")
        property_string = ", ".join(property_assignments)
        
        total_created = 0
        batch_size = 1000
        
        with self.driver.session() as session:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                query = f"""
                UNWIND $students as student
                CREATE (s:Student {{
                    {property_string}
                }})
                """
                
                students_data = batch.to_dict('records')
                session.run(query, students=students_data)
                total_created += len(students_data)
        
        print(f"[{get_timestamp()}] 👥 Neo4j Agent: Created {total_created} Student nodes")
        self.logger.info(f"Created {total_created} Student nodes")
        return total_created
    
    def _create_relationships(self, df: pd.DataFrame, field_mapping: Dict[str, str], primary_key: str) -> Dict[str, int]:
        """Create relationships between Students and categorical nodes"""
        print(f"[{get_timestamp()}] 🔗 Neo4j Agent: Creating relationships")
        relationships_created = {}
        
        with self.driver.session() as session:
            for field, column_name in field_mapping.items():
                label = field.title().replace('_', '')
                relationship_name = f"HAS_{field.upper()}"
                
                total_rels = 0
                batch_size = 1000
                
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    query = f"""
                    UNWIND $batch as row
                    MATCH (s:Student {{{primary_key}: row.{primary_key}}})
                    MATCH (n:{label} {{name: row.{column_name}}})
                    MERGE (s)-[:{relationship_name}]->(n)
                    """
                    
                    batch_data = []
                    for _, row in batch.iterrows():
                        if pd.notna(row[column_name]):
                            batch_data.append({
                                primary_key: row[primary_key],
                                column_name: str(row[column_name])
                            })
                    
                    if batch_data:
                        session.run(query, batch=batch_data)
                        total_rels += len(batch_data)
                
                relationships_created[field] = total_rels
                print(f"[{get_timestamp()}] 🔗 Neo4j Agent: Created {total_rels} {relationship_name} relationships")
                self.logger.info(f"Created {total_rels} {relationship_name} relationships")
        
        return relationships_created
    
    def _get_final_stats(self) -> Dict[str, Any]:
        """Get final database statistics"""
        print(f"[{get_timestamp()}] 📈 Neo4j Agent: Collecting final statistics")
        
        with self.driver.session() as session:
            # Count nodes by label
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            
            node_counts = {}
            for label in labels:
                count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = count_result.single()["count"]
                node_counts[label] = count
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
            total_rels = result.single()["total_rels"]
            
            stats = {
                "total_nodes": sum(node_counts.values()),
                "total_relationships": total_rels,
                "nodes_by_label": node_counts
            }
            
            print(f"[{get_timestamp()}] 📈 Neo4j Agent: Final stats - {stats['total_nodes']} nodes, {stats['total_relationships']} relationships")
            return stats
    
    async def load_to_neo4j(self, processed_data: pd.DataFrame, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load data to Neo4j - DIRECT EXECUTION (No LLM)"""
        try:
            print(f"[{get_timestamp()}] 🚀 Neo4j Agent: Starting data loading")
            self.logger.info("Starting Neo4j data loading...")
            
            # Step 1: Connect
            if not self._connect_to_neo4j():
                return {"status": "error", "error": "Failed to connect to Neo4j"}
            
            # Step 2: Analyze metrics
            analysis = self._analyze_metrics(metrics_data)
            self.load_results["analysis"] = analysis
            
            # Step 3: Create field mapping
            field_mapping = self._create_field_mapping(processed_data, analysis["categorical_fields"])
            self.load_results["field_mapping"] = field_mapping
            
            if not field_mapping:
                return {"status": "error", "error": "No categorical fields found for relationships"}
            
            # Step 4: Clear existing data
            self._clear_existing_data()
            
            # Step 5: Create categorical nodes
            categorical_nodes = self._create_categorical_nodes(processed_data, field_mapping)
            self.load_results["categorical_nodes"] = categorical_nodes
            
            # Step 6: Create Student nodes
            primary_key = analysis["primary_entities"][0] if analysis["primary_entities"] else "student_id"
            students_created = self._create_student_nodes(processed_data, field_mapping)
            self.load_results["students_created"] = students_created
            
            # Step 7: Create relationships
            relationships = self._create_relationships(processed_data, field_mapping, primary_key)
            self.load_results["relationships"] = relationships
            
            # Step 8: Get final stats
            final_stats = self._get_final_stats()
            self.load_results["final_stats"] = final_stats
            
            print(f"[{get_timestamp()}] ✅ Neo4j Agent: Loading completed successfully")
            self.logger.info("Neo4j data loading completed successfully")
            
            return {
                "status": "success",
                "agent_output": f"Successfully loaded {final_stats['total_nodes']} nodes and {final_stats['total_relationships']} relationships",
                "data_shape": processed_data.shape,
                "load_results": self.load_results
            }
            
        except Exception as e:
            print(f"[{get_timestamp()}] ❌ Neo4j Agent: Loading failed - {e}")
            self.logger.error(f"Neo4j loading failed: {e}")
            return {"status": "error", "error": str(e)}
        
        finally:
            if self.driver:
                self.driver.close()
    
    def get_load_results(self) -> Dict[str, Any]:
        """Get loading results"""
        return self.load_results
    
    def save_load_report(self, output_path: str) -> bool:
        """Save load report with timestamp"""
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
                "agent_type": "Neo4j_Agent_Direct",
                **self.load_results
            }
            
            print(f"[{get_timestamp()}] 💾 Neo4j Agent: Saving load report to {timestamped_path}")
            
            with open(timestamped_path, 'w') as f:
                json.dump(report_with_metadata, f, indent=2, default=str)
            
            print(f"[{get_timestamp()}] ✅ Neo4j Agent: Successfully saved load report with {len(self.load_results)} sections")
            return True
            
        except Exception as e:
            print(f"[{get_timestamp()}] ❌ Neo4j Agent: Failed to save report - {str(e)}")
            return False


def create_neo4j_agent(config_path: str = "config/config.yaml"):
    """Create Neo4j Agent"""
    return Neo4jAgent(config_path)