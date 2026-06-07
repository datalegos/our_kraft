#!/usr/bin/env python3
"""
Standalone Neo4j Multi-Node Graph Data Loader
Creates multiple node types with relationships for rich graph representation

USAGE:
1. Config Mode (Recommended): 
   - Set csv_settings.file_path in config.json
   - Define graph_model with nodes and relationships
   - Run: python standalone_data_loader.py
   
2. Command Line Mode:
   - Run: python standalone_data_loader.py <csv_file>
"""

import json
import pandas as pd
import sys
import os
from neo4j import GraphDatabase
from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime

class MultiNodeGraphLoader:
    """
    Advanced data loader that creates multiple node types with relationships
    Designed for rich graph representations from CSV data
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        self.driver = None
        self._setup_logging()
        self._connect_to_neo4j()
        
        # Track created nodes to avoid duplicates
        self.created_nodes = {
            'Degree': set(),
            'Branch': set(), 
            'Skill': set(),
            'PlacementStatus': set()
        }
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file '{config_file}' not found!")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Configure logging with UTF-8 encoding to handle special characters
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get('log_file', 'data_loading.log'), encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _connect_to_neo4j(self):
        """Connect to Neo4j database"""
        db_config = self.config['database']
        
        try:
            self.driver = GraphDatabase.driver(
                db_config['uri'],
                auth=(db_config['username'], db_config['password']),
                encrypted=False
            )
            
            # Test connection
            with self.driver.session(database=db_config.get('database', 'neo4j')) as session:
                session.run("RETURN 1").consume()
            
            self.logger.info(f"Connected to Neo4j at {db_config['uri']}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            sys.exit(1)
    
    def load_graph_from_config(self) -> Dict[str, Any]:
        """Load CSV and create multi-node graph based on config"""
        
        csv_config = self.config['csv_settings']
        csv_file = csv_config.get('file_path')
        
        if not csv_file:
            return {
                'status': 'error',
                'message': 'No CSV file path specified in config.json under csv_settings.file_path'
            }
        
        return self.load_multi_node_graph(csv_file)
    
    def load_multi_node_graph(self, csv_file: str) -> Dict[str, Any]:
        """
        Load CSV and create multiple node types with relationships
        
        Args:
            csv_file: Path to CSV file
        
        Returns:
            Dictionary with loading results
        """
        
        if not os.path.exists(csv_file):
            return {
                'status': 'error',
                'message': f'CSV file not found: {csv_file}'
            }
        
        try:
            self.logger.info(f"Starting multi-node graph loading from {csv_file}")
            
            # Step 1: Read and analyze CSV
            df = self._read_csv(csv_file)
            self.logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Step 1.5: Remove duplicates based on Student_ID
            original_count = len(df)
            df = df.drop_duplicates(subset=['Student_ID'], keep='first')
            duplicate_count = original_count - len(df)
            if duplicate_count > 0:
                self.logger.info(f"Removed {duplicate_count} duplicate Student_IDs, processing {len(df)} unique students")
            
            # Step 2: Create constraints and indexes
            if self.config['data_loading']['create_constraints']:
                self._create_all_constraints()
            
            if self.config['data_loading']['create_indexes']:
                self._create_all_indexes()
            
            # Step 3: Clear existing data if requested
            if self.config['data_loading']['clear_existing_data']:
                self._clear_all_data()
            
            # Step 4: Create all node types
            node_stats = self._create_all_nodes(df)
            
            # Step 5: Create relationships
            rel_stats = self._create_all_relationships(df)
            
            total_nodes = sum(node_stats.values())
            total_rels = sum(rel_stats.values())
            
            self.logger.info(f"Multi-node graph loading completed successfully")
            
            return {
                'status': 'success',
                'message': f'Successfully created multi-node graph from {csv_file}',
                'details': {
                    'total_records_processed': len(df),
                    'duplicates_removed': duplicate_count,
                    'nodes_created': node_stats,
                    'relationships_created': rel_stats,
                    'total_nodes': total_nodes,
                    'total_relationships': total_rels
                }
            }
            
        except KeyboardInterrupt:
            self.logger.info("Loading interrupted by user")
            return {
                'status': 'interrupted',
                'message': 'Loading was interrupted by user'
            }
        except Exception as e:
            self.logger.error(f"Multi-node graph loading failed: {e}")
            return {
                'status': 'error',
                'message': f'Graph loading failed: {str(e)}'
            }
    
    def _read_csv(self, csv_file: str) -> pd.DataFrame:
        """Read CSV file with proper configuration"""
        csv_config = self.config['csv_settings']
        
        df = pd.read_csv(
            csv_file,
            encoding=csv_config['encoding'],
            delimiter=csv_config['delimiter'],
            skiprows=csv_config['skip_rows']
        )
        
        return df
    
    def _create_all_constraints(self):
        """Create constraints for all node types"""
        graph_model = self.config['graph_model']
        db_name = self.config['database'].get('database', 'neo4j')
        
        for node_type, node_config in graph_model['nodes'].items():
            id_prop = node_config['id_property']
            
            constraint_query = f"""
            CREATE CONSTRAINT {node_type}_{id_prop}_unique IF NOT EXISTS 
            FOR (n:{node_type}) REQUIRE n.{id_prop} IS UNIQUE
            """
            
            try:
                with self.driver.session(database=db_name) as session:
                    session.run(constraint_query)
                self.logger.info(f"Created constraint for {node_type}.{id_prop}")
            except Exception as e:
                self.logger.warning(f"Failed to create constraint for {node_type}.{id_prop}: {e}")
    
    def _create_all_indexes(self):
        """Create indexes for better query performance"""
        graph_model = self.config['graph_model']
        db_name = self.config['database'].get('database', 'neo4j')
        
        # Create indexes for commonly queried properties
        index_configs = [
            ('Student', 'Gender'),
            ('Student', 'CGPA'),
            ('Degree', 'name'),
            ('Branch', 'name'),
            ('PlacementStatus', 'status')
        ]
        
        for node_type, prop in index_configs:
            index_query = f"""
            CREATE INDEX {node_type}_{prop}_index IF NOT EXISTS 
            FOR (n:{node_type}) ON (n.{prop})
            """
            
            try:
                with self.driver.session(database=db_name) as session:
                    session.run(index_query)
                self.logger.info(f"Created index for {node_type}.{prop}")
            except Exception as e:
                self.logger.warning(f"Failed to create index for {node_type}.{prop}: {e}")
    
    def _clear_all_data(self):
        """Clear all existing data"""
        db_name = self.config['database'].get('database', 'neo4j')
        
        clear_query = "MATCH (n) DETACH DELETE n"
        
        try:
            with self.driver.session(database=db_name) as session:
                result = session.run(clear_query)
                summary = result.consume()
                deleted_count = summary.counters.nodes_deleted
                self.logger.info(f"Cleared {deleted_count} existing nodes")
        except Exception as e:
            self.logger.warning(f"Failed to clear existing data: {e}")
    
    def _create_all_nodes(self, df: pd.DataFrame) -> Dict[str, int]:
        """Create all node types from the dataframe"""
        
        node_stats = {}
        
        # Create Student nodes
        node_stats['Student'] = self._create_student_nodes(df)
        
        # Create reference nodes (Degree, Branch, etc.)
        node_stats['Degree'] = self._create_degree_nodes(df)
        node_stats['Branch'] = self._create_branch_nodes(df)
        node_stats['Skill'] = self._create_skill_nodes(df)
        node_stats['PlacementStatus'] = self._create_placement_status_nodes(df)
        
        return node_stats
    
    def _create_student_nodes(self, df: pd.DataFrame) -> int:
        """Create Student nodes using bulk loading for better performance"""
        db_name = self.config['database'].get('database', 'neo4j')
        student_config = self.config['graph_model']['nodes']['Student']
        
        created_count = 0
        batch_size = self.config['data_loading']['batch_size']
        
        # Prepare data for bulk loading
        properties = student_config['properties']
        transformations = self.config['property_transformations']
        
        # Process in batches for memory efficiency
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            # Prepare batch parameters
            batch_params = []
            for _, row in batch.iterrows():
                params = self._prepare_student_params(row, properties)
                batch_params.append(params)
            
            # Bulk create nodes using UNWIND with MERGE to handle duplicates
            prop_string = ", ".join([f"{prop}: row.{prop}" for prop in properties])
            bulk_query = f"""
            UNWIND $batch as row
            MERGE (s:Student {{Student_ID: row.Student_ID}})
            SET s += {{
                {prop_string}
            }}
            """
            
            try:
                with self.driver.session(database=db_name) as session:
                    result = session.run(bulk_query, {'batch': batch_params})
                    summary = result.consume()
                    batch_created = summary.counters.nodes_created
                    created_count += batch_created
                    
                    self.logger.info(f"Batch {batch_num + 1}/{total_batches}: Created {batch_created} Student nodes (Total: {created_count})")
                    
            except KeyboardInterrupt:
                self.logger.info("Student node creation interrupted by user")
                raise
            except Exception as e:
                self.logger.error(f"Failed to create Student batch {batch_num + 1}: {e}")
                # For critical errors, stop the process
                if "constraint" in str(e).lower() and created_count == 0:
                    self.logger.error("Critical constraint error - stopping process")
                    raise
                # Otherwise, try fallback for this batch
                try:
                    fallback_created = self._create_student_nodes_individual(batch, properties)
                    created_count += fallback_created
                    self.logger.info(f"Fallback created {fallback_created} nodes for batch {batch_num + 1}")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed for batch {batch_num + 1}: {fallback_error}")
                    # Continue with next batch
        
        return created_count
    
    def _create_student_nodes_individual(self, batch: pd.DataFrame, properties: List[str]) -> int:
        """Fallback method for individual node creation using MERGE"""
        db_name = self.config['database'].get('database', 'neo4j')
        created_count = 0
        
        # Use MERGE to handle duplicates gracefully
        prop_assignments = ", ".join([f"s.{prop} = ${prop}" for prop in properties])
        create_query = f"""
        MERGE (s:Student {{Student_ID: $Student_ID}})
        SET {prop_assignments}
        """
        
        with self.driver.session(database=db_name) as session:
            for _, row in batch.iterrows():
                try:
                    params = self._prepare_student_params(row, properties)
                    result = session.run(create_query, params)
                    summary = result.consume()
                    if summary.counters.nodes_created > 0:
                        created_count += 1
                except KeyboardInterrupt:
                    self.logger.info("Individual node creation interrupted by user")
                    raise
                except Exception as e:
                    self.logger.warning(f"Failed to create individual Student node {row.get('Student_ID', 'unknown')}: {e}")
        
        return created_count
    
    def _prepare_student_params(self, row: pd.Series, properties: List[str]) -> Dict[str, Any]:
        """Prepare parameters for Student node creation"""
        params = {}
        transformations = self.config['property_transformations']
        
        for prop in properties:
            value = row[prop]
            
            if pd.isna(value):
                params[prop] = None
            else:
                # Apply transformations
                if prop in transformations:
                    transform_type = transformations[prop]
                    if transform_type == 'integer':
                        params[prop] = int(float(value))
                    elif transform_type == 'float':
                        params[prop] = float(value)
                    else:
                        params[prop] = str(value)
                else:
                    params[prop] = str(value)
        
        return params
    
    def _create_degree_nodes(self, df: pd.DataFrame) -> int:
        """Create unique Degree nodes"""
        unique_degrees = df['Degree'].dropna().unique()
        return self._create_reference_nodes('Degree', unique_degrees, 'name')
    
    def _create_branch_nodes(self, df: pd.DataFrame) -> int:
        """Create unique Branch nodes"""
        unique_branches = df['Branch'].dropna().unique()
        return self._create_reference_nodes('Branch', unique_branches, 'name')
    
    def _create_skill_nodes(self, df: pd.DataFrame) -> int:
        """Create Skill nodes for coding and communication"""
        skills = [
            {'name': 'Coding', 'type': 'Technical'},
            {'name': 'Communication', 'type': 'Soft'}
        ]
        
        db_name = self.config['database'].get('database', 'neo4j')
        created_count = 0
        
        with self.driver.session(database=db_name) as session:
            for skill in skills:
                try:
                    query = "CREATE (s:Skill {name: $name, type: $type})"
                    session.run(query, skill)
                    created_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to create Skill node {skill['name']}: {e}")
        
        return created_count
    
    def _create_placement_status_nodes(self, df: pd.DataFrame) -> int:
        """Create PlacementStatus nodes"""
        unique_statuses = df['Placement_Status'].dropna().unique()
        return self._create_reference_nodes('PlacementStatus', unique_statuses, 'status')
    
    def _create_reference_nodes(self, node_type: str, values: List[str], prop_name: str) -> int:
        """Create reference nodes (Degree, Branch, PlacementStatus)"""
        db_name = self.config['database'].get('database', 'neo4j')
        created_count = 0
        
        with self.driver.session(database=db_name) as session:
            for value in values:
                if value not in self.created_nodes[node_type]:
                    try:
                        query = f"CREATE (n:{node_type} {{{prop_name}: ${prop_name}}})"
                        session.run(query, {prop_name: str(value)})
                        self.created_nodes[node_type].add(value)
                        created_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to create {node_type} node {value}: {e}")
        
        return created_count
    
    def _create_all_relationships(self, df: pd.DataFrame) -> Dict[str, int]:
        """Create all relationships between nodes"""
        
        rel_stats = {}
        relationships = self.config['graph_model']['relationships']
        
        for rel_config in relationships:
            rel_type = rel_config['type']
            rel_stats[rel_type] = self._create_relationship_batch(df, rel_config)
        
        return rel_stats
    
    def _create_relationship_batch(self, df: pd.DataFrame, rel_config: Dict[str, Any]) -> int:
        """Create a specific type of relationship using bulk operations"""
        db_name = self.config['database'].get('database', 'neo4j')
        created_count = 0
        
        rel_type = rel_config['type']
        condition_col = rel_config['condition']
        batch_size = self.config['data_loading']['batch_size']
        
        # Filter out rows with null values for the condition
        valid_df = df[df[condition_col].notna()].copy()
        
        if len(valid_df) == 0:
            return 0
        
        # Process in batches
        total_batches = (len(valid_df) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(valid_df))
            batch = valid_df.iloc[start_idx:end_idx]
            
            try:
                batch_created = self._create_relationship_bulk(batch, rel_config)
                created_count += batch_created
                
                self.logger.info(f"Relationship {rel_type} - Batch {batch_num + 1}/{total_batches}: Created {batch_created} relationships")
                
            except Exception as e:
                self.logger.error(f"Failed to create {rel_type} relationships in batch {batch_num + 1}: {e}")
                # Fallback to individual creation
                batch_created = self._create_relationship_individual(batch, rel_config)
                created_count += batch_created
        
        return created_count
    
    def _create_relationship_bulk(self, batch: pd.DataFrame, rel_config: Dict[str, Any]) -> int:
        """Create relationships in bulk using UNWIND"""
        db_name = self.config['database'].get('database', 'neo4j')
        rel_type = rel_config['type']
        condition_col = rel_config['condition']
        
        # Prepare batch data
        batch_data = []
        for _, row in batch.iterrows():
            condition_value = row[condition_col]
            if not pd.isna(condition_value):
                batch_data.append({
                    'student_id': int(row['Student_ID']),
                    'condition_value': condition_value
                })
        
        if not batch_data:
            return 0
        
        # Build bulk query based on relationship type
        if rel_type == "PURSUING":
            query = """
            UNWIND $batch as row
            MATCH (s:Student {Student_ID: row.student_id})
            MATCH (d:Degree {name: row.condition_value})
            CREATE (s)-[:PURSUING]->(d)
            """
            
        elif rel_type == "SPECIALIZES_IN":
            query = """
            UNWIND $batch as row
            MATCH (s:Student {Student_ID: row.student_id})
            MATCH (b:Branch {name: row.condition_value})
            CREATE (s)-[:SPECIALIZES_IN]->(b)
            """
            
        elif rel_type == "HAS_CODING_SKILL":
            # Convert condition_value to rating for batch data
            for item in batch_data:
                item['rating'] = int(item['condition_value'])
            
            query = """
            UNWIND $batch as row
            MATCH (s:Student {Student_ID: row.student_id})
            MATCH (sk:Skill {name: 'Coding'})
            CREATE (s)-[:HAS_CODING_SKILL {rating: row.rating}]->(sk)
            """
            
        elif rel_type == "HAS_COMMUNICATION_SKILL":
            # Convert condition_value to rating for batch data
            for item in batch_data:
                item['rating'] = int(item['condition_value'])
            
            query = """
            UNWIND $batch as row
            MATCH (s:Student {Student_ID: row.student_id})
            MATCH (sk:Skill {name: 'Communication'})
            CREATE (s)-[:HAS_COMMUNICATION_SKILL {rating: row.rating}]->(sk)
            """
            
        elif rel_type == "HAS_STATUS":
            query = """
            UNWIND $batch as row
            MATCH (s:Student {Student_ID: row.student_id})
            MATCH (ps:PlacementStatus {status: row.condition_value})
            CREATE (s)-[:HAS_STATUS]->(ps)
            """
        else:
            return 0
        
        # Execute bulk query
        with self.driver.session(database=db_name) as session:
            result = session.run(query, {'batch': batch_data})
            summary = result.consume()
            return summary.counters.relationships_created
    
    def _create_relationship_individual(self, batch: pd.DataFrame, rel_config: Dict[str, Any]) -> int:
        """Fallback method for individual relationship creation"""
        db_name = self.config['database'].get('database', 'neo4j')
        created_count = 0
        
        with self.driver.session(database=db_name) as session:
            for _, row in batch.iterrows():
                try:
                    created = self._create_single_relationship(session, row, rel_config)
                    if created:
                        created_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to create individual {rel_config['type']} relationship: {e}")
        
        return created_count
    
    def _create_single_relationship(self, session, row: pd.Series, rel_config: Dict[str, Any]) -> bool:
        """Create a single relationship"""
        
        from_node = rel_config['from']
        to_node = rel_config['to']
        rel_type = rel_config['type']
        condition_col = rel_config['condition']
        
        # Get the condition value
        condition_value = row[condition_col]
        if pd.isna(condition_value):
            return False
        
        # Build the query based on relationship type
        if rel_type == "PURSUING":
            query = """
            MATCH (s:Student {Student_ID: $student_id})
            MATCH (d:Degree {name: $degree_name})
            CREATE (s)-[:PURSUING]->(d)
            """
            params = {
                'student_id': int(row['Student_ID']),
                'degree_name': str(condition_value)
            }
            
        elif rel_type == "SPECIALIZES_IN":
            query = """
            MATCH (s:Student {Student_ID: $student_id})
            MATCH (b:Branch {name: $branch_name})
            CREATE (s)-[:SPECIALIZES_IN]->(b)
            """
            params = {
                'student_id': int(row['Student_ID']),
                'branch_name': str(condition_value)
            }
            
        elif rel_type == "HAS_CODING_SKILL":
            query = """
            MATCH (s:Student {Student_ID: $student_id})
            MATCH (sk:Skill {name: 'Coding'})
            CREATE (s)-[:HAS_CODING_SKILL {rating: $rating}]->(sk)
            """
            params = {
                'student_id': int(row['Student_ID']),
                'rating': int(condition_value)
            }
            
        elif rel_type == "HAS_COMMUNICATION_SKILL":
            query = """
            MATCH (s:Student {Student_ID: $student_id})
            MATCH (sk:Skill {name: 'Communication'})
            CREATE (s)-[:HAS_COMMUNICATION_SKILL {rating: $rating}]->(sk)
            """
            params = {
                'student_id': int(row['Student_ID']),
                'rating': int(condition_value)
            }
            
        elif rel_type == "HAS_STATUS":
            query = """
            MATCH (s:Student {Student_ID: $student_id})
            MATCH (ps:PlacementStatus {status: $status})
            CREATE (s)-[:HAS_STATUS]->(ps)
            """
            params = {
                'student_id': int(row['Student_ID']),
                'status': str(condition_value)
            }
        else:
            return False
        
        # Execute the query
        session.run(query, params)
        return True
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        db_name = self.config['database'].get('database', 'neo4j')
        
        try:
            with self.driver.session(database=db_name) as session:
                # Node counts by label
                node_stats = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(n) as count
                    ORDER BY count DESC
                """).data()
                
                # Relationship counts by type
                rel_stats = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as relationship, count(r) as count
                    ORDER BY count DESC
                """).data()
                
                # Total counts
                total_nodes = session.run("MATCH (n) RETURN count(n) as total").single()['total']
                total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as total").single()['total']
                
                # Sample queries for verification
                sample_data = {
                    'placed_students': session.run("""
                        MATCH (s:Student)-[:HAS_STATUS]->(ps:PlacementStatus {status: 'Placed'})
                        RETURN count(s) as count
                    """).single()['count'],
                    
                    'degrees_offered': session.run("""
                        MATCH (d:Degree)
                        RETURN collect(d.name) as degrees
                    """).single()['degrees'],
                    
                    'branches_available': session.run("""
                        MATCH (b:Branch)
                        RETURN collect(b.name) as branches
                    """).single()['branches']
                }
                
                return {
                    'total_nodes': total_nodes,
                    'total_relationships': total_rels,
                    'nodes_by_label': node_stats,
                    'relationships_by_type': rel_stats,
                    'sample_insights': sample_data
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get graph stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Database connection closed")

def main():
    """Main function for command line usage"""
    
    # Initialize loader
    loader = MultiNodeGraphLoader()
    
    try:
        print(f"Starting multi-node graph creation...")
        
        # Check if CSV file is provided via command line
        if len(sys.argv) >= 2:
            # Command line mode
            csv_file = sys.argv[1]
            print(f"CSV File: {csv_file} (from command line)")
            print("-" * 60)
            
            result = loader.load_multi_node_graph(csv_file)
        else:
            # Config mode - read from config file
            csv_config = loader.config['csv_settings']
            csv_file = csv_config.get('file_path', 'Not specified')
            
            print(f"CSV File: {csv_file} (from config)")
            print("-" * 60)
            
            result = loader.load_graph_from_config()
        
        if result['status'] == 'success':
            print(f"\n{result['message']}")
            print(f"Details:")
            details = result['details']
            print(f"   • Records Processed: {details['total_records_processed']:,}")
            if details.get('duplicates_removed', 0) > 0:
                print(f"   • Duplicates Removed: {details['duplicates_removed']:,}")
            print(f"   • Total Nodes Created: {details['total_nodes']:,}")
            print(f"   • Total Relationships Created: {details['total_relationships']:,}")
            
            print(f"\nNode Breakdown:")
            for node_type, count in details['nodes_created'].items():
                print(f"   • {node_type}: {count:,}")
            
            print(f"\nRelationship Breakdown:")
            for rel_type, count in details['relationships_created'].items():
                print(f"   • {rel_type}: {count:,}")
            
            # Show comprehensive graph stats
            print(f"\nCurrent Graph Statistics:")
            stats = loader.get_graph_stats()
            
            if stats.get('sample_insights'):
                insights = stats['sample_insights']
                print(f"   • Placed Students: {insights['placed_students']:,}")
                print(f"   • Degrees: {', '.join(insights['degrees'])}")
                print(f"   • Branches: {', '.join(insights['branches'])}")
        elif result['status'] == 'interrupted':
            print(f"\n{result['message']}")
            print("Process was stopped by user (Ctrl+C)")
        else:
            print(f"\n{result['message']}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\nLoading interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    finally:
        loader.close()

if __name__ == "__main__":
    main()