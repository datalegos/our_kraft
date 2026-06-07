import pandas as pd
import json
from typing import Dict, Any, List
from .base_service import BaseService
from .llm_service import LLMService

class AgenticDataLoader(BaseService):
    """
    AI-powered data loader that can analyze and load any CSV dataset
    Uses LLM to understand data structure and generate appropriate loading strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_service = LLMService(config)
    
    def process(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Process data loading request"""
        
        # Check if CSV file path is provided
        csv_path = kwargs.get('csv_path') or self._extract_file_path(user_input)
        
        if not csv_path:
            return {
                'status': 'error',
                'message': 'Please specify the CSV file path to load',
                'suggestion': 'Example: "Load data from students_2024.csv"'
            }
        
        try:
            # Step 1: Analyze the CSV structure
            print("🔍 AI is analyzing the CSV structure...")
            analysis = self._analyze_csv_structure(csv_path)
            
            # Step 2: Generate loading strategy
            print("🧠 AI is creating loading strategy...")
            strategy = self._generate_loading_strategy(analysis)
            
            # Step 3: Execute loading
            print("⚡ AI is loading data...")
            result = self._execute_loading_strategy(csv_path, strategy)
            
            # Step 4: Update schema cache
            self._invalidate_schema_cache()
            
            return result
            
        except FileNotFoundError:
            return {
                'status': 'error',
                'message': f'CSV file not found: {csv_path}',
                'suggestion': 'Please check the file path and try again'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Data loading failed: {str(e)}'
            }
    
    def _extract_file_path(self, user_input: str) -> str:
        """Extract file path from user input using AI"""
        try:
            prompt = f"""Extract the file path from this user request about loading data:
            
User Input: "{user_input}"

If a specific file is mentioned, return just the filename/path.
If no specific file is mentioned, return "Dataset/train.csv" as default.
Return only the file path, nothing else.

Examples:
"Load data from students_2024.csv" → students_2024.csv
"Import the new dataset" → Dataset/train.csv
"Load employee_data.csv" → employee_data.csv
"""
            
            response = self.llm_service.generate(prompt, max_tokens=50)
            return response.strip().strip('"\'')
            
        except:
            return "Dataset/train.csv"  # Default fallback
    
    def _analyze_csv_structure(self, csv_path: str) -> Dict[str, Any]:
        """Use AI to analyze CSV structure and determine data types"""
        
        # Read sample of CSV
        df = pd.read_csv(csv_path, nrows=100)  # Sample first 100 rows
        
        # Get basic info
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        sample_data = df.head(5).to_dict('records')
        
        # Use AI to analyze the structure
        analysis_prompt = f"""Analyze this CSV dataset structure and determine the best way to model it in Neo4j.

Columns: {columns}
Data Types: {dtypes}
Sample Data: {sample_data[:3]}

Analyze and provide:
1. What type of entities (nodes) should be created?
2. What relationships exist between entities?
3. Which columns should be properties vs separate nodes?
4. What data transformations are needed?

Respond in JSON format:
{{
    "primary_entity": "Student",
    "primary_id_column": "student_id",
    "node_types": [
        {{"name": "Student", "properties": ["id", "name", "age"], "id_property": "id"}},
        {{"name": "Course", "properties": ["name", "code"], "id_property": "code"}}
    ],
    "relationships": [
        {{"from": "Student", "to": "Course", "type": "ENROLLED_IN", "condition": "course_id"}}
    ],
    "data_transformations": [
        {{"column": "age", "type": "integer"}},
        {{"column": "gpa", "type": "float"}}
    ]
}}
"""
        
        try:
            response = self.llm_service.generate(analysis_prompt)
            
            # Parse JSON response
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                # Fallback analysis
                return self._fallback_analysis(columns, dtypes)
                
        except Exception as e:
            print(f"AI analysis failed, using fallback: {e}")
            return self._fallback_analysis(columns, dtypes)
    
    def _fallback_analysis(self, columns: List[str], dtypes: Dict) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        
        # Simple heuristics
        id_column = next((col for col in columns if 'id' in col.lower()), columns[0])
        
        return {
            "primary_entity": "Record",
            "primary_id_column": id_column,
            "node_types": [
                {
                    "name": "Record",
                    "properties": columns,
                    "id_property": id_column
                }
            ],
            "relationships": [],
            "data_transformations": [
                {"column": col, "type": "string" if str(dtype) == 'object' else str(dtype)}
                for col, dtype in dtypes.items()
            ]
        }
    
    def _generate_loading_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Cypher queries for loading based on analysis"""
        
        strategy_prompt = f"""Based on this data analysis, generate Cypher queries to load the data into Neo4j.

Analysis: {json.dumps(analysis, indent=2)}

Generate a loading strategy with:
1. Constraint creation queries
2. Node creation queries  
3. Relationship creation queries
4. Index creation queries

Respond in JSON format:
{{
    "constraints": [
        "CREATE CONSTRAINT student_id IF NOT EXISTS FOR (s:Student) REQUIRE s.student_id IS UNIQUE"
    ],
    "node_creation": [
        "CREATE (s:Student {{student_id: $student_id, name: $name, age: $age}})"
    ],
    "relationships": [
        "MATCH (s:Student {{student_id: $student_id}}) MATCH (c:Course {{code: $course_code}}) CREATE (s)-[:ENROLLED_IN]->(c)"
    ],
    "indexes": [
        "CREATE INDEX student_name IF NOT EXISTS FOR (s:Student) ON (s.name)"
    ]
}}
"""
        
        try:
            response = self.llm_service.generate(strategy_prompt)
            
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                return self._fallback_strategy(analysis)
                
        except Exception as e:
            print(f"Strategy generation failed, using fallback: {e}")
            return self._fallback_strategy(analysis)
    
    def _fallback_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy when AI fails"""
        
        primary_entity = analysis['primary_entity']
        properties = analysis['node_types'][0]['properties']
        id_prop = analysis['primary_id_column']
        
        # Create property string for Cypher
        prop_string = ", ".join([f"{prop}: ${prop}" for prop in properties])
        
        return {
            "constraints": [
                f"CREATE CONSTRAINT {id_prop}_unique IF NOT EXISTS FOR (n:{primary_entity}) REQUIRE n.{id_prop} IS UNIQUE"
            ],
            "node_creation": [
                f"CREATE (n:{primary_entity} {{{prop_string}}})"
            ],
            "relationships": [],
            "indexes": []
        }
    
    def _execute_loading_strategy(self, csv_path: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the loading strategy"""
        
        try:
            # Read the full CSV
            df = pd.read_csv(csv_path)
            
            # Step 1: Create constraints
            for constraint in strategy.get('constraints', []):
                try:
                    self.execute_query(constraint)
                    print(f"✓ Created constraint")
                except Exception as e:
                    print(f"⚠️ Constraint creation failed: {e}")
            
            # Step 2: Create indexes
            for index in strategy.get('indexes', []):
                try:
                    self.execute_query(index)
                    print(f"✓ Created index")
                except Exception as e:
                    print(f"⚠️ Index creation failed: {e}")
            
            # Step 3: Load data in batches
            batch_size = 1000
            total_created = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                # Create nodes for this batch
                for node_query in strategy.get('node_creation', []):
                    for _, row in batch.iterrows():
                        try:
                            # Convert row to parameters
                            params = row.to_dict()
                            # Handle NaN values
                            params = {k: (v if pd.notna(v) else None) for k, v in params.items()}
                            
                            self.execute_query(node_query, params)
                            total_created += 1
                            
                        except Exception as e:
                            print(f"⚠️ Failed to create node: {e}")
                
                print(f"Processed batch {i//batch_size + 1}, total created: {total_created}")
            
            # Step 4: Create relationships (if any)
            for rel_query in strategy.get('relationships', []):
                try:
                    # This would need more sophisticated handling
                    # For now, skip relationships in fallback
                    pass
                except Exception as e:
                    print(f"⚠️ Relationship creation failed: {e}")
            
            return {
                'status': 'success',
                'message': f'Successfully loaded {total_created} records from {csv_path}',
                'details': {
                    'total_records': len(df),
                    'created_nodes': total_created,
                    'constraints_created': len(strategy.get('constraints', [])),
                    'indexes_created': len(strategy.get('indexes', []))
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Loading execution failed: {str(e)}'
            }
    
    def _invalidate_schema_cache(self):
        """Invalidate schema cache so it gets refreshed"""
        # This would need to be implemented in ai_query_service
        pass
    
    def analyze_dataset(self, csv_path: str) -> Dict[str, Any]:
        """Analyze a dataset without loading it"""
        try:
            analysis = self._analyze_csv_structure(csv_path)
            strategy = self._generate_loading_strategy(analysis)
            
            return {
                'status': 'success',
                'analysis': analysis,
                'strategy': strategy,
                'message': f'Dataset analysis complete for {csv_path}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            }