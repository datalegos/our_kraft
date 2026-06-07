import json
import re
from typing import Dict, Any, List, Optional
from .base_service import BaseService
from .llm_service import LLMService

class AIQueryService(BaseService):
    """AI-powered query service using LLMs for dynamic Cypher generation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_service = LLMService(config)
        self.conversation_history = []
        self.schema_cache = None
        self.agent_config = config.get('agent', {})
    
    def get_schema_info(self) -> str:
        """Get comprehensive schema information for LLM context"""
        if self.schema_cache:
            return self.schema_cache
        
        try:
            # Get node labels and sample properties
            node_info = self.execute_query("""
                MATCH (n)
                WITH labels(n)[0] as label, n
                WHERE label IS NOT NULL
                WITH label, collect(keys(n))[0] as sample_props, count(n) as count
                RETURN label, sample_props, count
                ORDER BY count DESC
            """)
            
            # Get relationship types
            rel_info = self.execute_query("""
                MATCH ()-[r]->()
                WITH type(r) as rel_type, count(r) as count
                RETURN rel_type, count
                ORDER BY count DESC
            """)
            
            # Build schema description
            schema_parts = ["Neo4j Database Schema:\n"]
            
            schema_parts.append("Node Types:")
            for node in node_info:
                props = ", ".join(node['sample_props']) if node['sample_props'] else "no properties"
                schema_parts.append(f"  - {node['label']} ({node['count']} nodes): {props}")
            
            schema_parts.append("\nRelationship Types:")
            for rel in rel_info:
                schema_parts.append(f"  - {rel['rel_type']} ({rel['count']} relationships)")
            
            schema_parts.append("""
IMPORTANT - Actual Student Properties:
- student_id (integer): Unique student identifier
- age (integer): Student age
- gender (string): 'Male' or 'Female'
- cgpa (float): Academic performance (0.0-10.0)
- communication_skills (integer): Communication ability (1-10)
- soft_skills_rating (integer): Soft skills rating (1-10)
- aptitude_test_score (integer): Test score (0-100)
- backlogs (integer): Number of academic backlogs
- placement_status (string): 'Placed' or 'Not Placed'

NOTE: There is NO 'coding_skills' property. Use Technology relationships instead.

Common Query Patterns:
- MATCH (s:Student) WHERE s.property = value
- MATCH (s:Student)-[:SPECIALIZES_IN]->(b:Branch)
- MATCH (s:Student)-[k:KNOWS]->(t:Technology) WHERE t.name = 'Python'
- Use s.placement_status = 'Placed' or 'Not Placed'
- For coding skills, use: MATCH (s:Student)-[:KNOWS]->(t:Technology)
""")
            
            self.schema_cache = "\n".join(schema_parts)
            return self.schema_cache
            
        except Exception as e:
            return f"Schema information unavailable: {str(e)}"
    
    def process(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Process user query using AI"""
        
        try:
            # Add to conversation history
            self.conversation_history.append(f"User: {user_input}")
            
            # Keep conversation history manageable
            max_history = self.agent_config.get('max_conversation_history', 10)
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]
            
            # Get schema information
            schema_info = self.get_schema_info()
            
            # Generate Cypher query using LLM
            llm_response = self.llm_service.generate_cypher(
                user_input, 
                schema_info, 
                self.conversation_history
            )
            
            cypher_query = llm_response.get('cypher', '').strip()
            explanation = llm_response.get('explanation', '')
            confidence = llm_response.get('confidence', 0.5)
            
            if not cypher_query:
                return {
                    'status': 'error',
                    'message': 'Could not generate a valid query from your request.',
                    'suggestion': 'Please try rephrasing your question or be more specific.'
                }
            
            # Execute the generated query with retry logic
            max_attempts = self.agent_config.get('max_query_attempts', 3)
            results = None
            final_query = cypher_query
            
            for attempt in range(max_attempts):
                try:
                    results = self.execute_query(final_query)
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"Query attempt {attempt + 1} failed: {error_msg}")
                    
                    if attempt < max_attempts - 1:
                        # Try to get correction from LLM
                        try:
                            correction = self.llm_service.suggest_corrections(
                                final_query, error_msg, schema_info
                            )
                            # Extract corrected query
                            corrected_query = self._extract_cypher_from_text(correction)
                            if corrected_query:
                                final_query = corrected_query
                                print(f"Attempting correction: {corrected_query}")
                            else:
                                break
                        except:
                            break
                    else:
                        return {
                            'status': 'error',
                            'message': f'Query execution failed: {error_msg}',
                            'generated_query': cypher_query,
                            'explanation': explanation
                        }
            
            if results is None:
                return {
                    'status': 'error',
                    'message': 'Failed to execute query after multiple attempts'
                }
            
            # Add successful query to conversation history
            self.conversation_history.append(f"Assistant: Executed query successfully, found {len(results)} results")
            
            # Generate insights using LLM
            insights = ""
            if self.agent_config.get('enable_suggestions', True) and results:
                try:
                    insights = self.llm_service.analyze_results(
                        final_query, results, user_input
                    )
                except Exception as e:
                    print(f"Failed to generate insights: {e}")
            
            return {
                'status': 'success',
                'results': results,
                'count': len(results),
                'query': final_query,
                'explanation': explanation,
                'confidence': confidence,
                'insights': insights,
                'formatted_response': self._format_ai_response(
                    user_input, results, explanation, insights
                )
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'AI processing failed: {str(e)}',
                'suggestion': 'Please try a simpler question or check your query.'
            }
    
    def _extract_cypher_from_text(self, text: str) -> Optional[str]:
        """Extract Cypher query from text response"""
        import re
        
        # First try to extract from JSON
        try:
            import json
            if text.strip().startswith('```json'):
                # Extract JSON from markdown
                json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    parsed = json.loads(json_str)
                    return parsed.get('cypher', '').strip()
            elif text.strip().startswith('{'):
                # Direct JSON
                parsed = json.loads(text)
                return parsed.get('cypher', '').strip()
        except:
            pass
        
        # Look for MATCH, CREATE, MERGE patterns
        patterns = [
            r'```cypher\s*(.*?)\s*```',
            r'```\s*(MATCH.*?)\s*```',
            r'"cypher":\s*"([^"]*)"',
            r'(MATCH.*?)(?:\n|$)',
            r'(CREATE.*?)(?:\n|$)',
            r'(MERGE.*?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                cypher = match.group(1).strip()
                # Clean up escaped characters
                cypher = cypher.replace('\\n', '\n').replace('\\"', '"')
                return cypher
        
        return None
    
    def _format_ai_response(self, user_query: str, results: List[Dict], 
                           explanation: str, insights: str) -> str:
        """Format AI response for display"""
        
        response_parts = []
        
        # Add explanation if enabled
        if self.agent_config.get('enable_query_explanation', True) and explanation:
            response_parts.append(f"🧠 **Understanding:** {explanation}")
        
        # Add results summary
        if results:
            response_parts.append(f"📊 **Found {len(results)} results**")
            
            # Show sample results
            if len(results) <= 5:
                response_parts.append("\n**Results:**")
                for i, result in enumerate(results, 1):
                    formatted_result = self._format_result_item(result)
                    response_parts.append(f"   {i}. {formatted_result}")
            else:
                response_parts.append(f"\n**Sample Results (showing first 5 of {len(results)}):**")
                for i, result in enumerate(results[:5], 1):
                    formatted_result = self._format_result_item(result)
                    response_parts.append(f"   {i}. {formatted_result}")
        else:
            response_parts.append("📭 **No results found**")
        
        # Add insights
        if insights:
            response_parts.append(f"\n💡 **Insights:**\n{insights}")
        
        return "\n".join(response_parts)
    
    def _format_result_item(self, result: Dict) -> str:
        """Format individual result item"""
        # Handle different result types
        if 'student_id' in result or 's.student_id' in result:
            # Student result
            student_id = result.get('student_id') or result.get('s.student_id')
            cgpa = result.get('cgpa') or result.get('s.cgpa')
            status = result.get('placement_status') or result.get('s.placement_status')
            
            status_emoji = "✅" if status == 'Placed' else "❌"
            return f"ID:{student_id} | CGPA:{cgpa} | {status_emoji}"
        
        elif 'count' in result or any('count' in str(k) for k in result.keys()):
            # Count result
            count_key = next((k for k in result.keys() if 'count' in str(k)), 'count')
            return f"Count: {result[count_key]}"
        
        elif 'branch' in result:
            # Branch result
            return f"Branch: {result['branch']}"
        
        else:
            # Generic result
            items = [f"{k}:{v}" for k, v in result.items() if v is not None]
            return " | ".join(items[:3])  # Limit to 3 items
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🧹 Conversation history cleared")