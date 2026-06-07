from typing import Dict, Any, List
from .base_service import BaseService

class SchemaService(BaseService):
    """Service for schema operations and database structure info"""
    
    def process(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Process schema-related requests"""
        
        try:
            schema_info = self._get_schema_info()
            return {
                'status': 'success',
                'schema': schema_info,
                'formatted_response': self._format_schema_response(schema_info)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to retrieve schema: {str(e)}'
            }
    
    def _get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information"""
        
        # Get node labels and counts
        node_info = self.execute_query("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
        """)
        
        # Get relationship types and counts
        relationship_info = self.execute_query("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship, count(r) as count
            ORDER BY count DESC
        """)
        
        # Get sample properties for each node type
        node_properties = {}
        for node in node_info:
            label = node['label']
            if label:  # Skip nodes without labels
                props = self.execute_query(f"""
                    MATCH (n:{label})
                    WITH keys(n) as props
                    UNWIND props as prop
                    RETURN DISTINCT prop
                    ORDER BY prop
                    LIMIT 20
                """)
                node_properties[label] = [p['prop'] for p in props]
        
        # Get database statistics
        db_stats = self.execute_query("""
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as total_nodes, count(r) as total_relationships
        """)[0]
        
        return {
            'nodes': node_info,
            'relationships': relationship_info,
            'node_properties': node_properties,
            'statistics': db_stats
        }
    
    def _format_schema_response(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information into readable response"""
        
        response = "🗄️ **Neo4j Database Schema**\n\n"
        
        # Database statistics
        stats = schema_info['statistics']
        response += f"📊 **Statistics:**\n"
        response += f"   • Total Nodes: {stats['total_nodes']:,}\n"
        response += f"   • Total Relationships: {stats['total_relationships']:,}\n\n"
        
        # Node types
        response += "🏷️ **Node Types:**\n"
        for node in schema_info['nodes']:
            if node['label']:
                response += f"   • {node['label']}: {node['count']:,} nodes\n"
        response += "\n"
        
        # Relationship types
        response += "🔗 **Relationship Types:**\n"
        for rel in schema_info['relationships']:
            response += f"   • {rel['relationship']}: {rel['count']:,} relationships\n"
        response += "\n"
        
        # Node properties
        response += "🔧 **Node Properties:**\n"
        for label, properties in schema_info['node_properties'].items():
            response += f"   • {label}: {', '.join(properties)}\n"
        
        return response
    
    def get_sample_data(self, node_type: str = None, limit: int = 5) -> Dict[str, Any]:
        """Get sample data from the database"""
        try:
            if node_type:
                query = f"""
                    MATCH (n:{node_type})
                    RETURN n
                    LIMIT {limit}
                """
            else:
                query = f"""
                    MATCH (n)
                    RETURN labels(n)[0] as type, n
                    LIMIT {limit}
                """
            
            results = self.execute_query(query)
            return {
                'status': 'success',
                'samples': results,
                'count': len(results)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get sample data: {str(e)}'
            }