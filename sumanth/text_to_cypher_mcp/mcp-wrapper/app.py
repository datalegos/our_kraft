from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import os
import yaml
import openai
import re
import json
from string import Template

app = Flask(__name__)
CORS(app)

class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load and process YAML configuration with environment variable substitution"""
        try:
            with open(self.config_path, 'r') as file:
                config_content = file.read()
                
            # Substitute environment variables
            template = Template(config_content)
            config_content = template.safe_substitute(os.environ)
            
            config = yaml.safe_load(config_content)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {str(e)}")
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'mcp.url')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

# Initialize configuration
config = ConfigManager()

# Configure logging
log_level = getattr(logging, config.get('service.log_level', 'INFO').upper())
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

class SecureTextToCypherClient:
    def __init__(self, config_manager):
        self.config = config_manager
        
        # MCP server configuration
        self.mcp_url = self.config.get('mcp.url')
        self.timeout = self.config.get('mcp.timeout', 30)
        
        if not self.mcp_url:
            raise ValueError("MCP server URL not configured in config.yaml")
        
        # OpenAI configuration
        self.openai_api_key = self.config.get('openai.api_key')
        self.openai_model = self.config.get('openai.model', 'gpt-3.5-turbo')
        self.openai_max_tokens = self.config.get('openai.max_tokens', 200)
        self.openai_temperature = self.config.get('openai.temperature', 0.1)
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client properly (v0.28.1 format)
        openai.api_key = self.openai_api_key
        
        # Security configuration
        self.max_query_length = self.config.get('security.max_query_length', 1000)
        self.allowed_operations = self.config.get('security.allowed_operations', [])
        self.blocked_operations = self.config.get('security.blocked_operations', [])
        self.blocked_keywords = self.config.get('security.blocked_keywords', [])
        
        logger.info(f"Initialized SecureTextToCypherClient with MCP: {self.mcp_url}")
        
    def call_mcp_tool(self, tool_name, arguments=None):
        """Call MCP tool with our secure credentials"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {}
                }
            }
            
            response = requests.post(
                self.mcp_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Debug: Log the raw response
            logger.debug(f"Raw MCP response: {response.text}")
            
            # Parse Server-Sent Events format
            response_text = response.text.strip()
            
            if 'event: message' in response_text and 'data: ' in response_text:
                # Extract JSON from SSE format - handle both \n and \r\n
                lines = response_text.replace('\r\n', '\n').split('\n')
                for line in lines:
                    if line.startswith('data: '):
                        json_data = line[6:]  # Remove 'data: ' prefix
                        result = json.loads(json_data)
                        break
                else:
                    raise ValueError("No data line found in SSE response")
            elif response_text.startswith('data: '):
                # Sometimes it's just data: without event:
                json_data = response_text.split('data: ', 1)[1]
                result = json.loads(json_data)
            else:
                # Try regular JSON parsing
                try:
                    result = response.json()
                except:
                    logger.error(f"Could not parse response: {response_text}")
                    raise ValueError(f"Invalid response format: {response_text}")
            
            if "result" in result:
                return result["result"]
            else:
                raise ValueError(f"Invalid MCP response: {result}")
                
        except Exception as e:
            logger.error(f"MCP tool call failed ({tool_name}): {str(e)}")
            raise
    
    def get_schema(self):
        """Get Neo4j schema via secure MCP connection"""
        try:
            logger.info("Getting Neo4j schema via MCP")
            result = self.call_mcp_tool("get_neo4j_schema")
            
            # Handle the MCP response format
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from the content array
                    schema_text = content[0].get("text", "")
                    if schema_text:
                        return schema_text
                elif isinstance(content, str):
                    return content
            
            raise ValueError("No schema returned from MCP server")
            
        except Exception as e:
            logger.error(f"Failed to get schema: {str(e)}")
            raise
    
    def validate_cypher_security(self, cypher_query):
        """Comprehensive security validation for Cypher queries"""
        if not cypher_query or not cypher_query.strip():
            raise ValueError("Empty query not allowed")
        
        cypher_upper = cypher_query.upper()
        
        # Check query length
        if len(cypher_query) > self.max_query_length:
            raise ValueError(f"Query exceeds maximum length of {self.max_query_length} characters")
        
        # Block dangerous operations
        for blocked_op in self.blocked_operations:
            if blocked_op.upper() in cypher_upper:
                raise ValueError(f"Operation '{blocked_op}' is not allowed for security reasons")
        
        # Block dangerous keywords
        for keyword in self.blocked_keywords:
            if keyword.upper() in cypher_upper:
                raise ValueError(f"Keyword '{keyword}' is not allowed for security reasons")
        
        # Ensure query contains allowed operations
        if not any(allowed_op.upper() in cypher_upper for allowed_op in self.allowed_operations):
            raise ValueError(f"Query must contain at least one allowed operation: {', '.join(self.allowed_operations)}")
        
        # Block multiple statements
        if cypher_query.count(';') > 1:
            raise ValueError("Multiple statements not allowed")
        
        # Remove trailing semicolon for single statement
        cypher_query = cypher_query.rstrip(';').strip()
        
        return cypher_query
    
    def ai_text_to_cypher(self, question, schema):
        """Convert text to Cypher using OpenAI with no database access"""
        try:
            logger.info(f"Converting text to Cypher using AI: {question}")
            
            prompt = f"""You are a Neo4j Cypher query generator. Convert natural language questions to Cypher queries.

Database Schema:
{schema}

Security Rules:
1. Generate ONLY read-only queries using: {', '.join(self.allowed_operations)}
2. NEVER use: {', '.join(self.blocked_operations)}
3. NEVER use admin commands or system functions
4. Always include LIMIT clause to prevent large result sets
5. Return ONLY the Cypher query, no explanation or formatting

Question: "{question}"

Cypher Query:"""

            # Use the older OpenAI client format (v0.28.1)
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.openai_max_tokens,
                temperature=self.openai_temperature
            )
            
            cypher_query = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown formatting)
            cypher_query = re.sub(r'```cypher\n?', '', cypher_query)
            cypher_query = re.sub(r'```\n?', '', cypher_query)
            cypher_query = cypher_query.strip()
            
            logger.info(f"AI generated Cypher: {cypher_query}")
            return cypher_query
            
        except Exception as e:
            logger.error(f"AI text-to-cypher conversion failed: {str(e)}")
            raise
    
    def secure_text_to_cypher(self, question):
        """Complete secure text-to-cypher pipeline"""
        try:
            # Step 1: Get schema via secure MCP connection
            schema = self.get_schema()
            
            # Step 2: Generate Cypher using AI (no database access)
            cypher_query = self.ai_text_to_cypher(question, schema)
            
            # Step 3: Security validation
            validated_query = self.validate_cypher_security(cypher_query)
            
            return validated_query
            
        except Exception as e:
            logger.error(f"Secure text-to-cypher pipeline failed: {str(e)}")
            raise
    
    def execute_cypher_securely(self, cypher_query):
        """Execute Cypher via MCP with security validation"""
        try:
            # Final security validation
            validated_query = self.validate_cypher_security(cypher_query)
            
            logger.info(f"Executing validated Cypher via MCP: {validated_query}")
            
            # Execute via MCP with our credentials
            result = self.call_mcp_tool("read_neo4j_cypher", {"query": validated_query})
            
            # Handle the MCP response format
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from the content array
                    return content[0].get("text", "No results")
                elif isinstance(content, str):
                    return content
            
            return "No results"
            
        except Exception as e:
            logger.error(f"Secure Cypher execution failed: {str(e)}")
            raise

# Initialize MCP client
mcp_client = SecureTextToCypherClient(config)

@app.route('/config', methods=['GET'])
def get_config():
    """Show current configuration (without sensitive data)"""
    return jsonify({
        'mcp_server': config.get('mcp.url'),
        'openai_model': config.get('openai.model'),
        'security': {
            'max_query_length': config.get('security.max_query_length'),
            'allowed_operations': config.get('security.allowed_operations'),
            'blocked_operations': config.get('security.blocked_operations')
        },
        'service': {
            'host': config.get('service.host'),
            'port': config.get('service.port'),
            'log_level': config.get('service.log_level')
        },
        'openai_configured': bool(config.get('openai.api_key'))
    }), 200

@app.route('/debug/tools', methods=['GET'])
def debug_tools():
    """List available MCP tools"""
    try:
        result = mcp_client.call_mcp_tool("tools/list", {})
        return jsonify({
            'mcp_server_url': mcp_client.mcp_url,
            'available_tools': result,
            'security_note': 'LLM has no direct DB access. All queries validated.',
            'status': 'success'
        }), 200
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'mcp_server_url': mcp_client.mcp_url,
            'status': 'failed'
        }), 500

@app.route('/schema', methods=['GET'])
def get_schema():
    """Get Neo4j database schema via secure MCP connection"""
    try:
        schema = mcp_client.get_schema()
        return jsonify({
            'schema': schema,
            'note': 'Retrieved via secure MCP connection with our credentials',
            'status': 'success'
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'service': 'secure-text-to-cypher',
        'security': 'LLM isolated from DB',
        'config_loaded': True
    }), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Secure Text to Cypher Converter',
        'description': 'AI-powered text-to-cypher with security isolation',
        'security_model': {
            'llm_role': 'Generates Cypher queries (no DB access)',
            'mcp_role': 'Executes queries with our credentials',
            'validation': 'All queries security-validated before execution'
        },
        'endpoints': {
            'POST /generate': 'Generate Cypher from English text (add "execute": true to run)',
            'POST /execute': 'Execute a pre-validated Cypher query',
            'GET /schema': 'Get database schema (secure)',
            'GET /config': 'View configuration (safe)',
            'GET /debug/tools': 'List available MCP tools',
            'GET /health': 'Health check'
        },
        'configuration': 'Managed via config.yaml'
    }), 200

@app.route('/execute', methods=['POST'])
def execute_cypher():
    """Execute a Cypher query with security validation"""
    try:
        data = request.get_json()
        
        if not data or 'cypher' not in data:
            return jsonify({'error': 'No cypher query provided. Send JSON with "cypher" field.'}), 400
        
        cypher_query = data['cypher']
        
        if not cypher_query.strip():
            return jsonify({'error': 'Empty cypher query provided'}), 400
        
        # Execute with security validation
        execution_result = mcp_client.execute_cypher_securely(cypher_query)
        
        return jsonify({
            'cypher_query': cypher_query,
            'execution_result': execution_result,
            'security_validated': True,
            'status': 'success'
        }), 200
    
    except Exception as e:
        logger.error(f"Error executing Cypher: {str(e)}")
        return jsonify({
            'error': str(e),
            'security_note': 'Query blocked by security validation',
            'status': 'failed'
        }), 400

@app.route('/generate', methods=['POST'])
def generate_cypher():
    """AI-powered text-to-Cypher conversion with security isolation"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided. Send JSON with "text" field.'}), 400
        
        english_text = data['text']
        execute_query = data.get('execute', False)
        
        if not english_text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Secure AI-powered conversion
        cypher_query = mcp_client.secure_text_to_cypher(english_text)
        
        response_data = {
            'input_text': english_text,
            'cypher_query': cypher_query,
            'security_model': 'AI generated, security validated',
            'llm_db_access': False,
            'status': 'success'
        }
        
        # Optionally execute the validated query
        if execute_query:
            try:
                execution_result = mcp_client.execute_cypher_securely(cypher_query)
                response_data['execution_result'] = execution_result
                response_data['executed'] = True
                response_data['execution_security'] = 'Validated and executed via secure MCP'
            except Exception as e:
                response_data['execution_error'] = str(e)
                response_data['executed'] = False
        else:
            response_data['executed'] = False
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Error in secure text-to-cypher: {str(e)}")
        return jsonify({
            'error': str(e),
            'security_note': 'Request blocked by security validation',
            'status': 'failed'
        }), 400

if __name__ == '__main__':
    try:
        host = config.get('service.host', '0.0.0.0')
        port = config.get('service.port', 8081)
        debug = config.get('service.debug', False)
        
        logger.info(f"Starting Secure Text-to-Cypher service on {host}:{port}")
        logger.info(f"MCP Server: {config.get('mcp.url')}")
        logger.info(f"OpenAI Model: {config.get('openai.model')}")
        logger.info(f"Security: Max query length = {config.get('security.max_query_length')}")
        
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        exit(1)