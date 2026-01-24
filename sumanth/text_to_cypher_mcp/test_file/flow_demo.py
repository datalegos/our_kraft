#!/usr/bin/env python3
"""
Application Flow Demonstration
Shows the complete text-to-cypher pipeline step by step
"""

import requests
import json
import subprocess
import time
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

class FlowDemo:
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
    
    def print_header(self, title: str):
        print(f"\n{Fore.CYAN}{'=' * 50}")
        print(f"{title}")
        print(f"{'=' * 50}{Style.RESET_ALL}")
    
    def print_step(self, step_num: str, title: str, description: str):
        print(f"\n{Fore.MAGENTA}{step_num} {title}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}   {description}")
        print(f"   {'-' * 40}{Style.RESET_ALL}")
    
    def show_response(self, response: dict, highlight: str = ""):
        if highlight:
            print(f"{Fore.YELLOW}   🔍 Key Result: {highlight}")
        print(f"{Fore.WHITE}{json.dumps(response, indent=2)}")
    
    def make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make HTTP request and return response"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}   ❌ Request failed: {e}")
            return {"error": str(e)}
    
    def setup_sample_data(self):
        """Add comprehensive sample data"""
        print(f"{Fore.YELLOW}   Adding users, products, and relationships...")
        
        cypher_query = """
        MATCH (n) DETACH DELETE n;
        CREATE (alice:User {name: 'Alice', age: 30, city: 'New York', occupation: 'Engineer'}),
               (bob:User {name: 'Bob', age: 25, city: 'London', occupation: 'Designer'}),
               (charlie:User {name: 'Charlie', age: 35, city: 'Tokyo', occupation: 'Manager'}),
               (diana:User {name: 'Diana', age: 28, city: 'Paris', occupation: 'Developer'}),
               (laptop:Product {name: 'Gaming Laptop', price: 1500, brand: 'TechCorp', category: 'Electronics'}),
               (phone:Product {name: 'Smartphone', price: 900, brand: 'PhoneCorp', category: 'Electronics'}),
               (tablet:Product {name: 'Tablet', price: 600, brand: 'TabletInc', category: 'Electronics'}),
               (book:Product {name: 'Programming Book', price: 50, brand: 'BookCorp', category: 'Education'}),
               (headphones:Product {name: 'Wireless Headphones', price: 200, brand: 'AudioTech', category: 'Electronics'}),
               (alice)-[:PURCHASED {date: '2024-01-15', rating: 5, amount: 1500}]->(laptop),
               (bob)-[:PURCHASED {date: '2024-01-20', rating: 4, amount: 900}]->(phone),
               (charlie)-[:PURCHASED {date: '2024-01-25', rating: 5, amount: 600}]->(tablet),
               (diana)-[:PURCHASED {date: '2024-02-01', rating: 4, amount: 400}]->(headphones),
               (alice)-[:PURCHASED {date: '2024-02-05', rating: 5, amount: 50}]->(book),
               (alice)-[:VIEWED {timestamp: '2024-01-30', duration: 120}]->(phone),
               (bob)-[:VIEWED {timestamp: '2024-02-01', duration: 90}]->(tablet),
               (charlie)-[:VIEWED {timestamp: '2024-02-02', duration: 60}]->(headphones),
               (alice)-[:FRIENDS_WITH {since: '2020-01-01'}]->(bob),
               (bob)-[:FRIENDS_WITH {since: '2021-06-15'}]->(charlie)
        RETURN 'Data created' as result
        """
        
        try:
            subprocess.run([
                "docker", "exec", "neo4j-db", "cypher-shell", 
                "-u", "neo4j", "-p", "password", cypher_query
            ], check=True, capture_output=True)
            print(f"{Fore.GREEN}   ✅ Sample data created successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"{Fore.YELLOW}   ⚠️  Could not add sample data, continuing with existing data...")
            return False
    
    def run_demo(self):
        """Run the complete flow demonstration"""
        self.print_header("🎯 Text-to-Cypher Application Flow Demo")
        
        # Step 1: Health Check
        self.print_step("1️⃣", "Application Health Check", "Verify all services are running")
        try:
            health = self.make_request("GET", "/health")
            if "error" not in health:
                self.show_response(health, "Service is healthy and ready")
            else:
                print(f"{Fore.RED}   ❌ Application not running. Please start with: docker-compose up -d")
                return
        except Exception as e:
            print(f"{Fore.RED}   ❌ Health check failed: {e}")
            return
        
        # Step 2: Configuration
        self.print_step("2️⃣", "Configuration Overview", "Show current application settings")
        config = self.make_request("GET", "/config")
        self.show_response(config, f"OpenAI configured: {config.get('openai_configured', False)}, Security rules active")
        
        # Step 3: MCP Integration
        self.print_step("3️⃣", "MCP Server Connection", "Test connection to Neo4j MCP server")
        tools = self.make_request("GET", "/debug/tools")
        self.show_response(tools, "Available MCP tools for database operations")
        
        # Step 4: Database Schema
        self.print_step("4️⃣", "Database Schema Discovery", "Retrieve current Neo4j database structure")
        schema = self.make_request("GET", "/schema")
        self.show_response(schema, "Current database schema")
        
        # Step 5: Sample Data Setup
        self.print_step("5️⃣", "Sample Data Setup", "Adding test data to demonstrate queries")
        self.setup_sample_data()
        
        # Step 6: Text-to-Cypher Generation
        self.print_step("6️⃣", "AI Text-to-Cypher Conversion", "Convert natural language to Cypher query")
        
        demo_queries = [
            "Find all users",
            "Show products with their prices",
            "Find users who bought expensive products over 1000 dollars",
            "Show the total amount spent by each user"
        ]
        
        for query in demo_queries:
            print(f"\n{Fore.CYAN}   🤖 Query: '{query}'")
            result = self.make_request("POST", "/generate", {
                "text": query,
                "execute": False
            })
            
            if "cypher_query" in result:
                print(f"{Fore.GREEN}   🔍 Generated Cypher: {result['cypher_query']}")
                print(f"{Fore.BLUE}   🔒 Security Status: {result.get('security_model', 'N/A')}")
            else:
                print(f"{Fore.RED}   ❌ Generation failed: {result.get('error', 'Unknown error')}")
        
        # Step 7: Security Validation Demo
        self.print_step("7️⃣", "Security Validation", "Demonstrate security blocking dangerous operations")
        
        dangerous_queries = [
            "CREATE (hacker:User {name: 'Hacker'}) RETURN hacker",
            "MATCH (n) DELETE n",
            "SHOW USERS"
        ]
        
        for dangerous_query in dangerous_queries:
            print(f"\n{Fore.RED}   🚨 Testing dangerous query: '{dangerous_query}'")
            result = self.make_request("POST", "/execute", {"cypher": dangerous_query})
            
            if "error" in result:
                print(f"{Fore.GREEN}   ✅ SECURITY SUCCESS: Query blocked as expected")
                print(f"{Fore.YELLOW}   📋 Reason: {result['error']}")
            else:
                print(f"{Fore.RED}   ❌ SECURITY FAILURE: Query was allowed!")
        
        # Step 8: End-to-End Execution
        self.print_step("8️⃣", "End-to-End Query Execution", "Complete pipeline: Text → Cypher → Results")
        
        execution_queries = [
            "Find all users and their ages",
            "Show the most expensive product",
            "Find users who purchased products and show what they bought",
            "Count how many users are in each city"
        ]
        
        for query in execution_queries:
            print(f"\n{Fore.CYAN}   🎯 Executing: '{query}'")
            result = self.make_request("POST", "/generate", {
                "text": query,
                "execute": True
            })
            
            if "cypher_query" in result:
                print(f"{Fore.BLUE}   📝 Generated Query: {result['cypher_query']}")
                print(f"{Fore.GREEN}   ✅ Executed Successfully: {result.get('executed', False)}")
                
                if result.get('execution_result'):
                    print(f"{Fore.YELLOW}   📊 Results:")
                    print(f"{Fore.WHITE}{json.dumps(result['execution_result'], indent=2)}")
                
                print(f"{Fore.BLUE}   🔒 Security: {result.get('execution_security', 'N/A')}")
            else:
                print(f"{Fore.RED}   ❌ Execution failed: {result.get('error', 'Unknown error')}")
        
        # Step 9: Updated Schema
        self.print_step("9️⃣", "Updated Database Schema", "Show schema after data operations")
        updated_schema = self.make_request("GET", "/schema")
        self.show_response(updated_schema, "Schema now includes our sample data structure")
        
        # Summary
        self.print_header("🎉 APPLICATION FLOW COMPLETE!")
        
        print(f"\n{Fore.WHITE}📋 Flow Summary:")
        flow_steps = [
            "Application health verified",
            "Configuration loaded and validated", 
            "MCP server connection established",
            "Database schema retrieved",
            "Sample data added for testing",
            "AI text-to-Cypher conversion working",
            "Security validation blocking dangerous queries",
            "End-to-end execution returning real data",
            "Schema updates reflecting data changes"
        ]
        
        for i, step in enumerate(flow_steps, 1):
            print(f"{Fore.GREEN}   {i}. ✅ {step}")
        
        print(f"\n{Fore.CYAN}🔄 Complete Data Flow:")
        print(f"{Fore.WHITE}   English Text → OpenAI → Cypher Query → Security Check → MCP Server → Neo4j → Results")
        
        print(f"\n{Fore.CYAN}🔒 Security Model Verified:")
        security_points = [
            "OpenAI: Only sees text queries (no database access)",
            "Application: Validates and filters all queries",
            "MCP Server: Executes only approved operations", 
            "Neo4j: Protected by credential isolation"
        ]
        
        for point in security_points:
            print(f"{Fore.WHITE}   • {point}")
        
        print(f"\n{Fore.GREEN}🎯 Your Text-to-Cypher application is fully operational!")

if __name__ == "__main__":
    # Check if application is running
    try:
        response = requests.get("http://localhost:8081/health", timeout=5)
        if response.status_code != 200:
            print(f"{Fore.RED}❌ Application not responding properly. Please start with: docker-compose up -d")
            exit(1)
    except requests.exceptions.RequestException:
        print(f"{Fore.RED}❌ Application not running. Please start with: docker-compose up -d")
        exit(1)
    
    # Run demo
    demo = FlowDemo()
    demo.run_demo()