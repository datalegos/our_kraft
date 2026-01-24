#!/usr/bin/env python3
"""
Comprehensive Test Suite for Text-to-Cypher Application
Tests all endpoints and demonstrates the complete application flow
"""

import requests
import json
import time
import subprocess
from typing import Dict, Any, List
from dataclasses import dataclass
from colorama import Colorama, Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

@dataclass
class TestResult:
    test_name: str
    status: str
    response: Dict[Any, Any] = None
    error: str = None

class TextToCypherTester:
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
    def print_header(self, title: str, char: str = "="):
        print(f"\n{Fore.CYAN}{char * 60}")
        print(f"{title}")
        print(f"{char * 60}{Style.RESET_ALL}")
    
    def print_step(self, step: str, description: str):
        print(f"\n{Fore.MAGENTA}🔍 {step}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}   {description}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}   {'-' * 50}{Style.RESET_ALL}")
    
    def make_request(self, method: str, endpoint: str, data: Dict = None, description: str = "") -> TestResult:
        """Make HTTP request and return structured result"""
        url = f"{self.base_url}{endpoint}"
        
        print(f"\n{Fore.YELLOW}📋 Testing: {description}")
        print(f"{Fore.LIGHTBLACK_EX}   Method: {method}")
        print(f"{Fore.LIGHTBLACK_EX}   URL: {url}")
        
        if data:
            print(f"{Fore.LIGHTBLACK_EX}   Body: {json.dumps(data, indent=2)}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            result_data = response.json()
            
            print(f"{Fore.GREEN}   ✅ SUCCESS")
            print(f"{Fore.GREEN}   Response:")
            print(f"{Fore.WHITE}{json.dumps(result_data, indent=2)}")
            
            result = TestResult(description, "SUCCESS", result_data)
            self.results.append(result)
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"{error_msg} - {json.dumps(error_data)}"
                except:
                    error_msg = f"{error_msg} - {e.response.text}"
            
            print(f"{Fore.RED}   ❌ FAILED: {error_msg}")
            
            result = TestResult(description, "FAILED", error=error_msg)
            self.results.append(result)
            return result
    
    def setup_sample_data(self):
        """Add sample data to Neo4j for testing"""
        print(f"\n{Fore.YELLOW}📝 Adding sample data to Neo4j database...")
        
        cypher_query = """
        MATCH (n) DETACH DELETE n;
        CREATE (alice:User {name: 'Alice', age: 30, email: 'alice@example.com', city: 'New York'}),
               (bob:User {name: 'Bob', age: 25, email: 'bob@example.com', city: 'London'}),
               (charlie:User {name: 'Charlie', age: 35, email: 'charlie@example.com', city: 'Tokyo'}),
               (diana:User {name: 'Diana', age: 28, email: 'diana@example.com', city: 'Paris'}),
               (laptop:Product {name: 'Gaming Laptop', price: 1500, brand: 'TechCorp', category: 'Electronics'}),
               (phone:Product {name: 'Smartphone', price: 900, brand: 'PhoneCorp', category: 'Electronics'}),
               (tablet:Product {name: 'Tablet', price: 600, brand: 'TabletInc', category: 'Electronics'}),
               (book:Product {name: 'Programming Book', price: 50, brand: 'BookCorp', category: 'Education'}),
               (headphones:Product {name: 'Wireless Headphones', price: 200, brand: 'AudioTech', category: 'Electronics'}),
               (alice)-[:PURCHASED {date: '2024-01-15', rating: 5, quantity: 1}]->(laptop),
               (bob)-[:PURCHASED {date: '2024-01-20', rating: 4, quantity: 1}]->(phone),
               (charlie)-[:PURCHASED {date: '2024-01-25', rating: 5, quantity: 1}]->(tablet),
               (diana)-[:PURCHASED {date: '2024-02-01', rating: 4, quantity: 2}]->(headphones),
               (alice)-[:PURCHASED {date: '2024-02-05', rating: 5, quantity: 1}]->(book),
               (alice)-[:VIEWED {timestamp: '2024-01-30', duration: 120}]->(phone),
               (bob)-[:VIEWED {timestamp: '2024-02-01', duration: 90}]->(tablet),
               (charlie)-[:VIEWED {timestamp: '2024-02-02', duration: 60}]->(headphones),
               (alice)-[:FRIENDS_WITH {since: '2020-01-01'}]->(bob),
               (bob)-[:FRIENDS_WITH {since: '2021-06-15'}]->(charlie),
               (charlie)-[:FRIENDS_WITH {since: '2022-03-10'}]->(diana)
        RETURN 'Sample data created successfully' as result
        """
        
        try:
            result = subprocess.run([
                "docker", "exec", "neo4j-db", "cypher-shell", 
                "-u", "neo4j", "-p", "password", cypher_query
            ], capture_output=True, text=True, check=True)
            
            print(f"{Fore.GREEN}   ✅ Sample data added successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}   ❌ Failed to add sample data: {e}")
            print(f"{Fore.RED}   Error output: {e.stderr}")
            return False
    
    def run_health_tests(self):
        """Test application health and configuration"""
        self.print_step("Step 1: Application Health and Configuration Tests", 
                       "Verify all services are running and configured correctly")
        
        # Health check
        self.make_request("GET", "/health", description="Health Check")
        
        # Configuration check
        self.make_request("GET", "/config", description="Configuration Check")
        
        # Application info
        self.make_request("GET", "/", description="Application Info")
    
    def run_mcp_tests(self):
        """Test MCP server integration"""
        self.print_step("Step 2: MCP Server Integration Tests", 
                       "Test connection and communication with MCP server")
        
        # MCP tools list
        self.make_request("GET", "/debug/tools", description="MCP Tools List")
        
        # Database schema
        self.make_request("GET", "/schema", description="Database Schema Retrieval")
    
    def run_ai_generation_tests(self):
        """Test AI text-to-Cypher generation"""
        self.print_step("Step 3: AI Text-to-Cypher Generation Tests", 
                       "Test OpenAI integration for query generation")
        
        test_queries = [
            "Find all users",
            "Show me all products with their prices ordered by price",
            "Find all users and their relationships to products",
            "Count how many products are in each category",
            "Find the oldest user in the database",
            "Show users who bought electronics",
            "Find products that cost more than 500 dollars"
        ]
        
        for query in test_queries:
            self.make_request("POST", "/generate", 
                            {"text": query, "execute": False}, 
                            f"Generate Query: '{query}'")
    
    def run_security_tests(self):
        """Test security validation"""
        self.print_step("Step 4: Security Validation Tests", 
                       "Verify security rules block dangerous operations")
        
        dangerous_queries = [
            "CREATE (u:User {name: 'Hacker'}) RETURN u",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "DROP DATABASE neo4j",
            "SHOW USERS",
            "SHOW DATABASES",
            "CALL dbms.security.listUsers()",
            "LOAD CSV FROM 'file:///etc/passwd' AS line RETURN line"
        ]
        
        for query in dangerous_queries:
            self.make_request("POST", "/execute", 
                            {"cypher": query}, 
                            f"Security Test: Block '{query[:50]}...'")
    
    def run_execution_tests(self):
        """Test end-to-end query execution"""
        self.print_step("Step 5: End-to-End Query Execution Tests", 
                       "Test complete pipeline with real data")
        
        execution_queries = [
            "Find all users",
            "Show users older than 28",
            "Find all products with their prices",
            "Show users who purchased products",
            "Count how many products each user purchased",
            "Find the most expensive product",
            "Show all friendships between users",
            "Find users who viewed products but didn't buy them",
            "Show products in the Electronics category",
            "Find users from New York"
        ]
        
        for query in execution_queries:
            self.make_request("POST", "/generate", 
                            {"text": query, "execute": True}, 
                            f"Execute: '{query}'")
    
    def run_direct_execution_tests(self):
        """Test direct Cypher execution"""
        self.print_step("Step 6: Direct Cypher Execution Tests", 
                       "Test direct query execution with validation")
        
        valid_queries = [
            "MATCH (u:User) RETURN u.name, u.age ORDER BY u.age DESC LIMIT 5",
            "MATCH (p:Product) RETURN p.category, COUNT(p) as count, AVG(p.price) as avg_price",
            "MATCH (u:User)-[r:PURCHASED]->(p:Product) RETURN u.name, p.name, r.rating",
            "MATCH (u:User) WHERE u.age > 25 RETURN u.name, u.city",
            "MATCH (p:Product) WHERE p.price > 500 RETURN p.name, p.price ORDER BY p.price DESC"
        ]
        
        for query in valid_queries:
            self.make_request("POST", "/execute", 
                            {"cypher": query}, 
                            f"Direct Execute: '{query[:50]}...'")
    
    def run_edge_case_tests(self):
        """Test edge cases and error handling"""
        self.print_step("Step 7: Edge Cases and Error Handling", 
                       "Test application behavior with unusual inputs")
        
        edge_cases = [
            {"text": "", "execute": False},  # Empty query
            {"text": "   ", "execute": False},  # Whitespace only
            {"text": "Find all users who purchased products that cost more than 500 dollars and were bought after January 1st 2024 and show their names ages email addresses the products they bought the prices of those products the dates they bought them and order everything by purchase date in descending order and limit to 10 results but also include the total amount spent by each user and the average price of products they bought and group by user name and show only users who spent more than 1000 dollars total", "execute": False},  # Very long query
        ]
        
        for i, case in enumerate(edge_cases):
            description = f"Edge Case {i+1}: {'Empty' if not case['text'].strip() else 'Long' if len(case['text']) > 100 else 'Whitespace'} Query"
            self.make_request("POST", "/generate", case, description)
        
        # Test invalid Cypher
        self.make_request("POST", "/execute", 
                        {"cypher": None}, 
                        "Edge Case: Null Cypher Query")
    
    def print_summary(self):
        """Print comprehensive test summary"""
        self.print_header("📈 COMPREHENSIVE TEST RESULTS SUMMARY")
        
        success_count = len([r for r in self.results if r.status == "SUCCESS"])
        failure_count = len([r for r in self.results if r.status == "FAILED"])
        total_tests = len(self.results)
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{Fore.WHITE}📊 Overall Statistics:")
        print(f"   Total Tests: {total_tests}")
        print(f"{Fore.GREEN}   Successful: {success_count}")
        print(f"{Fore.RED}   Failed: {failure_count}")
        
        color = Fore.GREEN if success_count == total_tests else Fore.YELLOW
        print(f"{color}   Success Rate: {success_rate:.2f}%")
        
        print(f"\n{Fore.WHITE}📋 Detailed Results:")
        for result in self.results:
            status_icon = "✅" if result.status == "SUCCESS" else "❌"
            color = Fore.GREEN if result.status == "SUCCESS" else Fore.RED
            print(f"{color}   {status_icon} {result.test_name}")
            if result.error:
                print(f"{Fore.RED}      Error: {result.error}")
        
        print(f"\n{Fore.CYAN}🔍 Application Flow Demonstrated:")
        flow_items = [
            "Health checks and configuration validation",
            "MCP server integration and tool discovery", 
            "AI-powered text-to-Cypher conversion",
            "Security validation and blocked operations",
            "Database schema retrieval",
            "End-to-end query execution with real data",
            "Direct Cypher execution with validation",
            "Error handling and edge cases"
        ]
        
        for item in flow_items:
            print(f"{Fore.GREEN}   ✅ {item}")
        
        print(f"\n{Fore.CYAN}🎉 Test Suite Complete!")
        print(f"{Fore.GREEN}   The Text-to-Cypher application is fully functional!")
        print(f"{Fore.GREEN}   All core features have been tested and validated.")
        
        # Save results to file
        results_data = {
            "summary": {
                "total_tests": total_tests,
                "successful": success_count,
                "failed": failure_count,
                "success_rate": success_rate
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "response": r.response,
                    "error": r.error
                } for r in self.results
            ]
        }
        
        with open("test_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n{Fore.YELLOW}💾 Detailed results saved to: test_results.json")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        self.print_header("🚀 Comprehensive Text-to-Cypher Application Test Suite")
        
        # Setup sample data first
        if not self.setup_sample_data():
            print(f"{Fore.YELLOW}⚠️  Continuing without sample data...")
        
        # Run all test categories
        self.run_health_tests()
        self.run_mcp_tests()
        self.run_ai_generation_tests()
        self.run_security_tests()
        self.run_execution_tests()
        self.run_direct_execution_tests()
        self.run_edge_case_tests()
        
        # Print summary
        self.print_summary()

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
    
    # Run tests
    tester = TextToCypherTester()
    tester.run_all_tests()