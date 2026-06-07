#!/usr/bin/env python3
"""
Load Sample Data for Agentic System Demo
Creates sample student placement data for testing the agents
"""

import json
import os
from neo4j import GraphDatabase

def load_sample_data():
    """Load sample student placement data"""
    
    print("📊 Loading Sample Data for Agentic System")
    print("=" * 50)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'simple_ai_system', 'ai_agent_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Configuration file not found!")
        return False
    
    # Get database config
    db_config = config.get('neo4j', config.get('database', {}))
    uri = db_config.get('uri', 'bolt://localhost:7687')
    username = db_config.get('username', 'neo4j')
    password = db_config.get('password', '')
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Get database name from config
        database_name = db_config.get('database', 'neo4j')
        print(f"🎯 Target database: {database_name}")
        
        with driver.session(database=database_name) as session:
            print("🧹 Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            print("📝 Creating sample students...")
            
            # Create sample student data
            students_query = """
            CREATE 
            (s1:Student {student_id: 1001, name: 'Alice Johnson', age: 22, gender: 'Female', 
                        cgpa: 8.5, communication_skills: 4.2, soft_skills_rating: 4.0, 
                        aptitude_test_score: 85, placement_status: 'Placed'}),
            (s2:Student {student_id: 1002, name: 'Bob Smith', age: 23, gender: 'Male', 
                        cgpa: 7.8, communication_skills: 3.8, soft_skills_rating: 3.5, 
                        aptitude_test_score: 78, placement_status: 'Placed'}),
            (s3:Student {student_id: 1003, name: 'Carol Davis', age: 21, gender: 'Female', 
                        cgpa: 9.2, communication_skills: 4.8, soft_skills_rating: 4.5, 
                        aptitude_test_score: 92, placement_status: 'Placed'}),
            (s4:Student {student_id: 1004, name: 'David Wilson', age: 22, gender: 'Male', 
                        cgpa: 6.5, communication_skills: 3.2, soft_skills_rating: 3.0, 
                        aptitude_test_score: 65, placement_status: 'Not Placed'}),
            (s5:Student {student_id: 1005, name: 'Eva Brown', age: 24, gender: 'Female', 
                        cgpa: 8.9, communication_skills: 4.5, soft_skills_rating: 4.2, 
                        aptitude_test_score: 88, placement_status: 'Placed'}),
            (s6:Student {student_id: 1006, name: 'Frank Miller', age: 23, gender: 'Male', 
                        cgpa: 7.2, communication_skills: 3.5, soft_skills_rating: 3.3, 
                        aptitude_test_score: 72, placement_status: 'Not Placed'}),
            (s7:Student {student_id: 1007, name: 'Grace Lee', age: 22, gender: 'Female', 
                        cgpa: 8.7, communication_skills: 4.3, soft_skills_rating: 4.1, 
                        aptitude_test_score: 86, placement_status: 'Placed'}),
            (s8:Student {student_id: 1008, name: 'Henry Taylor', age: 25, gender: 'Male', 
                        cgpa: 7.0, communication_skills: 3.0, soft_skills_rating: 2.8, 
                        aptitude_test_score: 68, placement_status: 'Not Placed'}),
            (s9:Student {student_id: 1009, name: 'Ivy Chen', age: 21, gender: 'Female', 
                        cgpa: 9.5, communication_skills: 4.9, soft_skills_rating: 4.7, 
                        aptitude_test_score: 95, placement_status: 'Placed'}),
            (s10:Student {student_id: 1010, name: 'Jack Anderson', age: 23, gender: 'Male', 
                         cgpa: 8.1, communication_skills: 4.0, soft_skills_rating: 3.8, 
                         aptitude_test_score: 81, placement_status: 'Placed'})
            """
            
            session.run(students_query)
            
            print("🎓 Creating degree programs...")
            
            # Create degree programs
            degrees_query = """
            CREATE 
            (cs:Degree {name: 'Computer Science', level: 'Bachelor'}),
            (ee:Degree {name: 'Electrical Engineering', level: 'Bachelor'}),
            (me:Degree {name: 'Mechanical Engineering', level: 'Bachelor'}),
            (it:Degree {name: 'Information Technology', level: 'Bachelor'})
            """
            
            session.run(degrees_query)
            
            print("🔗 Creating relationships...")
            
            # Create relationships between students and degrees
            relationships_query = """
            MATCH (s1:Student {student_id: 1001}), (cs:Degree {name: 'Computer Science'})
            CREATE (s1)-[:PURSUING]->(cs)
            WITH 1 as dummy
            MATCH (s2:Student {student_id: 1002}), (ee:Degree {name: 'Electrical Engineering'})
            CREATE (s2)-[:PURSUING]->(ee)
            WITH 1 as dummy
            MATCH (s3:Student {student_id: 1003}), (cs:Degree {name: 'Computer Science'})
            CREATE (s3)-[:PURSUING]->(cs)
            WITH 1 as dummy
            MATCH (s4:Student {student_id: 1004}), (me:Degree {name: 'Mechanical Engineering'})
            CREATE (s4)-[:PURSUING]->(me)
            WITH 1 as dummy
            MATCH (s5:Student {student_id: 1005}), (it:Degree {name: 'Information Technology'})
            CREATE (s5)-[:PURSUING]->(it)
            WITH 1 as dummy
            MATCH (s6:Student {student_id: 1006}), (ee:Degree {name: 'Electrical Engineering'})
            CREATE (s6)-[:PURSUING]->(ee)
            WITH 1 as dummy
            MATCH (s7:Student {student_id: 1007}), (cs:Degree {name: 'Computer Science'})
            CREATE (s7)-[:PURSUING]->(cs)
            WITH 1 as dummy
            MATCH (s8:Student {student_id: 1008}), (me:Degree {name: 'Mechanical Engineering'})
            CREATE (s8)-[:PURSUING]->(me)
            WITH 1 as dummy
            MATCH (s9:Student {student_id: 1009}), (cs:Degree {name: 'Computer Science'})
            CREATE (s9)-[:PURSUING]->(cs)
            WITH 1 as dummy
            MATCH (s10:Student {student_id: 1010}), (it:Degree {name: 'Information Technology'})
            CREATE (s10)-[:PURSUING]->(it)
            """
            
            session.run(relationships_query)
            
            print("📊 Creating constraints for better performance...")
            
            # Create constraints
            constraints_query = """
            CREATE CONSTRAINT student_id_unique IF NOT EXISTS FOR (s:Student) REQUIRE s.student_id IS UNIQUE
            """
            
            session.run(constraints_query)
            
            # Verify data was loaded
            count_result = session.run("MATCH (n) RETURN count(n) as total")
            total_nodes = count_result.single()["total"]
            
            student_count = session.run("MATCH (s:Student) RETURN count(s) as count").single()["count"]
            degree_count = session.run("MATCH (d:Degree) RETURN count(d) as count").single()["count"]
            relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            print(f"\n✅ Sample data loaded successfully!")
            print(f"   📊 Total nodes: {total_nodes}")
            print(f"   👥 Students: {student_count}")
            print(f"   🎓 Degrees: {degree_count}")
            print(f"   🔗 Relationships: {relationship_count}")
            
            # Show placement statistics
            placed_count = session.run("MATCH (s:Student {placement_status: 'Placed'}) RETURN count(s) as count").single()["count"]
            placement_rate = (placed_count / student_count) * 100
            
            print(f"\n📈 Placement Statistics:")
            print(f"   ✅ Placed: {placed_count}/{student_count} ({placement_rate:.1f}%)")
            
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return False

def main():
    """Main function"""
    
    success = load_sample_data()
    
    if success:
        print("\n🎉 Sample data loaded! You can now test the agentic system with:")
        print("   • 'How many students are placed?'")
        print("   • 'Show me students with CGPA > 8.0'")
        print("   • 'Analyze placement success factors'")
        print("   • 'What's the average CGPA of placed students?'")
    else:
        print("\n❌ Failed to load sample data.")

if __name__ == "__main__":
    main()