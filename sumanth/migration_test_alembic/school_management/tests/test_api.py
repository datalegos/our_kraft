import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_departments():
    print("=== Testing Department CRUD ===")
    
    # Create department
    dept_data = {
        "name": "Computer Science",
        "description": "Department of Computer Science and Engineering"
    }
    
    print("1. Creating department...")
    response = requests.post(f"{BASE_URL}/api/departments/", json=dept_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        dept = response.json()
        print(f"Created department: {dept['name']} (ID: {dept['id']})")
        dept_id = dept['id']
    else:
        print(f"Error: {response.text}")
        return None
    
    # Get all departments
    print("\n2. Getting all departments...")
    response = requests.get(f"{BASE_URL}/api/departments/")
    print(f"Status: {response.status_code}")
    departments = response.json()
    print(f"Found {len(departments)} departments")
    
    # Get specific department
    print(f"\n3. Getting department {dept_id}...")
    response = requests.get(f"{BASE_URL}/api/departments/{dept_id}")
    print(f"Status: {response.status_code}")
    dept = response.json()
    print(f"Department: {dept['name']}")
    
    return dept_id

def test_students(dept_id):
    print("\n=== Testing Student CRUD ===")
    
    # Create student
    student_data = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@university.edu",
        "student_id": "CS001",
        "phone": "+1234567890",
        "address": "123 Main St, City, State",
        "department_id": dept_id
    }
    
    print("1. Creating student...")
    response = requests.post(f"{BASE_URL}/api/students/", json=student_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        student = response.json()
        print(f"Created student: {student['first_name']} {student['last_name']} (ID: {student['id']})")
        student_id = student['id']
    else:
        print(f"Error: {response.text}")
        return None
    
    # Get all students
    print("\n2. Getting all students...")
    response = requests.get(f"{BASE_URL}/api/students/")
    print(f"Status: {response.status_code}")
    students = response.json()
    print(f"Found {len(students)} students")
    
    # Get specific student with department info
    print(f"\n3. Getting student {student_id} with department...")
    response = requests.get(f"{BASE_URL}/api/students/{student_id}")
    print(f"Status: {response.status_code}")
    student = response.json()
    print(f"Student: {student['first_name']} {student['last_name']}")
    if student.get('department'):
        print(f"Department: {student['department']['name']}")
    
    # Update student
    print(f"\n4. Updating student {student_id}...")
    update_data = {"phone": "+0987654321"}
    response = requests.put(f"{BASE_URL}/api/students/{student_id}", json=update_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        updated_student = response.json()
        print(f"Updated phone: {updated_student['phone']}")
    
    return student_id

def test_ui():
    print("\n=== Testing UI ===")
    
    print("1. Testing main page...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ UI is accessible")
    else:
        print("✗ UI is not accessible")
    
    print("\n2. Testing static files...")
    response = requests.get(f"{BASE_URL}/static/css/style.css")
    print(f"CSS Status: {response.status_code}")
    
    response = requests.get(f"{BASE_URL}/static/js/app.js")
    print(f"JS Status: {response.status_code}")

def test_relationships(dept_id):
    print("\n=== Testing Relationships ===")
    
    # Get students by department
    print(f"Getting students in department {dept_id}...")
    response = requests.get(f"{BASE_URL}/api/departments/{dept_id}/students")
    print(f"Status: {response.status_code}")
    students = response.json()
    print(f"Students in department: {len(students)}")
    for student in students:
        print(f"  - {student['first_name']} {student['last_name']} ({student['student_id']})")

def main():
    print("🎓 Testing School Management System")
    print("Make sure the API is running on http://localhost:8000")
    print("-" * 60)
    
    try:
        # Test UI first
        test_ui()
        
        # Test departments
        dept_id = test_departments()
        if not dept_id:
            return
        
        # Test students
        student_id = test_students(dept_id)
        if not student_id:
            return
        
        # Test relationships
        test_relationships(dept_id)
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("\n📱 Open your browser and visit:")
        print(f"   {BASE_URL}")
        print("   to see the beautiful UI in action!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API.")
        print("   Make sure it's running on http://localhost:8000")
        print("   Run: python main.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()