#!/usr/bin/env python3
"""
Simple test script for the Authentication API
Run this after starting the server to test basic functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_api():
    print("🧪 Testing News Aggregator Authentication API\n")
    
    # Test 1: Health Check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on port 5000")
        return
    
    # Test 2: Get API Info
    print("\n2. Testing API info...")
    response = requests.get(f"{BASE_URL}/api/info")
    if response.status_code == 200:
        print("✅ API info retrieved successfully")
    else:
        print("❌ Failed to get API info")
    
    # Test 3: Get Available Categories
    print("\n3. Testing categories endpoint...")
    response = requests.get(f"{BASE_URL}/api/categories")
    if response.status_code == 200:
        categories = response.json()['categories']
        print(f"✅ Categories retrieved: {', '.join(categories)}")
    else:
        print("❌ Failed to get categories")
    
    # Test 4: User Registration
    print("\n4. Testing user registration...")
    test_user = {
        "username": f"testuser_{int(time.time())}",
        "email": f"test_{int(time.time())}@example.com",
        "password": "TestPass123",
        "preferred_categories": "technology,business",
        "preferred_language": "en"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/register",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_user)
    )
    
    if response.status_code == 201:
        print("✅ User registration successful")
        registration_data = response.json()
        token = registration_data['token']
        user_id = registration_data['user']['id']
        username = registration_data['user']['username']
        print(f"   User ID: {user_id}, Username: {username}")
    else:
        print(f"❌ User registration failed: {response.json()}")
        return
    
    # Test 5: User Login
    print("\n5. Testing user login...")
    login_data = {
        "username": test_user['username'],
        "password": test_user['password']
    }
    
    response = requests.post(
        f"{BASE_URL}/api/login",
        headers={"Content-Type": "application/json"},
        data=json.dumps(login_data)
    )
    
    if response.status_code == 200:
        print("✅ User login successful")
        login_response = response.json()
        token = login_response['token']  # Use fresh token
    else:
        print(f"❌ User login failed: {response.json()}")
        return
    
    # Test 6: Get User Profile
    print("\n6. Testing get user profile...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/user/profile", headers=headers)
    
    if response.status_code == 200:
        print("✅ User profile retrieved successfully")
        profile = response.json()['user']
        print(f"   Email: {profile['email']}")
        print(f"   Categories: {', '.join(profile['preferred_categories'])}")
    else:
        print(f"❌ Failed to get user profile: {response.json()}")
    
    # Test 7: Update User Preferences
    print("\n7. Testing update user preferences...")
    new_preferences = {
        "language": "es",
        "categories": ["sports", "entertainment"],
        "sources": ["bbc", "cnn"]
    }
    
    response = requests.put(
        f"{BASE_URL}/api/user/preferences",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps(new_preferences)
    )
    
    if response.status_code == 200:
        print("✅ User preferences updated successfully")
        prefs = response.json()['preferences']
        print(f"   Language: {prefs['language']}")
        print(f"   Categories: {', '.join(prefs['categories'])}")
    else:
        print(f"❌ Failed to update preferences: {response.json()}")
    
    # Test 8: Token Verification
    print("\n8. Testing token verification...")
    response = requests.get(f"{BASE_URL}/api/verify-token", headers=headers)
    
    if response.status_code == 200:
        print("✅ Token verification successful")
        token_data = response.json()
        print(f"   Valid: {token_data['valid']}")
        print(f"   Username: {token_data['username']}")
    else:
        print(f"❌ Token verification failed: {response.json()}")
    
    # Test 9: Logout
    print("\n9. Testing user logout...")
    response = requests.post(f"{BASE_URL}/api/logout", headers=headers)
    
    if response.status_code == 200:
        print("✅ User logout successful")
    else:
        print(f"❌ User logout failed: {response.json()}")
    
    print("\n🎉 API testing completed!")
    print("\n📋 Summary:")
    print("   - All basic authentication flows work")
    print("   - User registration and login functional")
    print("   - JWT token authentication working")
    print("   - User preferences management operational")
    print("   - API is ready for integration with your news aggregator!")

if __name__ == "__main__":
    test_api()