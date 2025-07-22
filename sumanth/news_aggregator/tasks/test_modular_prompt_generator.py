import json
from prompt_orchestrator import PromptOrchestrator, generate_user_prompt_json

def test_anonymous_user():
    """Test anonymous user prompt generation"""
    print("=== Testing Anonymous User ===")
    try:
        # Test using the orchestrator directly
        orchestrator = PromptOrchestrator()
        prompt = orchestrator.generate_user_prompt(user_id=None)
        print("Anonymous user prompt (via orchestrator):")
        print(json.dumps(prompt, indent=2))
        
        # Test using the convenience function
        prompt2 = generate_user_prompt_json(user_id=None)
        print("\nAnonymous user prompt (via convenience function):")
        print(json.dumps(prompt2, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

def test_new_user():
    """Test new user prompt generation"""
    print("\n=== Testing New User ===")
    try:
        # Test with a user ID that would be classified as new user
        # (assuming user ID 1 exists in database with no bookmarks/likes)
        orchestrator = PromptOrchestrator()
        prompt = orchestrator.generate_user_prompt(user_id=1)
        print("New user prompt:")
        print(json.dumps(prompt, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

def test_existing_user():
    """Test existing user prompt generation"""
    print("\n=== Testing Existing User ===")
    try:
        # Test with a user ID that would be classified as existing user
        # (assuming user ID 2 exists in database with bookmarks/likes)
        orchestrator = PromptOrchestrator()
        prompt = orchestrator.generate_user_prompt(user_id=2)
        print("Existing user prompt:")
        print(json.dumps(prompt, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

def test_user_type_determination():
    """Test user type determination logic"""
    print("\n=== Testing User Type Determination ===")
    try:
        orchestrator = PromptOrchestrator()
        
        # Test anonymous
        user_type = orchestrator.determine_user_type(None)
        print(f"User ID None -> Type: {user_type}")
        
        # Test registered users (these will fail if database is not available)
        try:
            user_type = orchestrator.determine_user_type(1)
            print(f"User ID 1 -> Type: {user_type}")
        except Exception as e:
            print(f"User ID 1 -> Error: {e}")
            
        try:
            user_type = orchestrator.determine_user_type(2)
            print(f"User ID 2 -> Type: {user_type}")
        except Exception as e:
            print(f"User ID 2 -> Error: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing Modular Prompt Generator System")
    print("=" * 50)
    
    # Test all user types
    test_anonymous_user()
    test_new_user()
    test_existing_user()
    test_user_type_determination()
    
    print("\n" + "=" * 50)
    print("Testing completed!") 