#!/usr/bin/env python3
"""
Example usage of the Modular Prompt Generator System

This script demonstrates how to use the new modular system
to generate prompts for different user types.
"""

import json
from prompt_orchestrator import PromptOrchestrator, generate_user_prompt_json

def example_anonymous_user():
    """Example: Generate prompt for anonymous user"""
    print("🔍 Example: Anonymous User")
    print("-" * 40)
    
    # Method 1: Using the convenience function
    prompt = generate_user_prompt_json(user_id=None)
    
    print("Generated Prompt:")
    print(json.dumps(prompt, indent=2))
    print()

def example_registered_user():
    """Example: Generate prompt for registered user (type determined automatically)"""
    print("🔍 Example: Registered User (Auto-determined type)")
    print("-" * 40)
    
    # The system will automatically determine if this is a new or existing user
    # based on their bookmarks and likes history
    try:
        prompt = generate_user_prompt_json(user_id=1)
        print("Generated Prompt:")
        print(json.dumps(prompt, indent=2))
    except Exception as e:
        print(f"Error (likely database not available): {e}")
    print()

def example_orchestrator_usage():
    """Example: Using the orchestrator directly"""
    print("🔍 Example: Using Orchestrator Directly")
    print("-" * 40)
    
    orchestrator = PromptOrchestrator()
    
    # Determine user type
    try:
        user_type = orchestrator.determine_user_type(user_id=1)
        print(f"User ID 1 is classified as: {user_type}")
        
        # Generate prompt
        prompt = orchestrator.generate_user_prompt(user_id=1)
        print("Generated Prompt:")
        print(json.dumps(prompt, indent=2))
    except Exception as e:
        print(f"Error (likely database not available): {e}")
    print()

def example_direct_module_usage():
    """Example: Using individual modules directly"""
    print("🔍 Example: Direct Module Usage")
    print("-" * 40)
    
    from user_types import AnonymousUserPromptGenerator, NewUserPromptGenerator, ExistingUserPromptGenerator
    
    # Anonymous user
    anon_generator = AnonymousUserPromptGenerator()
    anon_prompt = anon_generator.generate_prompt()
    print("Anonymous User Prompt:")
    print(json.dumps(anon_prompt, indent=2))
    print()
    
    # New user (would need valid user_id in database)
    try:
        new_generator = NewUserPromptGenerator()
        new_prompt = new_generator.generate_prompt(user_id=1)
        print("New User Prompt:")
        print(json.dumps(new_prompt, indent=2))
    except Exception as e:
        print(f"New user error (likely database not available): {e}")
    print()
    
    # Existing user (would need valid user_id in database)
    try:
        existing_generator = ExistingUserPromptGenerator()
        existing_prompt = existing_generator.generate_prompt(user_id=2)
        print("Existing User Prompt:")
        print(json.dumps(existing_prompt, indent=2))
    except Exception as e:
        print(f"Existing user error (likely database not available): {e}")
    print()

def main():
    """Main function to run all examples"""
    print("🚀 Modular Prompt Generator System - Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_anonymous_user()
    example_registered_user()
    example_orchestrator_usage()
    example_direct_module_usage()
    
    print("✅ Examples completed!")
    print("\n💡 Note: Database-dependent examples will fail if the database is not available.")
    print("   This is expected behavior for demonstration purposes.")

if __name__ == "__main__":
    main() 