import yaml
import os
from user_types import (
    AnonymousUserPromptGenerator,
    NewUserPromptGenerator,
    ExistingUserPromptGenerator
)

class PromptOrchestrator:
    """
    Main orchestrator that determines user type and delegates prompt generation
    to the appropriate user type module. For testing, reads from user_config.yaml.
    """
    def __init__(self):
        try:
            self.anonymous_generator = AnonymousUserPromptGenerator()
            self.new_user_generator = NewUserPromptGenerator()
            self.existing_user_generator = ExistingUserPromptGenerator()
            
            # Check if config file exists
            if not os.path.exists("user_config.yaml"):
                print("Warning: user_config.yaml not found. Creating default config.")
                self.config = self._create_default_config()
            else:
                with open("user_config.yaml", "r") as f:
                    self.config = yaml.safe_load(f)
                    if self.config is None:
                        print("Warning: user_config.yaml is empty. Using default config.")
                        self.config = self._create_default_config()
        except Exception as e:
            print(f"Error initializing PromptOrchestrator: {e}")
            self.config = self._create_default_config()
    
    def _create_default_config(self):
        """Create a default configuration for testing"""
        return {
            "user_type": "anonymous",
            "history": {
                "bookmarks": [],
                "likes": [],
                "search_topics": []
            }
        }

    def determine_user_type(self, user_id):
        """Determine user type based on configuration"""
        try:
            # For testing, ignore user_id and use config
            user_type = self.config.get("user_type", "anonymous")
            
            if user_type == "anonymous":
                return "anonymous"
            
            # If bookmarks, likes, and search_topics are all empty, treat as new user
            history = self.config.get("history", {})
            if (
                not history.get("bookmarks") and
                not history.get("likes") and
                not history.get("search_topics")
            ):
                return "new_user"
            
            return "existing_user"
        except Exception as e:
            print(f"Error determining user type: {e}")
            return "anonymous"

    def generate_user_prompt(self, user_id=None):
        """Generate user prompt based on user type - optimized to fetch only necessary data"""
        try:
            user_type = self.determine_user_type(user_id)
            
            if user_type == "anonymous":
                # Anonymous users: No need to fetch profile or history
                # Just generate general news prompt
                return self.anonymous_generator.generate_prompt(user_id)
                
            elif user_type == "new_user":
                # New users: Only need profile (location), no history needed
                # Generate location-based news with general category
                return self.new_user_generator.generate_prompt(user_id)
                
            elif user_type == "existing_user":
                # Existing users: Need both profile and history for personalization
                # Generate personalized news based on bookmarks, likes, search topics
                prompt = self.existing_user_generator.generate_prompt(user_id)
                if prompt is None:
                    # Fallback to new user if no personalization data available
                    return self.new_user_generator.generate_prompt(user_id)
                return prompt
            else:
                raise ValueError(f"Unknown user type: {user_type}")
            
        except Exception as e:
            print(f"Error generating user prompt: {e}")
            return f"Error: {str(e)}"

def generate_user_prompt_json(user_id=None):
    """Main function to generate user prompt"""
    try:
        orchestrator = PromptOrchestrator()
        result = orchestrator.generate_user_prompt(user_id)
        return result
    except Exception as e:
        print(f"Error in generate_user_prompt_json: {e}")
        return f"Error: {str(e)}"

# Test the function if run directly
if __name__ == "__main__":
    result = generate_user_prompt_json()