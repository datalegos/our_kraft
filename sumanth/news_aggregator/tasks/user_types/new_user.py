import yaml
from datetime import datetime

class NewUserPromptGenerator:
    """
    Handles prompt generation for newly registered users.
    Only needs profile (location) - no history needed since they're new.
    """
    def __init__(self):
        self.user_type = "new_user"
        # Only load profile data, not history
        with open("user_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def get_user_profile(self, user_id):
        # Only fetch profile (location) for new users
        return self.config.get("profile", {})

    def generate_prompt(self, user_id):
        if user_id is None:
            raise ValueError("User ID is required for new users.")
        
        # New users only need profile (location) - no history needed
        user_profile = self.get_user_profile(user_id)
        required_fields = ["country", "state", "district"]
        for field in required_fields:
            if field not in user_profile:
                raise ValueError(f"Missing required profile field: {field}")
        
        country = user_profile["country"]
        state = user_profile["state"]
        district = user_profile["district"]
        category = "general"  # Default category for new users
        
        segments = [
            # {
            #     "region_level": "global",
            #     "region": None,
            #     "category": category,
            #     "count": 3,
            #     "prompt": f"Give me the top 3 global {category} news headlines today."
            # },
            {
                "region_level": "country",
                "region": country,
                "category": category,
                "count": 2,
                "prompt": f"Give me 2 recent {category} news from {country} today."
            },
            {
                "region_level": "state",
                "region": state,
                "category": category,
                "count": 1,
                "prompt": f"Give me 1 latest {category} news from {state}."
            },
            {
                "region_level": "district",
                "region": district,
                "category": category,
                "count": 1,
                "prompt": f"Give me 1 local {category} news from {district}."
            }
        ]
        return {
            "intent": "fetch_news",
            "user_type": self.user_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_profile": {
                "country": country,
                "state": state,
                "district": district,
                "preferred_category": category
            },
            "segments": segments
        } 