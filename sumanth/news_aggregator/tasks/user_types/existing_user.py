import yaml
from datetime import datetime
from collections import Counter

class ExistingUserPromptGenerator:
    """
    Handles prompt generation for existing registered users.
    For testing, reads user profile and history from user_config.yaml.
    """
    def __init__(self):
        self.user_type = "existing_user"
        with open("user_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def get_user_profile(self, user_id):
        return self.config.get("profile", {})

    def get_user_history(self, user_id):
        return self.config.get("history", {})

    def get_user_id(self, user_id):
        # If user_id is provided, use it; else try to get from config for testing
        if user_id is not None:
            return user_id
        return self.config.get("user_id", None)

    def get_preferred_category(self, history):
        categories = (history.get("bookmarks") or []) + (history.get("likes") or [])
        if categories:
            return Counter(categories).most_common(1)[0][0]
        search_topics = history.get("search_topics") or []
        if search_topics:
            return Counter(search_topics).most_common(1)[0][0]
        return None

    def generate_prompt(self, user_id):
        user_profile = self.get_user_profile(user_id)
        history = self.get_user_history(user_id)
        user_id_value = self.get_user_id(user_id)
        required_profile_fields = ["country", "state", "district"]
        for field in required_profile_fields:
            if field not in user_profile:
                raise ValueError(f"Missing required profile field: {field}")
        category = self.get_preferred_category(history)
        if not category:
            return None
        country = user_profile["country"]
        state = user_profile["state"]
        district = user_profile["district"]
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
            "user_id": user_id_value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_profile": {
                "country": country,
                "state": state,
                "district": district,
                "preferred_category": category
            },
            "segments": segments
        } 


