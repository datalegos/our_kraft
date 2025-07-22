from datetime import datetime

class AnonymousUserPromptGenerator:
    """
    Handles prompt generation for anonymous users.
    No profile or history needed - just general news.
    """
    def __init__(self):
        self.user_type = "anonymous"
    
    def generate_prompt(self, user_id=None):
        # Anonymous users get general global news - no personalization needed
        segments = [{
            "region_level": "global",
            "region": None,
            "category": "general",
            "count": 5,
            "prompt": "Give me the top 5 global news today in under 100 words each."
        }]
        return {
            "intent": "fetch_news",
            "user_type": self.user_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "segments": segments
        } 