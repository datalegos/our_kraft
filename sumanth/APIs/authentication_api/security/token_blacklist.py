"""
JWT Token Blacklist System
Tracks revoked/suspicious tokens to prevent misuse
"""

import redis
from datetime import datetime, timedelta
from flask import current_app
import hashlib

class TokenBlacklist:
    """Manages blacklisted JWT tokens"""
    
    def __init__(self, redis_client=None):
        # In production, use Redis for distributed blacklist
        # For development, use in-memory storage
        self.redis_client = redis_client
        self.memory_blacklist = set()  # For development
    
    def _get_token_hash(self, token):
        """Create hash of token for storage (don't store full token)"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def blacklist_token(self, token, reason="manual_revoke"):
        """Add token to blacklist"""
        token_hash = self._get_token_hash(token)
        expiry = datetime.utcnow() + timedelta(hours=24)  # Match JWT expiry
        
        if self.redis_client:
            # Production: Use Redis with expiration
            self.redis_client.setex(
                f"blacklist:{token_hash}", 
                int(timedelta(hours=24).total_seconds()),
                reason
            )
        else:
            # Development: Use memory
            self.memory_blacklist.add(token_hash)
        
        print(f"🚫 Token blacklisted: {reason}")
    
    def is_blacklisted(self, token):
        """Check if token is blacklisted"""
        token_hash = self._get_token_hash(token)
        
        if self.redis_client:
            return self.redis_client.exists(f"blacklist:{token_hash}")
        else:
            return token_hash in self.memory_blacklist
    
    def revoke_all_user_tokens(self, user_id):
        """Revoke all tokens for a specific user (when password changes)"""
        # This requires tracking tokens per user
        # Implementation depends on your needs
        pass

# Global blacklist instance
blacklist = TokenBlacklist()