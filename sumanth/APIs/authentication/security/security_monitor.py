"""
Security monitoring for JWT token abuse detection
Detects suspicious patterns that might indicate token theft
"""

from datetime import datetime, timedelta
from collections import defaultdict
import geoip2.database
from flask import request
import hashlib

class SecurityMonitor:
    """Monitor for suspicious JWT token usage"""
    
    def __init__(self):
        # Track token usage patterns
        self.token_usage = defaultdict(list)  # token_hash -> [usage_records]
        self.user_sessions = defaultdict(list)  # user_id -> [session_records]
    
    def record_token_usage(self, token, user_id, ip_address, user_agent):
        """Record token usage for analysis"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]  # Short hash
        
        usage_record = {
            'timestamp': datetime.utcnow(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'user_id': user_id
        }
        
        # Store usage history
        self.token_usage[token_hash].append(usage_record)
        self.user_sessions[user_id].append(usage_record)
        
        # Clean old records (keep last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.token_usage[token_hash] = [
            r for r in self.token_usage[token_hash] 
            if r['timestamp'] > cutoff
        ]
    
    def detect_suspicious_activity(self, token, user_id, ip_address, user_agent):
        """Detect if token usage looks suspicious"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        usage_history = self.token_usage.get(token_hash, [])
        
        if len(usage_history) < 2:
            return False, "Insufficient data"
        
        current_usage = {
            'ip_address': ip_address,
            'user_agent': user_agent,
            'timestamp': datetime.utcnow()
        }
        
        last_usage = usage_history[-1]
        
        # Check for suspicious patterns
        suspicious_reasons = []
        
        # 1. Different IP addresses in short time
        if (current_usage['ip_address'] != last_usage['ip_address'] and 
            (current_usage['timestamp'] - last_usage['timestamp']).seconds < 300):  # 5 minutes
            suspicious_reasons.append("IP address changed rapidly")
        
        # 2. Different user agents (different devices/browsers)
        if current_usage['user_agent'] != last_usage['user_agent']:
            suspicious_reasons.append("Different device/browser detected")
        
        # 3. Too many different IPs for same token
        unique_ips = set(r['ip_address'] for r in usage_history)
        if len(unique_ips) > 3:  # More than 3 different IPs
            suspicious_reasons.append("Token used from too many locations")
        
        # 4. Unusual usage frequency
        recent_usage = [r for r in usage_history 
                       if (current_usage['timestamp'] - r['timestamp']).seconds < 3600]  # Last hour
        if len(recent_usage) > 100:  # More than 100 requests per hour
            suspicious_reasons.append("Unusually high request frequency")
        
        return len(suspicious_reasons) > 0, suspicious_reasons
    
    def get_user_active_sessions(self, user_id):
        """Get all active sessions for a user"""
        cutoff = datetime.utcnow() - timedelta(hours=1)  # Active in last hour
        recent_sessions = [
            s for s in self.user_sessions.get(user_id, [])
            if s['timestamp'] > cutoff
        ]
        
        # Group by IP and User Agent
        sessions = {}
        for session in recent_sessions:
            key = f"{session['ip_address']}_{hash(session['user_agent'])}"
            if key not in sessions:
                sessions[key] = {
                    'ip_address': session['ip_address'],
                    'user_agent': session['user_agent'],
                    'first_seen': session['timestamp'],
                    'last_seen': session['timestamp'],
                    'request_count': 1
                }
            else:
                sessions[key]['last_seen'] = session['timestamp']
                sessions[key]['request_count'] += 1
        
        return list(sessions.values())

# Global security monitor
security_monitor = SecurityMonitor()