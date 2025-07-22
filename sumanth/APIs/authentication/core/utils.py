import jwt
import re
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask import current_app, request, jsonify
from functools import wraps

# Simple in-memory token blacklist
BLACKLISTED_TOKENS = set()

# Password utilities
def hash_password(password):
    return generate_password_hash(password)

def verify_password(password, password_hash):
    return check_password_hash(password_hash, password)

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# JWT utilities
def generate_jwt(user_id, username):
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=24),  # 24 hour expiry
        'iat': datetime.utcnow()
    }
    # Use dedicated JWT secret key for better security
    jwt_key = current_app.config.get('JWT_SECRET_KEY', current_app.config['SECRET_KEY'])
    return jwt.encode(payload, jwt_key, algorithm='HS256')

def decode_jwt(token):
    try:
        # Use same JWT secret key for verification
        jwt_key = current_app.config.get('JWT_SECRET_KEY', current_app.config['SECRET_KEY'])
        payload = jwt.decode(token, jwt_key, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_token_from_header():
    """Extract JWT token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            token = auth_header.split(' ')[1]  # Bearer <token>
            return token
        except IndexError:
            return None
    return None

# Decorators
def token_required(f):
    """Decorator to require JWT token for protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        # Check if token is blacklisted (revoked)
        if token in BLACKLISTED_TOKENS:
            return jsonify({'error': 'Token has been revoked'}), 401
        
        payload = decode_jwt(token)
        if not payload:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        # Security monitoring for suspicious activity
        try:
            from authentication.security.security_monitor import security_monitor
            
            ip_address = request.remote_addr or 'unknown'
            user_agent = request.headers.get('User-Agent', 'unknown')
            user_id = payload['user_id']
            
            # Record this token usage
            security_monitor.record_token_usage(token, user_id, ip_address, user_agent)
            
            # Check for suspicious activity
            is_suspicious, reasons = security_monitor.detect_suspicious_activity(
                token, user_id, ip_address, user_agent
            )
            
            if is_suspicious:
                # Log suspicious activity
                print(f"🚨 SUSPICIOUS ACTIVITY - User {user_id}: {', '.join(reasons)}")
                
                # Optionally blacklist token immediately
                # blacklist.blacklist_token(token, reason=f"suspicious_activity: {reasons[0]}")
                # return jsonify({'error': 'Suspicious activity detected - token revoked'}), 401
                
                # Or just log and continue (less aggressive)
                
        except ImportError:
            pass  # Security monitoring not available
        
        # Add user info to request context
        request.current_user_id = payload['user_id']
        request.current_username = payload['username']
        request.current_token = token  # Store token for potential blacklisting
        
        return f(*args, **kwargs)
    return decorated

# News categories validation
VALID_CATEGORIES = [
    'general', 'business', 'entertainment', 'health', 
    'science', 'sports', 'technology', 'politics'
]

def validate_categories(categories):
    """Validate news categories"""
    if isinstance(categories, str):
        categories = [cat.strip() for cat in categories.split(',')]
    
    invalid_cats = [cat for cat in categories if cat not in VALID_CATEGORIES]
    if invalid_cats:
        return False, f"Invalid categories: {', '.join(invalid_cats)}"
    
    return True, categories 