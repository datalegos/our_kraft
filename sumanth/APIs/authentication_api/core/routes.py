from flask import Blueprint, request, jsonify
from datetime import datetime
from models import db, User
from utils import (
    hash_password, verify_password, generate_jwt, token_required,
    validate_password, validate_email, validate_categories, VALID_CATEGORIES
)

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate input
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        is_valid, msg = validate_password(password)
        if not is_valid:
            return jsonify({'error': msg}), 400
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 409
        
        # Handle optional preferences
        preferred_language = data.get('preferred_language', 'en')
        preferred_categories = data.get('preferred_categories', 'general,technology,business')
        preferred_sources = data.get('preferred_sources', '')
        
        # Validate categories if provided
        if preferred_categories:
            is_valid, result = validate_categories(preferred_categories)
            if not is_valid:
                return jsonify({'error': result}), 400
            preferred_categories = ','.join(result)
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            preferred_language=preferred_language,
            preferred_categories=preferred_categories,
            preferred_sources=preferred_sources
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generate token
        token = generate_jwt(user.id, user.username)
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed', 'details': str(e)}), 500

@auth_bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password are required'}), 400
        
        username = data['username'].strip()
        password = data['password']
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if not user or not verify_password(password, user.password_hash):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Generate token
        token = generate_jwt(user.id, user.username)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Login failed', 'details': str(e)}), 500

@auth_bp.route('/api/logout', methods=['POST'])
@token_required
def logout():
    try:
        # Blacklist the current token to prevent reuse
        from authentication.core.utils import BLACKLISTED_TOKENS
        token = request.current_token
        BLACKLISTED_TOKENS.add(token)
        print(f"🚫 Token blacklisted for user logout")
        
        return jsonify({'message': 'Logged out successfully - token revoked'}), 200
    except ImportError:
        # Fallback: client-side logout only
        return jsonify({'message': 'Logged out successfully'}), 200

@auth_bp.route('/api/user/profile', methods=['GET'])
@token_required
def profile():
    try:
        user = User.query.get(request.current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get profile', 'details': str(e)}), 500

@auth_bp.route('/api/user/update', methods=['PUT'])
@token_required
def update_user():
    try:
        user = User.query.get(request.current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Update email
        if 'email' in data:
            email = data['email'].strip().lower()
            if not validate_email(email):
                return jsonify({'error': 'Invalid email format'}), 400
            
            # Check if email is already taken by another user
            existing_user = User.query.filter(User.email == email, User.id != user.id).first()
            if existing_user:
                return jsonify({'error': 'Email already in use'}), 409
            
            user.email = email
        
        # Update password
        if 'password' in data:
            is_valid, msg = validate_password(data['password'])
            if not is_valid:
                return jsonify({'error': msg}), 400
            user.password_hash = hash_password(data['password'])
        
        # Update preferences
        if 'preferred_language' in data:
            user.preferred_language = data['preferred_language']
        
        if 'preferred_categories' in data:
            categories = data['preferred_categories']
            is_valid, result = validate_categories(categories)
            if not is_valid:
                return jsonify({'error': result}), 400
            user.preferred_categories = ','.join(result)
        
        if 'preferred_sources' in data:
            sources = data['preferred_sources']
            if isinstance(sources, list):
                sources = ','.join(sources)
            user.preferred_sources = sources
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile', 'details': str(e)}), 500

@auth_bp.route('/api/user/preferences', methods=['GET'])
@token_required
def get_preferences():
    try:
        user = User.query.get(request.current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'preferences': {
                'language': user.preferred_language,
                'categories': user.get_categories_list(),
                'sources': user.get_sources_list()
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get preferences', 'details': str(e)}), 500

@auth_bp.route('/api/user/preferences', methods=['PUT'])
@token_required
def update_preferences():
    try:
        user = User.query.get(request.current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        if 'language' in data:
            user.preferred_language = data['language']
        
        if 'categories' in data:
            categories = data['categories']
            is_valid, result = validate_categories(categories)
            if not is_valid:
                return jsonify({'error': result}), 400
            user.preferred_categories = ','.join(result)
        
        if 'sources' in data:
            sources = data['sources']
            if isinstance(sources, list):
                sources = ','.join(sources)
            user.preferred_sources = sources
        
        db.session.commit()
        
        return jsonify({
            'message': 'Preferences updated successfully',
            'preferences': {
                'language': user.preferred_language,
                'categories': user.get_categories_list(),
                'sources': user.get_sources_list()
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to update preferences', 'details': str(e)}), 500

@auth_bp.route('/api/categories', methods=['GET'])
def get_available_categories():
    return jsonify({
        'categories': VALID_CATEGORIES
    }), 200

@auth_bp.route('/api/verify-token', methods=['GET'])
@token_required
def verify_token():
    return jsonify({
        'valid': True,
        'user_id': request.current_user_id,
        'username': request.current_username
    }), 200 