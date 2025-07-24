# auth_api/routes.py
# This file defines the API endpoints (routes) for authentication.
# It uses a Flask Blueprint to organize the routes.

from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import mysql.connector
from .database import get_db_connection
from .utils import token_required

# A Blueprint is a way to organize a group of related views and other code.
# Instead of registering views and other code directly with an application,
# they are registered with a blueprint.
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/register', methods=['POST'])
def register_user():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Username and password are required.'}), 400

    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    username = data['username']

    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500
    
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        user_id = cursor.lastrowid
        return jsonify({'message': 'User registered successfully.', 'userId': user_id}), 201
    except mysql.connector.Error as err:
        if err.errno == 1062:  # Duplicate entry
            return jsonify({'message': 'Conflict: Username already exists.'}), 409
        return jsonify({'message': f'Database error: {err}'}), 500
    finally:
        cursor.close()
        conn.close()

@auth_bp.route('/api/login', methods=['POST'])
def login_user():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Username and password are required.'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (data['username'],))
    user = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if not user or not check_password_hash(user['password'], data['password']):
        return jsonify({'message': 'Unauthorized: Incorrect username or password.'}), 401

    token = jwt.encode({
        'id': user['id'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, current_app.config['SECRET_KEY'], algorithm="HS256")

    return jsonify({'message': 'Login successful.', 'token': token})

@auth_bp.route('/api/logout', methods=['POST'])
def logout():
    return jsonify({'message': 'Logout successful. Please delete the token on the client side.'})

@auth_bp.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile(current_user_id):
    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username, language, categories FROM users WHERE id = %s", (current_user_id,))
    profile = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if not profile:
        return jsonify({'message': 'User profile not found.'}), 404

    return jsonify(profile)

@auth_bp.route('/api/user/update', methods=['PUT'])
@token_required
def update_user_profile(current_user_id):
    data = request.get_json()
    if not data or ('language' not in data and 'categories' not in data):
        return jsonify({'message': 'No update data provided (language or categories).'}), 400

    fields = []
    params = []
    if 'language' in data:
        fields.append("language = %s")
        params.append(data['language'])
    if 'categories' in data:
        fields.append("categories = %s")
        params.append(data['categories'])
    
    params.append(current_user_id)
    
    sql = f"UPDATE users SET {', '.join(fields)} WHERE id = %s"

    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    cursor = conn.cursor()
    cursor.execute(sql, tuple(params))
    conn.commit()
    changes = cursor.rowcount
    
    cursor.close()
    conn.close()

    return jsonify({'message': 'User preferences updated successfully.', 'changes': changes})
    