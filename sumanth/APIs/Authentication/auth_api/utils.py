# auth_api/utils.py
# This file contains helper functions and decorators.

from functools import wraps
from flask import request, jsonify, current_app
import jwt

def token_required(f):
    """Decorator to protect routes that require a valid JWT."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'authorization' in request.headers:
            try:
                token = request.headers['authorization'].split(' ')[1]
            except IndexError:
                return jsonify({'message': 'Unauthorized: Malformed token header.'}), 401


        if not token:
            return jsonify({'message': 'Unauthorized: Token is missing!'}), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Forbidden: Token has expired!'}), 403
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Forbidden: Token is invalid!'}), 403

        return f(current_user_id, *args, **kwargs)
    return decorated
