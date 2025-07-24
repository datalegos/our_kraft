# config.py
# This file contains the configuration variables for the Flask app.
# Using a class-based configuration allows for different environments (e.g., Development, Production).

import os

class Config:
    """Base configuration."""
    # SECRET_KEY is used by Flask for session management and signing.
    # It's crucial to keep this value secret.
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_super_secret_key_change_this')

    # --- Database Configuration ---
    # It's best practice to use environment variables for sensitive data.
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_USER = os.environ.get('DB_USER', 'your_mysql_user') # Replace with your MySQL username
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_mysql_password') # Replace with your MySQL password
    DB_NAME = os.environ.get('DB_NAME', 'auth_api_db')
