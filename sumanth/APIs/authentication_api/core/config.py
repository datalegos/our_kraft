# config.py
# Configuration for News Aggregator Authentication API
# Using environment variables for security and flexibility

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration for the authentication API."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database Configuration
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'news_auth_db')
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = 24 * 60 * 60  # 24 hours in seconds
    
    # API Configuration
    API_TITLE = "News Aggregator Authentication API"
    API_VERSION = "1.0.0"
    
    # Security Configuration
    BCRYPT_LOG_ROUNDS = 12
    
    @staticmethod
    def validate_config():
        """Validate that required configuration is present."""
        required_vars = ['SECRET_KEY', 'DB_HOST', 'DB_USER', 'DB_NAME']
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var) and not hasattr(Config, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Override with more secure defaults for production
    BCRYPT_LOG_ROUNDS = 15

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    DB_NAME = 'news_auth_test_db'
    BCRYPT_LOG_ROUNDS = 4  # Faster for tests

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}