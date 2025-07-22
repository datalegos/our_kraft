import os
from flask import Flask
from flask_cors import CORS
from config import config
from models import db
from routes import auth_bp

def create_app(config_name=None):
    """Application factory pattern for creating Flask app."""
    
    # Determine configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Validate configuration
    try:
        config[config_name].validate_config()
    except ValueError as e:
        print(f"⚠️  Configuration Warning: {e}")
    
    # Configure SQLAlchemy database URI
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        f"mysql+pymysql://{app.config['DB_USER']}:{app.config['DB_PASSWORD']}"
        f"@{app.config['DB_HOST']}/{app.config['DB_NAME']}"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)  # Enable CORS for frontend integration
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    
    # Create tables
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"⚠️  Database setup warning: {e}")
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {
            'status': 'healthy', 
            'service': 'authentication-api',
            'version': app.config.get('API_VERSION', '1.0.0')
        }, 200
    
    # API info endpoint
    @app.route('/api/info')
    def api_info():
        return {
            'service': app.config.get('API_TITLE', 'News Aggregator Authentication API'),
            'version': app.config.get('API_VERSION', '1.0.0'),
            'environment': config_name,
            'endpoints': {
                'auth': {
                    'register': 'POST /api/register',
                    'login': 'POST /api/login',
                    'logout': 'POST /api/logout',
                    'profile': 'GET /api/user/profile',
                    'update_profile': 'PUT /api/user/update',
                    'get_preferences': 'GET /api/user/preferences',
                    'update_preferences': 'PUT /api/user/preferences',
                    'verify_token': 'GET /api/verify-token'
                },
                'utility': {
                    'categories': 'GET /api/categories',
                    'health': 'GET /health',
                    'info': 'GET /api/info'
                }
            },
            'documentation': 'See API_DOCUMENTATION.md for detailed usage'
        }, 200
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Endpoint not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app