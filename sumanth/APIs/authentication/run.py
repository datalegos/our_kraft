#!/usr/bin/env python3
"""
News Aggregator Authentication API
Run this file to start the authentication service
"""

import os
import sys
sys.path.append('core')
from core.app import create_app

# Create Flask app
app = create_app()

if __name__ == '__main__':
    print("🚀 Starting News Aggregator Authentication API...")
    print("📍 API will be available at: http://localhost:5000")
    print("📋 API documentation at: http://localhost:5000/api/info")
    print("❤️  Health check at: http://localhost:5000/health")
    
    app.run(
        debug=True, 
        host='0.0.0.0',  # Allow external connections
        port=5000
    )