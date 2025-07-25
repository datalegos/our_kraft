from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # News aggregator specific fields
    preferred_language = db.Column(db.String(10), default='en')
    preferred_categories = db.Column(db.Text, default='general,technology,business')
    preferred_sources = db.Column(db.Text, nullable=True)
    
    # User management
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'preferred_language': self.preferred_language,
            'preferred_categories': self.preferred_categories.split(',') if self.preferred_categories else [],
            'preferred_sources': self.preferred_sources.split(',') if self.preferred_sources else [],
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def get_categories_list(self):
        return self.preferred_categories.split(',') if self.preferred_categories else []
    
    def get_sources_list(self):
        return self.preferred_sources.split(',') if self.preferred_sources else [] 