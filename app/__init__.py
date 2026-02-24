"""
Flask Application Initialization
"""

from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'sheep-disease-detection-secret-key-2024'
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    
    # ✨ NEW: Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # ✨ NEW: Initialize database
    from app.models import db
    db.init_app(app)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    # Register routes
    from app import routes
    app.register_blueprint(routes.bp)
    
    return app