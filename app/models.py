"""
Database Models
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Prediction(db.Model):
    """Store prediction history"""
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    all_predictions = db.Column(db.JSON)  # Store all class probabilities
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

class HealthReading(db.Model):
    """Store health monitoring readings"""
    
    id = db.Column(db.Integer, primary_key=True)
    sheep_id = db.Column(db.String(50), nullable=False)
    
    # Vital signs
    heart_rate = db.Column(db.Float, nullable=False)
    body_temp = db.Column(db.Float, nullable=False)
    spo2 = db.Column(db.Float, nullable=False)
    activity_level = db.Column(db.Float, nullable=False)
    
    # Environmental
    ambient_temp = db.Column(db.Float)
    humidity = db.Column(db.Float)
    
    # Health assessment
    ohi_score = db.Column(db.Float, nullable=False)
    health_status = db.Column(db.String(20), nullable=False)
    
    # Timestamp
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'sheep_id': self.sheep_id,
            'heart_rate': self.heart_rate,
            'body_temp': self.body_temp,
            'spo2': self.spo2,
            'activity_level': self.activity_level,
            'ambient_temp': self.ambient_temp,
            'humidity': self.humidity,
            'ohi_score': self.ohi_score,
            'health_status': self.health_status,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }