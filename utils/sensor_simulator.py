"""
Sensor Simulator for Health Monitoring
Simulates realistic sheep health parameters
"""

import random
from datetime import datetime


class SensorSimulator:
    """Simulates health monitoring sensor data"""
    
    def __init__(self):
        self.health_state = 'normal'
        
    def set_health_state(self, state):
        """Set health state: normal, fever, respiratory, critical"""
        self.health_state = state
        
    def get_reading(self):
        """Generate simulated sensor readings"""
        
        if self.health_state == 'normal':
            temp = random.uniform(38.5, 39.5)
            hr = random.randint(70, 90)
            spo2 = random.randint(96, 100)
            activity = random.uniform(1.0, 1.5)
            
        elif self.health_state == 'fever':
            temp = random.uniform(40.0, 41.0)
            hr = random.randint(90, 110)
            spo2 = random.randint(94, 98)
            activity = random.uniform(0.5, 0.9)
            
        elif self.health_state == 'respiratory':
            temp = random.uniform(39.0, 40.0)
            hr = random.randint(85, 105)
            spo2 = random.randint(88, 94)
            activity = random.uniform(0.6, 1.0)
            
        elif self.health_state == 'critical':
            temp = random.uniform(40.5, 42.0)
            hr = random.randint(100, 120)
            spo2 = random.randint(85, 92)
            activity = random.uniform(0.3, 0.7)
            
        else:  # Default to normal
            temp = random.uniform(38.5, 39.5)
            hr = random.randint(70, 90)
            spo2 = random.randint(96, 100)
            activity = random.uniform(1.0, 1.5)
        
        # Environmental sensors
        ambient_temp = random.uniform(20, 35)
        humidity = random.uniform(40, 80)
        
        return {
            'heart_rate': hr,
            'body_temp': round(temp, 1),
            'spo2': spo2,
            'activity_level': round(activity, 2),
            'ambient_temp': round(ambient_temp, 1),
            'humidity': round(humidity, 1),
            'timestamp': datetime.now().isoformat(),
            'source': 'simulator',
            'status': 'simulated',
            'sheep_id': 'sheep_001'
        }