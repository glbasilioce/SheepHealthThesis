"""
Disease Prediction Module
"""

from tensorflow import keras
import numpy as np
from PIL import Image
import os

# ✨ NEW MODEL PATH - Stratified 4-class model
MODEL_PATH = 'models/sheep_disease_4class_stratified.keras'

# ✨ UPDATED CLASS NAMES (4 classes, no ringworm)
CLASS_NAMES = ['Flystrike', 'Healthy', 'Orf', 'Sheep Scab']

# Load model (once at startup)
print(f"Loading model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded successfully!")
print(f"Classes: {CLASS_NAMES}")

# ✨ Disease Information Dictionary
DISEASE_INFO = {
    'Flystrike': {
        'name': 'Flystrike (Myiasis)',
        'description': 'A condition caused by flies laying eggs in the sheep\'s wool, leading to maggot infestation.',
        'symptoms': 'Discolored wool, restlessness, wool loss, visible maggots, foul odor',
        'treatment': 'Immediate veterinary attention required. Remove maggots, clean wound, apply insecticide',
        'severity': 'High - Urgent',
        'color': 'danger'
    },
    'Healthy': {
        'name': 'Healthy Sheep',
        'description': 'No disease detected. The sheep appears to be in good health.',
        'symptoms': 'Normal behavior, clean wool, no visible lesions',
        'treatment': 'Continue regular health monitoring and preventive care',
        'severity': 'None',
        'color': 'success'
    },
    'Orf': {
        'name': 'Orf (Contagious Ecthyma)',
        'description': 'A viral infection causing crusty scabs, primarily around the mouth and lips.',
        'symptoms': 'Crusty scabs on mouth/lips, difficulty eating, lesions on udder',
        'treatment': 'Usually self-limiting. Keep affected areas clean, isolate infected sheep, vaccination available',
        'severity': 'Moderate',
        'color': 'warning'
    },
    'Sheep Scab': {
        'name': 'Sheep Scab (Psoroptic Mange)',
        'description': 'A parasitic infection caused by mites, leading to intense itching and wool loss.',
        'symptoms': 'Intense itching/scratching, wool loss, crusty skin, yellow discharge',
        'treatment': 'Injectable or topical acaricides, quarantine affected sheep, treat entire flock',
        'severity': 'Moderate to High',
        'color': 'warning'
    }
}

def predict_disease(image_path):
    """
    Predict disease from sheep image
    
    Args:
        image_path: Path to uploaded image
        
    Returns:
        dict with prediction results
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Expand dimensions and preprocess
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Get predicted class
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx] * 100)
        
        # Get all predictions (sorted by confidence)
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i] * 100)
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort by confidence (highest first)
        all_predictions = dict(
            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Get disease info
        disease_info = DISEASE_INFO.get(predicted_class, DISEASE_INFO['Healthy'])
        
        return {
            'predicted_class': predicted_class,
            'disease_name': disease_info['name'],
            'confidence': confidence,
            'all_predictions': all_predictions,
            'disease_info': disease_info
        }
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")