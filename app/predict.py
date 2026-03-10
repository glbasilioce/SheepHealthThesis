"""
Disease Prediction Module
"""

import os
import numpy as np
from PIL import Image
import keras
import requests

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sheep_disease_4class_stratified.keras')

# Google Drive file ID
GDRIVE_FILE_ID = '1UqAnizs0WPHK74rrs-jfxz_PvGPl5wZr'

# ✨ UPDATED CLASS NAMES (4 classes, no ringworm)
CLASS_NAMES = ['Flystrike', 'Healthy', 'Orf', 'Sheep Scab']

# ✨ Disease Information Dictionary
DISEASE_INFO = {
    'Flystrike': {
        'name': 'Flystrike (Myiasis)',
        'description': 'A condition caused by flies laying eggs in the sheep\'s wool, leading to maggot infestation.',
        'symptoms': [
            'Discolored wool',
            'Restlessness',
            'Wool loss',
            'Visible maggots',
            'Foul odor'
        ],
        'treatment': [
            'Immediate veterinary attention required',
            'Remove maggots manually',
            'Clean wound thoroughly',
            'Apply insecticide as prescribed'
        ],
        'prevention': [
            'Regular shearing and crutching',
            'Daily inspection during fly season',
            'Keep sheep dry and clean',
            'Fly control measures',
            'Tail docking in lambs'
        ],
        'severity': 'High - Urgent',
        'contagious': 'Not Contagious',
        'color': 'danger'
    },
    
    'Healthy': {
        'name': 'Healthy Sheep',
        'description': 'No disease detected. The sheep appears to be in good health.',
        'symptoms': [
            'Normal behavior',
            'Clean wool',
            'No visible lesions',
            'Good appetite',
            'Active movement'
        ],
        'treatment': [
            'Continue regular health monitoring',
            'Maintain preventive care schedule',
            'Ensure proper nutrition and clean water',
            'Keep vaccination records up to date'
        ],
        'prevention': [
            'Regular health checkups',
            'Proper nutrition and clean water',
            'Good hygiene and shelter',
            'Vaccination schedule',
            'Parasite control program'
        ],
        'severity': 'None',
        'contagious': 'Not Applicable',
        'color': 'success'
    },
    
    'Orf': {
        'name': 'Orf (Contagious Ecthyma)',
        'description': 'A viral infection causing crusty scabs, primarily around the mouth and lips.',
        'symptoms': [
            'Crusty scabs on mouth/lips',
            'Difficulty eating',
            'Lesions on udder',
            'Weight loss',
            'Painful sores'
        ],
        'treatment': [
            'Usually self-limiting (heals in 2-4 weeks)',
            'Keep affected areas clean',
            'Isolate infected sheep',
            'Provide soft food and clean water',
            'Antibiotics for secondary infections if needed'
        ],
        'prevention': [
            'Vaccination before exposure',
            'Isolate new or infected sheep',
            'Disinfect feeding equipment',
            'Wear gloves when handling (zoonotic)',
            'Quarantine infected animals'
        ],
        'severity': 'Moderate',
        'contagious': 'Highly Contagious (including to humans)',
        'color': 'warning'
    },
    
    'Sheep Scab': {
        'name': 'Sheep Scab (Psoroptic Mange)',
        'description': 'A parasitic infection caused by mites, leading to intense itching and wool loss.',
        'symptoms': [
            'Intense itching/scratching',
            'Wool loss',
            'Crusty skin',
            'Yellow discharge',
            'Restlessness and weight loss'
        ],
        'treatment': [
            'Veterinary-prescribed acaricide treatment',
            'Complete isolation of infected animals',
            'Treat entire flock if outbreak occurs',
            'Repeat treatment as directed by vet',
            'Disinfect equipment and housing'
        ],
        'prevention': [
            'Regular inspection of flock',
            'Quarantine new animals (17+ days)',
            'Avoid contact with infected flocks',
            'Monitor for signs after exposure',
            'Report suspected cases (notifiable disease)'
        ],
        'severity': 'High - Requires veterinary treatment',
        'contagious': 'Highly Contagious',
        'color': 'danger'
    }
}

def download_model_from_gdrive():
    """Download model from Google Drive if not exists"""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return
    
    print(f"Downloading model from Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    
    # Download the file
    response = requests.get(url, stream=True)
    
    # Handle large file download confirmation
    if 'download_warning' in response.text or response.status_code != 200:
        # Try alternative method for large files
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check for virus scan warning
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': GDRIVE_FILE_ID, 'confirm': value}
                response = session.get(url, params=params, stream=True)
                break
    
    # Save the file
    total_size = 0
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
    
    print(f"Model downloaded successfully! Size: {total_size / (1024*1024):.2f} MB")

# Download model if needed
download_model_from_gdrive()

# Load model (once at startup)
print(f"Loading model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded successfully!")
print(f"Classes: {CLASS_NAMES}")

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
