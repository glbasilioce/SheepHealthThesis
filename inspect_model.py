"""
Inspect the saved model
Check if it's actually broken or if there's a code issue
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

print("="*60)
print("🔍 MODEL INSPECTION")
print("="*60)

# Load model
MODEL_PATH = 'models/sheep_disease_4class_basic.keras'
model = keras.models.load_model(MODEL_PATH)

print(f"\nModel: {MODEL_PATH}")
print(f"Total layers: {len(model.layers)}")

print("\n" + "="*60)
print("MODEL ARCHITECTURE:")
print("="*60)

for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")
    if hasattr(layer, 'trainable'):
        print(f"   Trainable: {layer.trainable}")

print("\n" + "="*60)
print("OUTPUT LAYER:")
print("="*60)

output_layer = model.layers[-1]
print(f"Type: {output_layer.__class__.__name__}")
print(f"Units: {output_layer.units if hasattr(output_layer, 'units') else 'N/A'}")
print(f"Activation: {output_layer.activation.__name__ if hasattr(output_layer.activation, '__name__') else output_layer.activation}")

print("\n" + "="*60)
print("MODEL SUMMARY:")
print("="*60)
model.summary()

print("\n" + "="*60)
print("TEST PREDICTION WITH DUMMY DATA:")
print("="*60)

# Create dummy image (all zeros)
dummy = np.zeros((1, 224, 224, 3))

print("\nPredicting on all-black image...")
pred = model.predict(dummy, verbose=0)

print(f"\nRaw output shape: {pred.shape}")
print(f"Raw output values:")
print(pred[0])
print(f"\nSum of probabilities: {pred[0].sum():.4f}")

# Check which class it predicts
class_names = ['flystrike', 'healthy', 'orf', 'sheep_scab']
predicted_idx = np.argmax(pred[0])
predicted_class = class_names[predicted_idx]
confidence = pred[0][predicted_idx] * 100

print(f"\nPredicted class: {predicted_class} (index {predicted_idx})")
print(f"Confidence: {confidence:.2f}%")

print("\n" + "="*60)
print("CHECKING EACH CLASS PROBABILITY:")
print("="*60)

for i, cls in enumerate(class_names):
    prob = pred[0][i] * 100
    bar = "█" * int(prob / 2)
    print(f"{cls:12} (idx {i}): {prob:6.2f}% {bar}")

print("\n" + "="*60)
print("WEIGHTS CHECK:")
print("="*60)

# Check if output layer has reasonable weights
output_weights = output_layer.get_weights()
if output_weights:
    print(f"Output layer has {len(output_weights)} weight matrices")
    print(f"Weight matrix shape: {output_weights[0].shape}")
    print(f"Bias shape: {output_weights[1].shape if len(output_weights) > 1 else 'None'}")
    
    if len(output_weights) > 1:
        print(f"\nBias values:")
        print(output_weights[1])
        
        # Check if one bias is much higher (would cause always predicting that class)
        bias_diff = output_weights[1].max() - output_weights[1].min()
        print(f"\nBias difference (max - min): {bias_diff:.4f}")
        
        if bias_diff > 2.0:
            print("⚠️  WARNING: Large bias difference detected!")
            print("   This could cause bias toward one class!")
        else:
            print("✅ Bias values seem reasonable")

print("\n" + "="*60)
print("DONE!")
print("="*60)