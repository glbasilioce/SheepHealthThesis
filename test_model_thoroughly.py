"""
Thoroughly Test 4-Class Model
Upload various sheep images and see predictions
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from pathlib import Path

# Load model
MODEL_PATH = 'models/sheep_disease_4class_stratified.keras'
model = keras.models.load_model(MODEL_PATH)

class_names = ['flystrike', 'healthy', 'orf', 'sheep_scab']

print("="*60)
print("🧪 COMPREHENSIVE MODEL TEST")
print("="*60)
print(f"\nModel: {MODEL_PATH}")
print(f"Classes: {class_names}")

# Test folder
TEST_FOLDER = input("\nEnter folder with test images (or press Enter for validation set): ").strip()

if not TEST_FOLDER:
    # Use validation images from training data
    TEST_FOLDER = r'C:\Users\QWEQWEQWEQWEQW\OneDrive\Documents\backup_5class_original\SheepHealthThesis\dataset\processed\train'
    print(f"\nUsing validation images from: {TEST_FOLDER}")
else:
    if not os.path.exists(TEST_FOLDER):
        print("❌ Folder not found!")
        exit()

print("\n" + "="*60)
print("TESTING PREDICTIONS...")
print("="*60)

# Track results
results = {cls: {'correct': 0, 'total': 0, 'predictions': []} for cls in class_names}
all_predictions = []

# Test images from each class
for true_class in class_names:
    class_folder = os.path.join(TEST_FOLDER, true_class)
    
    if not os.path.exists(class_folder):
        print(f"⚠️  Skipping {true_class} (folder not found)")
        continue
    
    # Get images
    images = [f for f in os.listdir(class_folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Test up to 20 images per class
    test_images = images[:20]
    
    print(f"\n{'='*60}")
    print(f"Testing {true_class.upper()} ({len(test_images)} images)")
    print(f"{'='*60}")
    
    for i, img_file in enumerate(test_images, 1):
        img_path = os.path.join(class_folder, img_file)
        
        try:
            # Load and preprocess
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            pred_class = class_names[np.argmax(predictions[0])]
            confidence = predictions[0].max() * 100
            
            # Track
            results[true_class]['total'] += 1
            if pred_class == true_class:
                results[true_class]['correct'] += 1
            
            results[true_class]['predictions'].append({
                'predicted': pred_class,
                'confidence': confidence,
                'correct': pred_class == true_class
            })
            
            all_predictions.append({
                'true': true_class,
                'predicted': pred_class,
                'confidence': confidence
            })
            
            # Print result
            match = "✅" if pred_class == true_class else "❌"
            print(f"{i:2d}. {match} Pred: {pred_class:12} ({confidence:5.1f}%) | True: {true_class}")
            
            # Show probabilities if wrong
            if pred_class != true_class:
                print("    Probabilities:")
                for j, cls in enumerate(class_names):
                    prob = predictions[0][j] * 100
                    print(f"      {cls:12} : {prob:5.1f}%")
        
        except Exception as e:
            print(f"{i:2d}. ❌ Error: {str(e)[:50]}")

# Summary
print("\n" + "="*60)
print("📊 DETAILED RESULTS BY CLASS")
print("="*60)

for cls in class_names:
    total = results[cls]['total']
    correct = results[cls]['correct']
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n{cls.upper()}:")
        print(f"  Tested: {total} images")
        print(f"  Correct: {correct} ({accuracy:.1f}%)")
        print(f"  Wrong: {total - correct}")
        
        # Show what it was misclassified as
        if total - correct > 0:
            misclassified = {}
            for pred in results[cls]['predictions']:
                if not pred['correct']:
                    wrong_class = pred['predicted']
                    misclassified[wrong_class] = misclassified.get(wrong_class, 0) + 1
            
            print("  Misclassified as:")
            for wrong_cls, count in misclassified.items():
                print(f"    → {wrong_cls}: {count} times")

# Overall accuracy
print("\n" + "="*60)
print("📊 OVERALL PERFORMANCE")
print("="*60)

total_tested = sum(r['total'] for r in results.values())
total_correct = sum(r['correct'] for r in results.values())

if total_tested > 0:
    overall_accuracy = (total_correct / total_tested) * 100
    print(f"\nTotal Images Tested: {total_tested}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Wrong Predictions:   {total_tested - total_correct}")
    print(f"\n🎯 OVERALL ACCURACY: {overall_accuracy:.2f}%")

# Prediction distribution
print("\n" + "="*60)
print("📊 PREDICTION DISTRIBUTION")
print("="*60)

pred_counts = {cls: 0 for cls in class_names}
for pred in all_predictions:
    pred_counts[pred['predicted']] += 1

print("\nWhat model predicted:")
for cls, count in pred_counts.items():
    pct = (count / total_tested * 100) if total_tested > 0 else 0
    bar = "█" * int(pct / 3)
    print(f"{cls:12} : {count:3d} ({pct:5.1f}%) {bar}")

# Confidence analysis
print("\n" + "="*60)
print("📊 CONFIDENCE ANALYSIS")
print("="*60)

confidences = [p['confidence'] for p in all_predictions]
if confidences:
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    print(f"\nAverage Confidence: {avg_confidence:.1f}%")
    print(f"Min Confidence:     {min_confidence:.1f}%")
    print(f"Max Confidence:     {max_confidence:.1f}%")
    
    # Confidence by correctness
    correct_confidences = [p['confidence'] for p in all_predictions 
                          if p['true'] == p['predicted']]
    wrong_confidences = [p['confidence'] for p in all_predictions 
                        if p['true'] != p['predicted']]
    
    if correct_confidences:
        avg_correct = sum(correct_confidences) / len(correct_confidences)
        print(f"\nAvg Confidence (Correct): {avg_correct:.1f}%")
    
    if wrong_confidences:
        avg_wrong = sum(wrong_confidences) / len(wrong_confidences)
        print(f"Avg Confidence (Wrong):   {avg_wrong:.1f}%")

print("\n" + "="*60)
print("✅ TESTING COMPLETE!")
print("="*60)

# Final verdict
if total_tested > 0 and overall_accuracy >= 90:
    print("\n🎉 MODEL PERFORMS WELL!")
    print(f"   {overall_accuracy:.1f}% accuracy is excellent!")
elif total_tested > 0 and overall_accuracy >= 85:
    print("\n✅ MODEL PERFORMS GOOD")
    print(f"   {overall_accuracy:.1f}% accuracy is acceptable")
elif total_tested > 0:
    print("\n⚠️  MODEL NEEDS IMPROVEMENT")
    print(f"   {overall_accuracy:.1f}% accuracy is below target")

print("="*60)