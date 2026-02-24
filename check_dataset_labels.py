"""
Check if dataset labels are correct
"""

import tensorflow as tf
import os

TRAIN_DIR = r'C:\Users\QWEQWEQWEQWEQW\OneDrive\Documents\backup_5class_original\SheepHealthThesis\dataset\processed\train'

print("="*60)
print("🔍 DATASET LABEL CHECK")
print("="*60)

class_names = ['flystrike', 'healthy', 'orf', 'sheep_scab']

# Load dataset like training script
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=16,
    label_mode='int',
    class_names=class_names
)

print(f"\nDataset class names: {train_ds.class_names}")

# Check a batch
for images, labels in train_ds.take(1):
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label values: {labels.numpy()[:10]}")
    
    print("\nFirst 10 samples:")
    for i in range(min(10, len(labels))):
        label_idx = labels[i].numpy()
        class_name = class_names[label_idx]
        print(f"  Sample {i}: Label {label_idx} = {class_name}")

# Count labels in entire dataset
print("\n" + "="*60)
print("LABEL DISTRIBUTION IN TRAINING SET:")
print("="*60)

label_counts = {i: 0 for i in range(4)}

for images, labels in train_ds:
    for label in labels.numpy():
        label_counts[label] += 1

total = sum(label_counts.values())

for i, cls in enumerate(class_names):
    count = label_counts[i]
    pct = (count / total * 100) if total > 0 else 0
    print(f"{cls:12} (label {i}): {count:3d} images ({pct:5.1f}%)")

print(f"\nTotal: {total} images")