"""
Train 4-Class Model with STRATIFIED validation split
Ensures balanced validation set!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split

print("="*60)
print("🐑 4-CLASS MODEL - STRATIFIED SPLIT")
print("="*60)

# Paths
TRAIN_DIR = r'C:\Users\QWEQWEQWEQWEQW\OneDrive\Documents\backup_5class_original\SheepHealthThesis\dataset\processed\train'
MODEL_SAVE_PATH = 'models/sheep_disease_4class_stratified.keras'

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

class_names = ['flystrike', 'healthy', 'orf', 'sheep_scab']

print("\n📁 Loading dataset with stratified split...")

# ✨ MANUALLY CREATE STRATIFIED SPLIT
image_paths = []
labels = []

for idx, class_name in enumerate(class_names):
    class_path = os.path.join(TRAIN_DIR, class_name)
    
    if not os.path.exists(class_path):
        continue
    
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img in images:
        image_paths.append(os.path.join(class_path, img))
        labels.append(idx)
    
    print(f"  ✅ {class_name:12} : {len(images)} images (label {idx})")

print(f"\nTotal: {len(image_paths)} images")

# ✨ STRATIFIED SPLIT - ensures balanced validation!
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, 
    labels, 
    test_size=0.2, 
    random_state=123,
    stratify=labels  # ← KEY! This ensures balanced split!
)

print("\n" + "="*60)
print("SPLIT VERIFICATION:")
print("="*60)

# Count distribution in each split
from collections import Counter

train_dist = Counter(train_labels)
val_dist = Counter(val_labels)

print("\nTraining set:")
for i, cls in enumerate(class_names):
    count = train_dist[i]
    pct = (count / len(train_labels) * 100)
    print(f"  {cls:12} : {count:3d} ({pct:5.1f}%)")

print("\nValidation set:")
for i, cls in enumerate(class_names):
    count = val_dist[i]
    pct = (count / len(val_labels) * 100)
    print(f"  {cls:12} : {count:3d} ({pct:5.1f}%)")

# Create TensorFlow datasets
def create_dataset(paths, labels, batch_size, training=True):
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_dataset(train_paths, train_labels, BATCH_SIZE, training=True)
val_ds = create_dataset(val_paths, val_labels, BATCH_SIZE, training=False)

print("\n" + "="*60)
print("BUILDING MODEL...")
print("="*60)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Preprocessing function
def preprocess(images, labels):
    images = data_augmentation(images, training=True)
    images = keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

# Apply preprocessing
train_ds = train_ds.map(preprocess)

# Validation preprocessing (no augmentation)
def preprocess_val(images, labels):
    images = keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

val_ds = val_ds.map(preprocess_val)

# Base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model created")

print("\n" + "="*60)
print("TRAINING...")
print("="*60)

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# Save
os.makedirs('models', exist_ok=True)
model.save(MODEL_SAVE_PATH)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.title('Stratified Split - Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Stratified Split - Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_4class_stratified.png')

# Results
final_acc = history.history['accuracy'][-1]
final_val = history.history['val_accuracy'][-1]

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"📊 Training Accuracy:   {final_acc*100:.2f}%")
print(f"📊 Validation Accuracy: {final_val*100:.2f}%")
print("="*60)

# Test predictions
print("\n" + "="*60)
print("PREDICTION TEST...")
print("="*60)

prediction_counts = {name: 0 for name in class_names}

for images, labels in val_ds.take(3):  # Test 3 batches
    predictions = model.predict(images, verbose=0)
    
    for i in range(len(images)):
        true_idx = labels[i].numpy()
        pred_idx = predictions[i].argmax()
        
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]
        confidence = predictions[i].max() * 100
        
        prediction_counts[pred_class] += 1
        
        match = "✅" if true_class == pred_class else "❌"
        print(f"{match} True: {true_class:12} | Pred: {pred_class:12} ({confidence:5.1f}%)")

print("\n" + "="*60)
print("PREDICTION DISTRIBUTION:")
print("="*60)

total = sum(prediction_counts.values())
for cls, count in prediction_counts.items():
    pct = (count / total * 100) if total > 0 else 0
    bar = "█" * int(pct / 3)
    print(f"{cls:12} : {count:2d} ({pct:5.1f}%) {bar}")

print("\n" + "="*60)
print("🎉 STRATIFIED MODEL COMPLETE!")
print("="*60)