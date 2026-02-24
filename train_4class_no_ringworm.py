"""
Train 5-Class Model with Better Parameters
Test if better training fixes the ringworm bias
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

print("="*60)
print("🐑 5-CLASS SHEEP DISEASE MODEL (BETTER TRAINING)")
print("="*60)

# Paths
TRAIN_DIR = r'C:\Users\QWEQWEQWEQWEQW\OneDrive\Documents\backup_5class_original\SheepHealthThesis\dataset\processed\train'
MODEL_SAVE_PATH = 'models/sheep_disease_5class_improved.keras'

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

print("\n📁 Scanning dataset...")

# Auto-detect all classes
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
print(f"\n✅ Classes: {class_names}")
print(f"✅ Number of classes: {len(class_names)}")

# Optimize
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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

# Base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model with {len(class_names)} classes")
print(f"📊 Parameters: {model.count_params():,}")

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
print(f"\n✅ Model saved: {MODEL_SAVE_PATH}")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.title('Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_5class_improved.png')

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
print("TESTING PREDICTIONS...")
print("="*60)

for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    
    print("\nSample predictions (first 10):")
    for i in range(min(10, len(images))):
        true_class = class_names[labels[i]]
        pred_class = class_names[predictions[i].argmax()]
        confidence = predictions[i].max() * 100
        
        # Show all class predictions for this image
        print(f"\n{i+1}. True: {true_class}")
        for j, cls in enumerate(class_names):
            prob = predictions[i][j] * 100
            print(f"   {cls:15} : {prob:5.1f}%")

print("\n" + "="*60)
print("DONE!")
print("="*60)