"""
Train 4-Class Model with ResNet50
Different architecture, might work better!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50  # ✨ Different!
import matplotlib.pyplot as plt
import os

print("="*60)
print("🔄 ATTEMPT 3: RESNET50 ARCHITECTURE")
print("="*60)

# Paths
TRAIN_DIR = r'C:\Users\QWEQWEQWEQWEQW\OneDrive\Documents\backup_5class_original\SheepHealthThesis\dataset\processed\train'
MODEL_SAVE_PATH = 'models/sheep_disease_4class_resnet.keras'

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

class_names = ['flystrike', 'healthy', 'orf', 'sheep_scab']

print("\n📁 Loading data...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=class_names
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    class_names=class_names
)

print(f"✅ Classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n" + "="*60)
print("BUILDING RESNET50 MODEL...")
print("="*60)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
])

# ✨ ResNet50 instead of MobileNetV2
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = keras.applications.resnet50.preprocess_input(x)  # ✨ Different preprocessing
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)  # Larger dense layer
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ ResNet50-based model")
print(f"📊 Parameters: {model.count_params():,}")

print("\n" + "="*60)
print("TRAINING...")
print("="*60)

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
plt.title('ResNet50 Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('ResNet50 Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_4class_resnet.png')

# Results
final_acc = history.history['accuracy'][-1]
final_val = history.history['val_accuracy'][-1]

print("\n" + "="*60)
print("ATTEMPT 3 COMPLETE!")
print("="*60)
print(f"📊 Validation Accuracy: {final_val*100:.2f}%")
print("="*60)

# Test
print("\n" + "="*60)
print("PREDICTION TEST...")
print("="*60)

prediction_counts = {name: 0 for name in class_names}

for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    
    for i in range(min(15, len(images))):
        true_class = class_names[labels[i]]
        pred_class = class_names[predictions[i].argmax()]
        confidence = predictions[i].max() * 100
        
        prediction_counts[pred_class] += 1
        
        match = "✅" if true_class == pred_class else "❌"
        print(f"{i+1:2d}. {match} True: {true_class:12} | Pred: {pred_class:12} ({confidence:5.1f}%)")

print("\n" + "="*60)
print("DISTRIBUTION:")
print("="*60)

total = sum(prediction_counts.values())
for cls, count in prediction_counts.items():
    pct = (count / total * 100) if total > 0 else 0
    bar = "█" * int(pct / 5)
    print(f"{cls:12} : {count:2d} ({pct:5.1f}%) {bar}")

print("\n" + "="*60)
print("🎉 ALL ATTEMPTS COMPLETE!")
print("="*60)