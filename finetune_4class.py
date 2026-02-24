"""
Fine-Tune 4-Class Model
Unfreeze and retrain with lower learning rate
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

print("="*60)
print("🔧 ATTEMPT 2: FINE-TUNING 4-CLASS MODEL")
print("="*60)

# Paths
TRAIN_DIR = r'C:\Users\QWEQWEQWEQWEQW\OneDrive\Documents\backup_5class_original\SheepHealthThesis\dataset\processed\train'
EXISTING_MODEL = 'models/sheep_disease_4class_basic.keras'
FINETUNED_MODEL = 'models/sheep_disease_4class_finetuned.keras'

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
FINETUNE_EPOCHS = 20
FINETUNE_LR = 0.00001  # Much lower!

class_names = ['flystrike', 'healthy', 'orf', 'sheep_scab']

print("\n📁 Loading model from Attempt 1...")
model = keras.models.load_model(EXISTING_MODEL)
print("✅ Loaded")

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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n" + "="*60)
print("UNFREEZING BASE MODEL...")
print("="*60)

# Find base model
base_model = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break

if base_model:
    print(f"✅ Found: {base_model.name}")
    
    # Unfreeze last layers
    freeze_until = 100
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
    
    print(f"🔓 Unfroze {len(base_model.layers) - freeze_until} layers")
else:
    print("⚠️  Base model not found, will train all layers")

print("\n" + "="*60)
print("RECOMPILING & FINE-TUNING...")
print("="*60)

# Recompile with lower LR
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=[early_stop]
)

# Save
model.save(FINETUNED_MODEL)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.title('Fine-Tuning Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Fine-Tuning Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_4class_finetuned.png')

# Results
final_acc = history.history['accuracy'][-1]
final_val = history.history['val_accuracy'][-1]

print("\n" + "="*60)
print("ATTEMPT 2 COMPLETE!")
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
if final_val >= 0.92 and max(prediction_counts.values()) < 10:
    print("✅ GOOD! Use this model!")
else:
    print("⚠️  Still issues. Try ResNet50 (Script 3)")
print("="*60)