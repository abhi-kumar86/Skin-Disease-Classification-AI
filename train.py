"""
IMPROVED Skin Disease Classification Training Script
Fixes: Low accuracy, Class imbalance, Poor learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

# ==================== CONFIGURATION ====================
class Config:
    DATA_DIR = 'dataset'
    CLASSES = ['acne', 'hyperpigmentation', 'nail_psoriasis', 'sjs_ten', 'vitiligo']
    
    MODEL_DIR = 'models'
    OUTPUT_DIR = 'outputs'
    
    # IMPROVED PARAMETERS
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Smaller batch for better gradients
    EPOCHS = 50
    INITIAL_LR = 0.001  # Higher learning rate
    WARMUP_EPOCHS = 5
    
    RANDOM_SEED = 42

config = Config()

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

# GPU Config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except: pass
else:
    print("⚠️  CPU mode (slower)")

print("="*80)
print("IMPROVED SKIN DISEASE CLASSIFICATION")
print("="*80)
print(f"TensorFlow: {tf.__version__}")
print(f"Seed: {config.RANDOM_SEED}")
print("="*80)

# ==================== LOAD DATA ====================
print("\n[1] Loading Dataset...")

all_paths = []
all_labels = []

for cls in config.CLASSES:
    cls_path = os.path.join(config.DATA_DIR, cls)
    if not os.path.exists(cls_path):
        print(f"⚠️  {cls} not found!")
        continue
    
    images = [f for f in os.listdir(cls_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    for img in images:
        all_paths.append(os.path.join(cls_path, img))
        all_labels.append(cls)
    
    print(f"✅ {cls:20s}: {len(images):5d} images")

print(f"\n📊 Total: {len(all_paths)}")

if len(all_paths) == 0:
    print("❌ No images!")
    exit(1)

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)

# ==================== SPLIT ====================
print("\n[2] Splitting...")

X_train, X_temp, y_train, y_temp = train_test_split(
    all_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=config.RANDOM_SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=config.RANDOM_SEED
)

print(f"Train: {len(X_train)} ({len(X_train)/len(all_paths)*100:.1f}%)")
print(f"Val:   {len(X_val)} ({len(X_val)/len(all_paths)*100:.1f}%)")
print(f"Test:  {len(X_test)} ({len(X_test)/len(all_paths)*100:.1f}%)")

# ==================== CLASS WEIGHTS ====================
print("\n[3] Computing Class Weights...")

unique_classes = np.unique(y_train)
class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y_train)

class_weight_dict = {}
for idx, cls in enumerate(config.CLASSES):
    cls_idx = np.where(unique_classes == cls)[0]
    if len(cls_idx) > 0:
        class_weight_dict[idx] = class_weights_array[cls_idx[0]]

print("Weights:")
for idx, cls in enumerate(config.CLASSES):
    print(f"  {cls:20s}: {class_weight_dict.get(idx, 1.0):.2f}")

# ==================== DATA GENERATORS ====================
print("\n[4] Creating Generators...")

# IMPROVED AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # More rotation
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,  # Added shear
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],  # More brightness variation
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_df = pd.DataFrame({'filepath': X_train, 'class': y_train})
val_df = pd.DataFrame({'filepath': X_val, 'class': y_val})
test_df = pd.DataFrame({'filepath': X_test, 'class': y_test})

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='filepath', y_col='class',
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    classes=config.CLASSES,
    shuffle=True,
    seed=config.RANDOM_SEED
)

val_gen = val_test_datagen.flow_from_dataframe(
    val_df, x_col='filepath', y_col='class',
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    classes=config.CLASSES,
    shuffle=False
)

test_gen = val_test_datagen.flow_from_dataframe(
    test_df, x_col='filepath', y_col='class',
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    classes=config.CLASSES,
    shuffle=False
)

print(f"✅ Generators ready: {len(train_gen)} batches")

# ==================== BUILD MODEL ====================
print("\n[5] Building Model...")

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
)

# IMPORTANT: Unfreeze top layers for fine-tuning
base_model.trainable = True

# Freeze first 100 layers, unfreeze rest
for layer in base_model.layers[:100]:
    layer.trainable = False

for layer in base_model.layers[100:]:
    layer.trainable = True

print(f"Trainable layers: {sum([1 for l in base_model.layers if l.trainable])}")

# Build model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(config.CLASSES), activation='softmax')
], name='ImprovedSkinClassifier')

# Compile with higher learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
)

print("\n📊 Model Summary:")
model.summary()

total_params = model.count_params()
trainable_params = sum([int(np.prod(w.shape)) for w in model.trainable_weights])
print(f"\nTotal: {total_params:,}")
print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

# ==================== CALLBACKS ====================
print("\n[6] Setting Callbacks...")

def lr_schedule(epoch, lr):
    """Warmup + Decay"""
    if epoch < config.WARMUP_EPOCHS:
        return config.INITIAL_LR * (epoch + 1) / config.WARMUP_EPOCHS
    else:
        return config.INITIAL_LR * np.exp(0.1 * (config.WARMUP_EPOCHS - epoch))

callbacks = [
    LearningRateScheduler(lr_schedule, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(config.MODEL_DIR, 'best_model.h5'), 
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ==================== TRAIN ====================
print("\n[7] Training...")
print("="*80)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=config.EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n✅ Training Done!")

# ==================== PLOT HISTORY ====================
print("\n[8] Plotting...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(history.history['accuracy'], 'o-', label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], 's-', label='Val', linewidth=2)
axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(history.history['loss'], 'o-', label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], 's-', label='Val', linewidth=2)
axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(history.history['precision'], 'o-', label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], 's-', label='Val', linewidth=2)
axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(history.history['recall'], 'o-', label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], 's-', label='Val', linewidth=2)
axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, 'training_history.png'), dpi=300)
print(f"✅ Saved: training_history.png")
plt.close()

# ==================== EVALUATE ====================
print("\n[9] Evaluating...")

test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

test_acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")

print("\n📋 Classification Report:")
print("="*80)
report = classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4)
print(report)

with open(os.path.join(config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("IMPROVED MODEL RESULTS\n" + "="*80 + "\n\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
    f.write(report)

# ==================== CONFUSION MATRIX ====================
print("\n[10] Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=config.CLASSES, yticklabels=config.CLASSES,
            cbar_kws={'label': 'Count'}, linewidths=0.5)
plt.title('Confusion Matrix (Improved Model)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True', fontsize=12, fontweight='bold')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
print(f"✅ Saved: confusion_matrix.png")
plt.close()

# ==================== SAVE ====================
print("\n[11] Saving...")

model.save(os.path.join(config.MODEL_DIR, 'final_model.h5'))
print(f"✅ Saved: final_model.h5")

model_info = {
    'test_accuracy': float(test_acc),
    'total_params': int(total_params),
    'trainable_params': int(trainable_params),
    'epochs': len(history.history['loss']),
    'best_val_acc': float(max(history.history['val_accuracy'])),
    'classes': config.CLASSES,
    'img_size': config.IMG_SIZE,
    'batch_size': config.BATCH_SIZE,
    'initial_lr': config.INITIAL_LR
}

with open(os.path.join(config.MODEL_DIR, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=4)
print(f"✅ Saved: model_info.json")

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"\n📊 Data: {len(all_paths)} images")
print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"\n🏗️  Model: EfficientNetB0 (Fine-tuned)")
print(f"   Total params: {total_params:,}")
print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"\n📈 Training:")
print(f"   Epochs: {len(history.history['loss'])}")
print(f"   Best val acc: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"   Final train acc: {history.history['accuracy'][-1]*100:.2f}%")
print(f"\n🎯 Test: {test_acc*100:.2f}%")
print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
print("\nNext: python app.py")