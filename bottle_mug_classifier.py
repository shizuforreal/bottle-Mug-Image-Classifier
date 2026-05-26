# -*- coding: utf-8 -*-
"""
=============================================================================
  BOTTLE vs MUG IMAGE CLASSIFIER
  Haar Feature Extraction + SVM (Support Vector Machine)
  Author: shizuforreal
  Dataset: 100 bottle images + 100 mug images (200 total)
  Accuracy: 72.50%
=============================================================================
"""

import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import haar_like_feature, haar_like_feature_coord

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "/content/ds/ds"   # Folder with subfolders: /bottles, /mugs
IMG_SIZE = (64, 64)                # All images resized to this
NUM_FEATURES = 128                 # Number of Haar features to extract
MODEL_OUTPUT_PATH = "svm_classifier.pkl"

# ============================================================================
# STEP 1: LOAD AND EXTRACT FEATURES
# ============================================================================

print("=" * 60)
print("  BOTTLE vs MUG CLASSIFIER — Haar + SVM")
print("=" * 60)
print("\n[STEP 1/4] Loading images and extracting Haar features...")

data = []
labels = []
skipped = 0

# Pre-compute Haar feature coordinates once (more efficient)
coords, types = haar_like_feature_coord(
    width=IMG_SIZE[0],
    height=IMG_SIZE[1],
    feature_type=['type-2-x']
)
coords = coords[:NUM_FEATURES]
types = types[:NUM_FEATURES]

for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)

    # BUG FIX: indentation — this must be inside the for loop
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            skipped += 1
            continue

        # BUG FIX: resize every image to ensure consistent dimensions
        img = cv2.resize(img, IMG_SIZE)

        # Extract Haar-like features
        features = haar_like_feature(
            img, 0, 0,
            IMG_SIZE[0], IMG_SIZE[1],
            feature_coord=coords,
            feature_type=types
        )

        data.append(features)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print(f"✓ Total images loaded: {len(data)}")
print(f"✓ Unique labels found: {np.unique(labels).tolist()}")
print(f"✓ Feature dimension per image: {data.shape[1]}")
if skipped > 0:
    print(f"⚠ Skipped {skipped} unreadable images")

# ============================================================================
# STEP 2: TRAIN / TEST SPLIT
# ============================================================================

print("\n[STEP 2/4] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels   # ensures equal class distribution in split
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples:  {len(X_test)}")

# ============================================================================
# STEP 3: TRAIN SVM CLASSIFIER
# ============================================================================

print("\n[STEP 3/4] Training SVM classifier...")

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# NEW: Save the trained model so you don't have to retrain every time
joblib.dump(svm, MODEL_OUTPUT_PATH)
print(f"✓ Model saved to: {MODEL_OUTPUT_PATH}")

# ============================================================================
# STEP 4: EVALUATE MODEL
# ============================================================================

print("\n[STEP 4/4] Evaluating model...")

y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✓ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    cmap='Blues',
    fmt='d',
    xticklabels=np.unique(labels),
    yticklabels=np.unique(labels)
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix — Accuracy: {acc * 100:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✓ Confusion matrix saved as confusion_matrix.png")

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Total images:     {len(data)}")
print(f"  Train / Test:     {len(X_train)} / {len(X_test)}")
print(f"  Haar features:    {NUM_FEATURES}")
print(f"  SVM kernel:       linear")
print(f"  Final accuracy:   {acc * 100:.2f}%")
print("=" * 60)
