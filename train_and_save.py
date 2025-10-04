"""
Script to train the model and save weights for the API
"""

import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from models import NeuralNetworkV2
import pickle

DATA = Path("data")
CLASSES = ["Cloudy", "Rain", "Shine", "Sunrise"]
name2id = {c: i for i, c in enumerate(CLASSES)}
EXTS = {".jpg", ".jpeg", ".png"}

print("=" * 60)
print("Weather Classifier - Model Training & Saving")
print("=" * 60)

# --- Load images & labels ---
X_list, y_list = [], []
for c in CLASSES:
    files = [p for p in (DATA / c).rglob("*") if p.suffix.lower() in EXTS]
    print(f"{c}: {len(files)} images")
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (50, 50))
        X_list.append(img)
        y_list.append(name2id[c])

if not X_list:
    raise SystemExit("No images found. Check extensions and data path.")

X = (np.stack(X_list).astype(np.float32) / 255.0)
y = np.array(y_list, dtype=np.int64)

print(f"\nTotal dataset: {X.shape[0]} images")

# Stratified split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train: {Xtr.shape[0]} images | Test: {Xte.shape[0]} images")

# Flatten to (D, N)
Xtr_f = Xtr.reshape(Xtr.shape[0], -1).T
Xte_f = Xte.reshape(Xte.shape[0], -1).T

# One-hot to (C, N)
Ytr = np.zeros((4, Xtr.shape[0]), np.float32)
Ytr[ytr, np.arange(len(ytr))] = 1
Yte = np.zeros((4, Xte.shape[0]), np.float32)
Yte[yte, np.arange(len(yte))] = 1

# --- Train ---
print("\n" + "=" * 60)
print("Training Neural Network V2 (4 layers)")
print("=" * 60)

net = NeuralNetworkV2()
net.train(Xtr_f, Ytr, epochs=150, lr=1e-3, batch_size=64, verbose=True)

# --- Evaluate ---
Pte = net.predict(Xte_f)
acc = net.accuracy(Pte, Yte)
print(f"\n{'=' * 60}")
print(f"Test Accuracy: {acc:.2f}%")
print("=" * 60)

# --- Save model weights ---
print("\nSaving model weights...")
weights = {
    'W1': net.W1,
    'b1': net.b1,
    'W2': net.W2,
    'b2': net.b2,
    'W3': net.W3,
    'b3': net.b3,
    'W4': net.W4,
    'b4': net.b4,
    'accuracy': acc,
    'classes': CLASSES,
    'input_shape': (50, 50, 3)
}

model_path = Path("model_weights.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(weights, f)

print(f"✓ Model weights saved to: {model_path}")
print(f"✓ Test accuracy: {acc:.2f}%")
print(f"\nYou can now run the API with: python api.py")
print("Or: uvicorn api:app --reload")
