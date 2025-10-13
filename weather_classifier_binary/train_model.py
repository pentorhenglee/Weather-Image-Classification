"""
BALANCED improved training - optimal augmentation
"""

import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from models_binary import NeuralNetworkBinaryV2
import pickle

print("=" * 70)
print("BALANCED Training - Optimal Rain Detection")
print("=" * 70)

DATA = Path("data")
CLASSES = ["Rain", "No-Rain"]
name2id = {c: i for i, c in enumerate(CLASSES)}
EXTS = {".jpg", ".jpeg", ".png"}

# Load
X_list, y_list = [], []
for c in CLASSES:
    files = [p for p in (DATA / c).rglob("*") if p.suffix.lower() in EXTS]
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (50, 50))
        X_list.append(img)
        y_list.append(name2id[c])

X = (np.stack(X_list).astype(np.float32) / 255.0)
y = np.array(y_list, dtype=np.int64)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Original: Rain={np.sum(ytr == 0)}, No-Rain={np.sum(ytr == 1)}")

# MODERATE augmentation - only 2x rain images (not 5x)
rain_idx = ytr == 0
rain_imgs = Xtr[rain_idx]

# Just horizontal flip + darker version
rain_aug = np.concatenate([
    rain_imgs[:, :, ::-1, :],     # H-flip only
    rain_imgs * 0.9,              # Slightly darker
], axis=0)

Xtr = np.concatenate([Xtr, rain_aug])
ytr = np.concatenate([ytr, np.zeros(rain_aug.shape[0], dtype=np.int64)])

print(f"Augmented: Rain={np.sum(ytr == 0)}, No-Rain={np.sum(ytr == 1)}")
print(f"Balance: 1:{np.sum(ytr == 1) / np.sum(ytr == 0):.2f}\n")

# Prepare
Xtr_f = Xtr.reshape(Xtr.shape[0], -1).T
Xte_f = Xte.reshape(Xte.shape[0], -1).T
Ytr = np.zeros((2, Xtr.shape[0]), np.float32)
Ytr[ytr, np.arange(len(ytr))] = 1
Yte = np.zeros((2, Xte.shape[0]), np.float32)
Yte[yte, np.arange(len(yte))] = 1

# Train with moderate parameters
print("Training (150 epochs, lr=0.001)...")
net = NeuralNetworkBinaryV2()
net.train(Xtr_f, Ytr, epochs=150, lr=1e-3, batch_size=64, verbose=True)

# Evaluate
Pte = net.predict(Xte_f)
acc = net.accuracy(Pte, Yte)
pred = np.argmax(Pte, axis=0)
true = np.argmax(Yte, axis=0)

rain_acc = np.mean(pred[true == 0] == 0) * 100
no_rain_acc = np.mean(pred[true == 1] == 1) * 100

print(f"\n{'='*70}")
print(f"FINAL RESULTS:")
print(f"  Rain:    {rain_acc:.2f}% ({np.sum(true == 0)} samples)")
print(f"  No-Rain: {no_rain_acc:.2f}% ({np.sum(true == 1)} samples)")
print(f"  Overall: {acc:.2f}%")
print(f"{'='*70}\n")

# Confusion matrix
tp = np.sum((true == 0) & (pred == 0))
fn = np.sum((true == 0) & (pred == 1))
fp = np.sum((true == 1) & (pred == 0))
tn = np.sum((true == 1) & (pred == 1))

print("Confusion Matrix:")
print(f"              Predicted")
print(f"            Rain  No-Rain")
print(f"Actual Rain  {tp:3d}    {fn:3d}")
print(f"    No-Rain  {fp:3d}    {tn:3d}\n")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Rain Detection Metrics:")
print(f"  Precision: {precision:.2%} (accuracy when predicting rain)")
print(f"  Recall:    {recall:.2%} (% of rain events detected)")
print(f"  F1-Score:  {f1:.2%}")

# Save
weights = {
    'W1': net.W1, 'b1': net.b1, 'W2': net.W2, 'b2': net.b2,
    'W3': net.W3, 'b3': net.b3, 'W4': net.W4, 'b4': net.b4,
    'accuracy': acc, 'rain_accuracy': rain_acc,
    'no_rain_accuracy': no_rain_acc, 'classes': CLASSES,
    'input_shape': (50, 50, 3)
}

with open("model_weights_binary.pkl", 'wb') as f:
    pickle.dump(weights, f)

print(f"\n✓ Model saved to: model_weights_binary.pkl")
print(f"\nRestart API:")
print(f"  kill -9 $(lsof -ti:8001)")
print(f"  python3 -m uvicorn api_binary:app --reload --port 8001 &")
