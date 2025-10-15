"""
BALANCED improved training - optimal augmentation
"""
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from models_binary import NeuralNetworkBinaryV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

print("=" * 70)
print("BALANCED Training - Optimal Rain Detection")
print("=" * 70)

DATA = Path("data")
CLASSES = ["Rain", "No-Rain"]
name2id = {c: i for i, c in enumerate(CLASSES)}
EXTS = {".jpg", ".jpeg", ".png"}

# Load images
print("\n📂 Loading images...")
X_list, y_list = [], []

for c in CLASSES:
    folder = DATA / c
    if not folder.exists():
        print(f"❌ Warning: Folder '{folder}' not found!")
        continue
    
    images = [p for p in folder.iterdir() if p.suffix.lower() in EXTS]
    print(f"  {c}: {len(images)} images")
    
    for img_path in images:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (50, 50))
            X_list.append(img)
            y_list.append(name2id[c])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

X = np.array(X_list)
y = np.array(y_list)

print(f"\n✅ Loaded {len(X)} images total")
print(f"   Rain: {np.sum(y == 0)}")
print(f"   No-Rain: {np.sum(y == 1)}")

# Split data
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 Split:")
print(f"   Training: {len(Xtr)} images")
print(f"   Testing: {len(Xte)} images")

# Balance training data with augmentation
print("\n🔄 Augmenting Rain images for balance...")
rain_mask = (ytr == 0)
rain_imgs = Xtr[rain_mask]
no_rain_imgs = Xtr[~rain_mask]

# Augment rain images
rain_flip = rain_imgs[:, :, ::-1, :]  # Horizontal flip
rain_dark = (rain_imgs * 0.9).astype(np.uint8)  # Darker
rain_bright = np.clip(rain_imgs * 1.1, 0, 255).astype(np.uint8)  # Brighter

# Combine
Xtr_balanced = np.concatenate([no_rain_imgs, rain_imgs, rain_flip, rain_dark, rain_bright])
ytr_balanced = np.concatenate([
    np.ones(len(no_rain_imgs)),  # No-Rain
    np.zeros(len(rain_imgs) * 4)  # Rain (original + 3 augmented)
]).astype(int)

print(f"   After augmentation:")
print(f"   Rain: {np.sum(ytr_balanced == 0)}")
print(f"   No-Rain: {np.sum(ytr_balanced == 1)}")

# Reshape for neural network
Xtr_f = Xtr_balanced.reshape(Xtr_balanced.shape[0], -1).T / 255.0
Xte_f = Xte.reshape(Xte.shape[0], -1).T / 255.0

# One-hot encode
N_tr = Xtr_f.shape[1]
N_te = Xte_f.shape[1]
Ytr = np.zeros((2, N_tr))
Yte = np.zeros((2, N_te))
Ytr[ytr_balanced, np.arange(N_tr)] = 1
Yte[yte, np.arange(N_te)] = 1

# Train model
print("\n🚀 Training model...")
net = NeuralNetworkBinaryV2()
net.train(Xtr_f, Ytr, epochs=150, lr=0.001, batch_size=64, verbose=True)

# After training, evaluate and create confusion matrix
print("\n📈 Evaluating model...")
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

# CREATE CONFUSION MATRIX VISUALIZATION
plt.figure(figsize=(10, 8))

# Calculate confusion matrix
cm = confusion_matrix(true, pred)

# Normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create heatmap
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES,
            cbar_kws={'label': 'Percentage'})

plt.title('Normalized Confusion Matrix\nBinary Weather Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add counts to the plot
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        plt.text(j + 0.5, i + 0.7, f'n={cm[i, j]}', 
                ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('confusion_matrix_binary.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved as 'confusion_matrix_binary.png'")

# Also create a raw counts confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (Counts)\nBinary Weather Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_counts_binary.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix (counts) saved as 'confusion_matrix_counts_binary.png'\n")

# Print detailed metrics
tp = np.sum((true == 0) & (pred == 0))
fn = np.sum((true == 0) & (pred == 1))
fp = np.sum((true == 1) & (pred == 0))
tn = np.sum((true == 1) & (pred == 1))

print("Confusion Matrix (Raw Counts):")
print(f"              Predicted")
print(f"            Rain  No-Rain")
print(f"Actual Rain  {tp:3d}    {fn:3d}")
print(f"    No-Rain  {fp:3d}    {tn:3d}\n")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Detailed Metrics:")
print(f"  Precision (Rain): {precision:.2%}")
print(f"  Recall (Rain):    {recall:.2%}")
print(f"  F1-Score (Rain):  {f1:.2%}")

# Save model with metadata
print(f"\n{'='*70}")
print("Saving model weights...")
weights = {
    'W1': net.W1, 'b1': net.b1,
    'W2': net.W2, 'b2': net.b2,
    'W3': net.W3, 'b3': net.b3,
    'W4': net.W4, 'b4': net.b4,
    'accuracy': acc,
    'rain_accuracy': rain_acc,
    'no_rain_accuracy': no_rain_acc,
    'confusion_matrix': cm.tolist()
}

with open('model_weights_binary.pkl', 'wb') as f:
    pickle.dump(weights, f)

print(f"✅ Model saved to 'model_weights_binary.pkl'")
print(f"{'='*70}\n")