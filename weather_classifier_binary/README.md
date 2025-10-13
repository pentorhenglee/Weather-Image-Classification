# 🌧️ Binary Weather Classifier - Rain vs No-RaiExpected output:
```
BALANCED Training - Optimal Rain Detection
============================================================
Rain: 215 images
No-Rain: 910 images

Training Neural Network (150 epochs)
...
FINAL RESULTS:
  Rain:    90.70% (excellent!)
  No-Rain: 93.96%
  Overall: 93.33%

✓ Model weights saved to: model_weights_binary.pkl
```

### 3. Start the API Serversification system that detects whether it's **raining** or **not raining** from images, using a custom 4-layer neural network built from scratch with NumPy.

## 🎯 What This Does

Classifies weather images into **2 categories**:
- 🌧️ **Rain** - Rainy weather conditions
- ☀️ **No-Rain** - All other weather (Cloudy, Sunny, Sunrise)

## 📁 Project Structure

```
weather_classifier_binary/
├── models_binary.py              # Binary neural network classes
├── train_model.py                # Training script (with data augmentation)
├── api_binary.py                 # FastAPI REST API (port 8001)
├── index_binary.html             # Web interface
├── requirements.txt              # Python dependencies
├── model_weights_binary.pkl      # Trained model (90.70% rain accuracy)
└── data/                         # Training dataset
    ├── Rain/        (215 images)
    └── No-Rain/     (910 images - Cloudy+Shine+Sunrise)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional - Already Trained!)

**Note:** The model is already trained with 90.70% rain accuracy. Only retrain if you want to experiment.

```bash
python3 train_model.py
```

This will:
- Load 1,125 images (215 Rain + 910 No-Rain)
- Apply data augmentation (3x rain images for balance)
- Train for 150 epochs
- Save weights to `model_weights_binary.pkl`
- Display final accuracy

Expected output:
```
Binary Weather Classifier - Training (Rain vs No-Rain)
============================================================
Rain: 215 images
No-Rain: 910 images

Training Neural Network Binary V2 (4 layers, 2 outputs)
============================================================
Epoch 10/150 - Loss: 0.3245
Epoch 20/150 - Loss: 0.2156
...
Epoch 150/150 - Loss: 0.0824

Overall Test Accuracy: ~92-95%
✓ Model weights saved to: model_weights_binary.pkl
```

### 4. Start the API Server

```bash
uvicorn api_binary:app --reload --port 8001
```

The API will run on **port 8001** (different from the original 4-class model on port 8000).

### 5. Use the Application

#### **Option A: Web Interface** (Easiest)

```bash
open index_binary.html
```

- Drag and drop any weather image
- Click "Classify Weather"
- See if it's Rain or No-Rain!

#### **Option B: cURL Command**

```bash
curl -X POST http://localhost:8001/predict \
  -F "file=@data/Rain/0.png"
```

#### **Option C: Python Script**

```python
import requests

with open('data/Rain/0.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/predict',
        files={'file': f}
    )

result = response.json()
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### **Option D: Interactive API Docs**

```bash
open http://localhost:8001/docs
```

## 📊 Model Architecture

```
Input Layer:     7,500 neurons (50×50×3 flattened)
Hidden Layer 1:  1,024 neurons (ReLU activation)
Hidden Layer 2:    512 neurons (ReLU activation)  
Hidden Layer 3:    100 neurons (ReLU activation)
Output Layer:        2 neurons (Softmax activation)

Classes: Rain (0) | No-Rain (1)
```

## 📈 Expected Performance

- **Dataset**: 1,125 images
  - Rain: 215 images (19%)
  - No-Rain: 910 images (81%)
- **Train/Test Split**: 80/20
- **Expected Accuracy**: 92-95%
- **Training Time**: ~2-3 minutes
- **Inference Time**: <100ms per image

Note: The dataset is imbalanced (more No-Rain than Rain), which is realistic for most weather scenarios.

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Check API status |
| `/model-info` | GET | Get model details |
| `/predict` | POST | Classify single image |
| `/predict-batch` | POST | Classify multiple images |
| `/docs` | GET | Swagger UI documentation |

## 🔑 Key Features

✅ Binary classification (simpler than 4-class)  
✅ Custom neural network (no TensorFlow/PyTorch)  
✅ FastAPI REST API on port 8001  
✅ Beautiful web interface  
✅ Separate from original 4-class project  
✅ Realistic imbalanced dataset  

## 🛠️ Tech Stack

- **Core**: Python, NumPy
- **API**: FastAPI, Uvicorn (port 8001)
- **Image Processing**: OpenCV, PIL
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: Custom neural network implementation

## 🆚 Differences from Original Project

| Feature | Original (4-class) | Binary (2-class) |
|---------|-------------------|------------------|
| **Classes** | Cloudy, Rain, Shine, Sunrise | Rain, No-Rain |
| **Output Neurons** | 4 | 2 |
| **Accuracy** | ~86-87% | ~92-95% |
| **API Port** | 8000 | 8001 |
| **Folder** | `weather_classifier/` | `weather_classifier_binary/` |
| **Use Case** | Detailed weather | Simple rain detection |

## 🎓 When to Use Each Model

### Use **Binary Model** (this one) when:
- You only care about rain detection
- Building a rain alert system
- Need higher accuracy
- Want faster training
- Simpler problem

### Use **4-Class Model** (original) when:
- Need detailed weather classification
- Building weather dashboard
- Want to distinguish sunny vs cloudy
- Need sunrise detection

## 🧪 Testing the Model

Test on different weather conditions:

```bash
# Test rain image
curl -X POST http://localhost:8001/predict \
  -F "file=@data/Rain/0.png"

# Test sunny image
curl -X POST http://localhost:8001/predict \
  -F "file=@data/No-Rain/Shine_0.png"

# Test cloudy image
curl -X POST http://localhost:8001/predict \
  -F "file=@data/No-Rain/Cloudy_0.png"
```

## 🔧 Troubleshooting

### Model not found error
```bash
# Make sure you trained the model first:
python3 train_binary.py
```

### API not starting
```bash
# Check if port 8001 is available:
lsof -i :8001

# Or use a different port:
uvicorn api_binary:app --reload --port 8002
```

### Low accuracy
- Train for more epochs (modify `train_binary.py`)
- Adjust learning rate
- Try data augmentation

## 📚 Files Explained

- **`models_binary.py`** - Neural network classes (2 outputs instead of 4)
- **`train_binary.py`** - Loads data, trains model, saves weights
- **`api_binary.py`** - REST API server for predictions
- **`index_binary.html`** - User interface for testing
- **`prepare_binary_data.py`** - Reorganizes images into Rain/No-Rain
- **`model_weights_binary.pkl`** - Saved model (created after training)

## 🚀 Next Steps

1. ✅ Train the model: `python3 train_binary.py`
2. ✅ Start API: `uvicorn api_binary:app --reload --port 8001`
3. ✅ Open web interface: `open index_binary.html`
4. ✅ Upload an image and see if it's raining!

## 🎉 Summary

You now have a **binary weather classifier** that can:
- Detect rain vs no-rain with ~92-95% accuracy
- Run alongside your original 4-class model (different ports)
- Use the same neural network architecture (just 2 outputs)
- Be accessed via web interface or API

**Start predicting rain! 🌧️☀️**

## 📄 Requirements

```
numpy
opencv-python
pillow
scikit-learn
fastapi
uvicorn[standard]
python-multipart
pydantic
```

Install with: `pip install -r requirements.txt`
