# 🌤️ Weather Classifier - Deep Learning Image Classification

A production-ready weather classification system using a custom 4-layer neural network built from scratch with NumPy. Achieves **86.67% accuracy** on test data.

## 🎯 What This Project Does

Classifies weather images into 4 categories:
- ☁️ **Cloudy** - Overcast and cloudy skies
- 🌧️ **Rain** - Rainy weather conditions
- ☀️ **Shine** - Clear, sunny weather
- 🌅 **Sunrise** - Sunrise scenes

## 📁 Project Structure

```
weather_classifier/
├── models.py              # Neural network classes (inference)
├── train_and_save.py      # Training script (saves model weights)
├── model_weights.pkl      # Trained model (86.67% accuracy)
├── api.py                 # FastAPI REST API server
├── index.html             # Web interface for testing
├── test_api.py            # API testing script
├── requirements.txt       # Python dependencies
└── data/                  # Training dataset (1,125 images)
    ├── Cloudy/    (300 images)
    ├── Rain/      (215 images)
    ├── Shine/     (253 images)
    └── Sunrise/   (357 images)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
uvicorn api:app --reload --port 8000
```

### 3. Use the Application

**Option A: Web Interface** (Easiest)
```bash
# Open index.html in your browser
open index.html
```

**Option B: API Testing Script**
```bash
python test_api.py
```

**Option C: cURL Command**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/Shine/0.png"
```

**Option D: Interactive API Docs**
```bash
# Swagger UI with live testing
open http://localhost:8000/docs
```

## 📊 Model Performance

- **Architecture**: 4-layer neural network (7,500 → 1,024 → 512 → 100 → 4)
- **Test Accuracy**: 86.67%
- **Training**: 150 epochs, mini-batch gradient descent
- **Input**: 50×50 RGB images (normalized)
- **Output**: Probability distribution over 4 classes

## 🔧 Training Your Own Model

```bash
# Train from scratch and save weights
python train_and_save.py

# This will:
# 1. Load 1,125 images from data/
# 2. Train for 150 epochs
# 3. Save weights to model_weights.pkl
# 4. Show final accuracy
```

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API status |
| `/model-info` | GET | Get model details |
| `/predict` | POST | Classify single image |
| `/predict-batch` | POST | Classify multiple images |
| `/docs` | GET | Swagger UI documentation |

## 🧪 Testing

```bash
# Run automated tests
python test_api.py

# Test specific endpoint
curl http://localhost:8000/health
```

## 📚 Documentation

- **README.md** (this file) - Quick start and overview
- **CODE_BREAKDOWN.md** - Detailed code explanation
- **DOCUMENTATION_INDEX.md** - Complete project guide

## 🔑 Key Features

✅ Custom neural network built from scratch (no TensorFlow/PyTorch)  
✅ FastAPI REST API with automatic documentation  
✅ Beautiful web interface with drag-drop upload  
✅ Comprehensive error handling  
✅ Production-ready model persistence  
✅ Detailed logging and monitoring  
✅ Full test coverage  

## 🛠️ Tech Stack

- **Core**: Python, NumPy
- **API**: FastAPI, Uvicorn
- **Image Processing**: OpenCV, PIL
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: Custom neural network implementation

## 📈 Model Architecture

```
Input Layer:     7,500 neurons (50×50×3 flattened)
Hidden Layer 1:  1,024 neurons (ReLU activation)
Hidden Layer 2:    512 neurons (ReLU activation)  
Hidden Layer 3:    100 neurons (ReLU activation)
Output Layer:        4 neurons (Softmax activation)

Total Parameters: ~8.5 million
Weight Initialization: He initialization
Optimization: Mini-batch gradient descent
Learning Rate: 0.001
Batch Size: 64
```

## 🎓 How to Use This Project

1. **For Learning**: Study the code to understand neural networks
2. **For Development**: Use the API in your applications
3. **For Production**: Deploy the API to cloud servers
4. **For Research**: Experiment with different architectures

## 🤝 Contributing

Feel free to:
- Add more weather classes
- Improve the model architecture
- Enhance the web interface
- Deploy to cloud platforms

## 📄 License

This project is open source and available for educational purposes.# Weather-Image-Classification
