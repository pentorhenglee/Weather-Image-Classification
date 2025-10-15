"""
FastAPI application for Binary Weather Classification (Rain vs No-Rain)
Provides REST API endpoints to classify weather images
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import pickle
from pathlib import Path
import logging

# Import the model
from models_binary import NeuralNetworkBinaryV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Binary Weather Classifier API",
    description="Deep Learning API for binary weather classification: Rain vs No-Rain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CLASSES = ["Rain", "No-Rain"]
MODEL_PATH = Path("model_weights_binary.pkl")
IMAGE_SIZE = (50, 50)

# Global model variable
model = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str
    confidence: float
    probabilities: dict
    message: str


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    classes: List[str]
    input_shape: tuple
    model_loaded: bool


def load_model():
    """Load the trained model weights"""
    global model
    try:
        if MODEL_PATH.exists():
            logger.info(f"Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                weights = pickle.load(f)
            
            model = NeuralNetworkBinaryV2()
            model.W1 = weights['W1']
            model.b1 = weights['b1']
            model.W2 = weights['W2']
            model.b2 = weights['b2']
            model.W3 = weights['W3']
            model.b3 = weights['b3']
            model.W4 = weights['W4']
            model.b4 = weights['b4']
            logger.info("Model loaded successfully")
            logger.info(f"Model accuracy: {weights.get('accuracy', 'N/A')}%")
            return True
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Please train the model first.")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, IMAGE_SIZE)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Flatten to (7500, 1) for the model
        img_flattened = img_normalized.reshape(-1, 1)
        
        return img_flattened
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Binary Weather Classifier API...")
    load_model()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Binary Weather Classifier API (Rain vs No-Rain)",
        "version": "1.0.0",
        "classes": CLASSES,
        "endpoints": {
            "predict": "/predict - POST - Upload image for classification",
            "model_info": "/model-info - GET - Get model information",
            "health": "/health - GET - Check API health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "binary_classification",
        "classes": CLASSES
    }


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    return ModelInfo(
        model_name="NeuralNetworkBinaryV2",
        version="1.0.0",
        classes=CLASSES,
        input_shape=(50, 50, 3),
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_weather(file: UploadFile = File(...)):
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure model weights are available. Run: python3 train_binary.py"
        )
    
    # Validate file type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (jpg, png, jpeg)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.feed_forward(processed_image)
        
        # Get probabilities (softmax output)
        probabilities = prediction.flatten()
        
        # Get predicted class
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create probability dictionary
        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(CLASSES, probabilities)
        }
        
        # Add weather emoji
        emoji = "🌧️" if predicted_class == "Rain" else "☀️"
        
        logger.info(f"Prediction: {emoji} {predicted_class} with confidence {confidence:.2%}")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            message=f"Prediction successful: {emoji} {predicted_class}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure model weights are available."
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            prediction = model.feed_forward(processed_image)
            probabilities = prediction.flatten()
            predicted_idx = int(np.argmax(probabilities))
            
            results.append({
                "filename": file.filename,
                "predicted_class": CLASSES[predicted_idx],
                "confidence": float(probabilities[predicted_idx]),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return {"predictions": results, "total": len(files)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
