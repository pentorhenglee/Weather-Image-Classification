"""
Simple test script for the Weather Classifier API
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("Testing /model-info endpoint...")
    response = requests.get(f"{API_URL}/model-info")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Model: {data['model_name']}")
    print(f"Classes: {data['classes']}")
    print(f"Input shape: {data['input_shape']}")
    print(f"Model loaded: {data['model_loaded']}\n")
    return response.status_code == 200

def test_prediction(image_path):
    """Test prediction endpoint with an image"""
    print(f"Testing /predict endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}\n")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Predicted class: {data['predicted_class']}")
        print(f"Confidence: {data['confidence']:.2%}")
        print("Probabilities:")
        for cls, prob in data['probabilities'].items():
            print(f"  {cls}: {prob:.2%}")
        print()
        return True
    else:
        print(f"Error: {response.text}\n")
        return False

def main():
    print("=" * 60)
    print("Weather Classifier API Test")
    print("=" * 60)
    print()
    
    # Test health
    if not test_health():
        print("❌ Health check failed. Is the API running?")
        return
    
    # Test model info
    if not test_model_info():
        print("❌ Model info failed")
        return
    
    # Test predictions with sample images
    test_images = [
        "data/Cloudy/0.png",
        "data/Rain/0.png",
        "data/Shine/0.png",
        "data/Sunrise/0.png"
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            test_prediction(img_path)
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print()
    print("Try the web interface:")
    print("  1. Open index.html in your browser")
    print("  2. Upload an image")
    print("  3. Click 'Classify Weather'")
    print()
    print("Or test with curl:")
    print(f'  curl -X POST {API_URL}/predict -F "file=@path/to/image.jpg"')

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start it with:")
        print("   python api.py")
        print("   or")
        print("   uvicorn api:app --reload")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
