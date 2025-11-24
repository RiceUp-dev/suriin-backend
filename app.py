from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import random

app = Flask(__name__)
CORS(app)

# Simple image analysis without TensorFlow
def analyze_image_simple(image_data):
    """Simple image analysis that doesn't require TensorFlow"""
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # Basic image analysis
        width, height = img.size
        img_rgb = img.convert('RGB')
        
        # Get average color (very basic "analysis")
        pixels = img_rgb.getdata()
        total_pixels = width * height
        
        # Simple color analysis (this is where you'd add real logic)
        green_pixels = sum(1 for pixel in pixels if pixel[1] > pixel[0] and pixel[1] > pixel[2])
        green_ratio = green_pixels / total_pixels
        
        # Simulate pest detection based on simple heuristics
        if green_ratio > 0.6:
            # Mostly green - likely healthy
            pest_prob = random.uniform(0.1, 0.3)
        else:
            # Less green - possible pest damage
            pest_prob = random.uniform(0.6, 0.9)
        
        healthy_prob = 1 - pest_prob
        has_pest = pest_prob > 0.5
        
        return {
            'pest': 'Insects' if has_pest else 'Healthy',
            'confidence': round(max(healthy_prob, pest_prob) * 100, 2),
            'hasPest': has_pest,
            'analysis': {
                'image_size': f"{width}x{height}",
                'green_ratio': round(green_ratio * 100, 2),
                'method': 'Basic Color Analysis'
            }
        }
        
    except Exception as e:
        return {
            'pest': 'Healthy',
            'confidence': 50.0,
            'hasPest': False,
            'error': str(e)
        }

@app.route('/')
def home():
    return jsonify({
        "message": "RiceUp - Suriin ang Palay Backend API",
        "status": "active",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict_pest": "/predict/pest",
            "predict_disease": "/predict/disease"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "backend": "active"})

@app.route('/predict/pest', methods=['POST'])
def predict_pest():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Analyze image using simple method
        image_data = file.read()
        result = analyze_image_simple(image_data)
        
        return jsonify({
            'pest': result['pest'],
            'confidence': result['confidence'],
            'hasPest': result['hasPest'],
            'modelUsed': 'Basic Image Analysis',
            'analysis': result.get('analysis', {})
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Simulate disease detection
        diseases = [
            'Bacterial Leaf Blight', 'Brown Spot', 'Leaf Blast', 
            'Leaf Scald', 'Narrow Brown Leaf Spot', 'Rice Hispa',
            'Sheath Blight', 'Tungro', 'Healthy Rice Leaf'
        ]
        
        weights = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.36]  # Higher weight for healthy
        
        simulated_disease = random.choices(diseases, weights=weights)[0]
        confidence = round(75 + random.random() * 20, 2)
        
        return jsonify({
            'disease': simulated_disease,
            'confidence': confidence,
            'isHealthy': simulated_disease == 'Healthy Rice Leaf',
            'modelUsed': 'Statistical Simulation'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
