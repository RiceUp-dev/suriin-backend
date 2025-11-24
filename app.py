from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your model
print("Loading rice pest model...")
model = tf.keras.models.load_model('rice_pest_model.h5')
print("✅ Model loaded successfully!")

@app.route('/')
def home():
    return jsonify({
        "message": "Rice Pest Detection API",
        "status": "active",
        "model_loaded": True
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Read and preprocess image
        image_data = file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array)
        probabilities = prediction[0].tolist()
        
        # For binary classification [healthy_prob, pest_prob]
        healthy_prob = probabilities[0] if len(probabilities) == 2 else (1 - probabilities[0])
        pest_prob = probabilities[1] if len(probabilities) == 2 else probabilities[0]
        
        has_pest = pest_prob > healthy_prob
        confidence = max(healthy_prob, pest_prob) * 100
        
        return jsonify({
            'pest': 'Insects' if has_pest else 'Healthy',
            'confidence': round(confidence, 2),
            'hasPest': has_pest,
            'probabilities': {
                'healthy': round(healthy_prob * 100, 2),
                'pest': round(pest_prob * 100, 2)
            },
            'modelUsed': 'Render Backend (.h5)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)