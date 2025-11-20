"""
Skin Disease Classification Web Application
Dataset: 5 classes (acne, hyperpigmentation, nail_psoriasis, sjs_ten, vitiligo)
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease information
DISEASE_INFO = {
    'acne': {
        'name': 'Acne',
        'full_name': 'Acne Vulgaris',
        'description': 'Common skin condition characterized by pimples, blackheads, and whiteheads. Occurs when hair follicles become clogged with oil and dead skin cells.',
        'severity': 'Low-Medium',
        'color': '#FF9800',
        'recommendation': 'Use non-comedogenic skincare products, maintain good hygiene, avoid touching face. Consult dermatologist for severe cases or prescription treatments.'
    },
    'hyperpigmentation': {
        'name': 'Hyperpigmentation',
        'full_name': 'Skin Hyperpigmentation',
        'description': 'Darkening of skin patches caused by excess melanin production. Can result from sun exposure, inflammation, hormones, or skin injuries.',
        'severity': 'Low',
        'color': '#8D6E63',
        'recommendation': 'Use sunscreen daily (SPF 30+), topical treatments like vitamin C, retinoids. Laser therapy available for persistent cases. Consult dermatologist.'
    },
    'nail_psoriasis': {
        'name': 'Nail Psoriasis',
        'full_name': 'Psoriatic Nail Disease',
        'description': 'Chronic condition affecting nails, causing pitting, discoloration, thickening, and separation from nail bed. Associated with psoriasis.',
        'severity': 'Medium',
        'color': '#E91E63',
        'recommendation': 'Keep nails trimmed, moisturize regularly. Medical treatments include topical corticosteroids, vitamin D analogs. Consult dermatologist for treatment plan.'
    },
    'sjs_ten': {
        'name': 'SJS-TEN',
        'full_name': 'Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis',
        'description': 'SEVERE life-threatening skin reaction, usually triggered by medication. Causes widespread blistering and skin peeling. Medical emergency.',
        'severity': 'Critical',
        'color': '#D32F2F',
        'recommendation': '🚨 MEDICAL EMERGENCY: Seek immediate hospital care. Discontinue suspected medications. Requires intensive care unit treatment. DO NOT DELAY.'
    },
    'vitiligo': {
        'name': 'Vitiligo',
        'full_name': 'Vitiligo',
        'description': 'Autoimmune condition causing loss of skin pigmentation in patches. Melanocytes (pigment cells) are destroyed, resulting in white patches.',
        'severity': 'Low-Medium',
        'color': '#9C27B0',
        'recommendation': 'Protect affected areas from sun exposure (SPF 50+). Treatment options include topical corticosteroids, light therapy, skin grafting. Consult dermatologist.'
    }
}

# Model configuration
IMG_SIZE = 224
CLASS_NAMES = ['acne', 'hyperpigmentation', 'nail_psoriasis', 'sjs_ten', 'vitiligo']

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    model_path = 'models/best_model.h5'
    
    if os.path.exists(model_path):
        print(f"📦 Loading model from {model_path}...")
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return True
    else:
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first: python train.py")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_prediction(image_path):
    """Get prediction from model"""
    if model is None:
        return None
    
    # Preprocess
    img_array = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Get top prediction
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx])
    predicted_class = CLASS_NAMES[pred_idx]
    
    # Get all predictions
    all_predictions = []
    for idx, prob in enumerate(predictions[0]):
        all_predictions.append({
            'class': CLASS_NAMES[idx],
            'name': DISEASE_INFO[CLASS_NAMES[idx]]['name'],
            'probability': float(prob),
            'percentage': f"{float(prob) * 100:.2f}%"
        })
    
    # Sort by probability
    all_predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'percentage': f"{confidence * 100:.2f}%",
        'disease_info': DISEASE_INFO[predicted_class],
        'all_predictions': all_predictions
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG, BMP allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction
        result = get_prediction(filepath)
        
        if result is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Add filepath to result
        result['image_url'] = f"/static/uploads/{filename}"
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔬 SKIN DISEASE CLASSIFICATION WEB APP")
    print("="*80)
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print("="*80)
    
    # Load model
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not found!")
        print("Please train the model first:")
        print("   python train.py")
        print("\nThe app will still start but predictions won't work.\n")
    
    print("\n🚀 Starting Flask server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)