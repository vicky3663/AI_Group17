from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model (ensure the model is trained on Fashion MNIST)
MODEL_PATH = "fashion_mnist_model.h5"  # Replace with the actual path to your model
model = load_model(MODEL_PATH)

# Fashion MNIST class labels
CLASS_LABELS = [
    'T-shirt/top', 
    'Trouser', 
    'Pullover', 
    'Dress', 
    'Coat', 
    'Sandal', 
    'Shirt', 
    'Sneaker', 
    'Bag', 
    'Ankle boot'
]

@app.route('/')
def index():
    return render_template('index.html')  # Home page with upload form

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Classify the uploaded image
        result, confidence = classify_image(file_path)

        # Return the result to the user
        return render_template('result.html', result=result, confidence=confidence)

def classify_image(img_path):
    """Processes and classifies an image for the Fashion MNIST dataset."""
    # For Fashion MNIST, images are 28x28 grayscale images.
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100  # Confidence percentage

    return CLASS_LABELS[predicted_index], confidence

if __name__ == '__main__':
    app.run(debug=True)
