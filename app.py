from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
# from flask import Flask, render_template, url_for

# app = Flask(__name__, static_folder='static')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained models
models = {
    'brain': load_model('models/best_brain_model.keras'),
    'kidney': load_model('models/kidney_model.h5'),
    'lung': load_model('models/lung_model.h5'),
    'tb': load_model('models/best_tuberculosis_model.keras')
}

# Define the class labels for each model
class_labels = {
    'brain': ['No_Tumor', 'Tumor'],
    'kidney': ['Cyst', 'Kidney_Stone', 'Normal', 'Tumor'],
    'lung': ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma'],
    'tb': ['No_TB', 'TB']
}

# Serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# About Us route
@app.route('/about')
def about():
    return render_template('about.html')

# Book Appointment route
@app.route('/appointment')
def appointment():
    return render_template('appointment.html')


# Route for specific disease pages
@app.route('/<disease>')
def disease_page(disease):
    if disease in models:
        return render_template('upload.html', disease=disease)
    else:
        return "Invalid disease type", 404

# Prediction route to handle all disease types and redirect to specific result pages
@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['model_type']
    
    if model_type not in models:
        return "Invalid disease type", 400

    # Check if a file is uploaded
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save the file to a temporary location
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Preprocess the image for prediction
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction using the appropriate model
    model = models[model_type]
    prediction = model.predict(img_array)

    # For binary classification (like brain or TB), get the probability
    if len(prediction[0]) == 1:
        predicted_class = class_labels[model_type][1] if prediction[0][0] > 0.5 else class_labels[model_type][0]
    else:
        predicted_class = class_labels[model_type][np.argmax(prediction)]

    # Remove the file after prediction
    os.remove(filepath)

    # Redirect to the specific result page for the disease
    return redirect(url_for(f'{model_type}_result', prediction=predicted_class))

# Routes for separate result pages (one for each disease)
@app.route('/kidney_result')
def kidney_result():
    prediction = request.args.get('prediction')
    return render_template('kidney.html', prediction=prediction)

@app.route('/lung_result')
def lung_result():
    prediction = request.args.get('prediction')
    return render_template('lung.html', prediction=prediction)

@app.route('/brain_result')
def brain_result():
    prediction = request.args.get('prediction')
    return render_template('brain.html', prediction=prediction)

@app.route('/tb_result')
def tb_result():
    prediction = request.args.get('prediction')
    return render_template('tb.html', prediction=prediction)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
