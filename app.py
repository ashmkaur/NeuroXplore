import os
import tensorflow as tf
from flask import Flask, request, redirect, url_for, render_template_string
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from flask import render_template

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Load the TensorFlow model
model = tf.keras.models.load_model('model/model.h5')  # Adjust path as needed

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the main page (file upload form)
@app.route('/')
def index():
    return render_template('front.html') 

# POST endpoint to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'mriScan' not in request.files:
        return 'No file part'
    
    file = request.files['mriScan']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        # Save the file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the MRI scan
        image = Image.open(file_path)
        image = image.resize((224, 224))  # Resize to model input size
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Run the prediction
        prediction = model.predict(image)
        result = 'Healthy' if prediction[0] > 0.5 else 'Not Healthy'
        
        # Delete the uploaded file
        os.remove(file_path)
        
        # Redirect to the result page
        return redirect(url_for('result', result=result, name=request.form['fullName'],
                                age=request.form['age'], gender=request.form['gender']))
    
    return 'File not allowed'

# Endpoint to display the result
# Endpoint to display the result
@app.route('/result')
def result():
    result = request.args.get('result')
    name = request.args.get('name')
    age = request.args.get('age')
    gender = request.args.get('gender')

    print(f"Result: {result}, Name: {name}, Age: {age}, Gender: {gender}")  # Debugging print

    return render_template('result.html', result=result, name=name, age=age, gender=gender)


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=3000)
