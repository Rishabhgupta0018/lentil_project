from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('lentil_disease_resnet_model.h5')
class_names = ['Ascochyta blight', 'Lentil Rust', 'Normal', 'Powdery Mildew']

IMAGE_SIZE = 224  # Adjust based on your model's input size

# Function to preprocess the image and make predictions
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class_name, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_class, confidence = predict_image(file_path)

            return render_template(
                'index.html', 
                image_path=file_path, 
                predicted_label=predicted_class, 
                confidence=confidence
            )

    return render_template('index.html')

# Function to check if the file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)