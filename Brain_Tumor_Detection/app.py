import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('brain_tumor_model.h5')

# Define allowed file extensions (for security)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Save the image temporarily for prediction
            img_path = os.path.join('static', filename)
            file.save(img_path)
            
            # Load the image for prediction
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make the prediction
            prediction = model.predict(img_array)
            result = 'Tumor' if prediction[0] > 0.5 else 'Healthy'
            image_url = img_path  # URL to the image for rendering
            
    return render_template('index.html', result=result, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
