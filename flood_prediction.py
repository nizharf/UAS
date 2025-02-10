from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model CNN yang sudah dilatih
MODEL_PATH = "models/model_flood.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Konfigurasi upload folder
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memproses gambar
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Sesuai ukuran input model
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimensi
    return img

# Route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        # Simpan file yang diupload
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preproses dan prediksi
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        result = "Banjir Terprediksi" if prediction[0][0] > 0.5 else "Tidak Ada Banjir"
        
        return render_template('index.html', result=result, image_path=filepath)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
