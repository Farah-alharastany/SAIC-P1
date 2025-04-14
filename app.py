from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import os

# استخدم CPU فقط
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# المسار إلى النموذج بعد رفعه عبر Git LFS
model_path = "VGG16_model.h5"  # تأكد أن النموذج في نفس المجلد الذي يحتوي على ملف app.py

# تحميل النموذج المدرب
model = load_model(model_path)

# إعدادات الصورة
IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = image.resize(IMG_SIZE).convert('RGB')
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')

# نقطة التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    img = Image.open(file.stream)
    img_tensor = preprocess_image(img)
    prediction = model.predict(img_tensor)
    predicted_class_index = np.argmax(prediction[0])
    
    # أسماء الفئات
    class_names = [
        'Central Serous Chorioretinopathy', 'Diabetic Retinopathy', 'Disc Edema',
        'Glaucoma', 'Healthy', 'Macular Scar', 'Myopia', 'Pterygium',
        'Retinal Detachment', 'Retinitis Pigmentosa'
    ]
    
    predicted_class = class_names[predicted_class_index]
    
    return jsonify({'prediction': predicted_class})

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)
