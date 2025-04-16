from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import os

# استخدم CPU فقط
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# المسار إلى النموذج بعد رفعه عبر Git LFS
model_path = "EYE_disease_model.h5"  # تأكد أن النموذج في نفس المجلد الذي يحتوي على ملف app.py

# تحميل النموذج المدرب
print("[INFO] Loading model...")
model = load_model(model_path)
print("[INFO] Model loaded successfully.")

# إعدادات الصورة
IMG_SIZE = (224, 224)

def preprocess_image(image):
    try:
        print("[INFO] Preprocessing image...")
        image = image.resize(IMG_SIZE).convert('RGB')
        print("[DEBUG] Image resized and converted to RGB.")
        image_array = np.array(image) / 255.0
        print(f"[DEBUG] Image array shape after normalization: {image_array.shape}")
        image_tensor = np.expand_dims(image_array, axis=0)
        print(f"[DEBUG] Final tensor shape (with batch dimension): {image_tensor.shape}")
        return image_tensor
    except Exception as e:
        print(f"[ERROR] Error during preprocessing: {str(e)}")
        raise

# الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')

# نقطة التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    print("[INFO] Received request for prediction.")
    
    if 'image' not in request.files:
        print("[ERROR] No image found in request.")
        return jsonify({'error': 'No image uploaded'})

    try:
        file = request.files['image']
        print(f"[INFO] Image file received: {file.filename}")

        img = Image.open(file.stream)
        print("[INFO] Image opened successfully.")

        img_tensor = preprocess_image(img)

        print("[INFO] Making prediction...")
        prediction = model.predict(img_tensor)
        print(f"[DEBUG] Raw prediction output: {prediction}")

        predicted_class_index = np.argmax(prediction[0])
        print(f"[INFO] Predicted class index: {predicted_class_index}")

        class_names = [
            'Central Serous Chorioretinopathy', 'Diabetic Retinopathy', 'Disc Edema',
            'Glaucoma', 'Healthy', 'Macular Scar', 'Myopia', 'Pterygium',
            'Retinal Detachment', 'Retinitis Pigmentosa'
        ]

        predicted_class = class_names[predicted_class_index]
        print(f"[INFO] Final predicted class: {predicted_class}")

        return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        print(f"[ERROR] Exception during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)})

# تشغيل التطبيق
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # خذ البورت من البيئة أو استخدم 10000 كافتراضي
    app.run(host='0.0.0.0', port=port)
