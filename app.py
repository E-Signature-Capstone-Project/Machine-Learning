from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import cv2

# ==========================================================
# 🔹 Daftarkan fungsi custom agar model bisa diload
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom", name="euclidean_distance")
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# ==========================================================
# 🔹 Inisialisasi Flask app
# ==========================================================
app = Flask(__name__)

# ==========================================================
# 🔹 Load model Siamese
# ==========================================================
MODEL_PATH = "signature_siamese_final.keras"  # ubah sesuai path kamu

print("🔹 Loading Siamese model...")
model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"euclidean_distance": euclidean_distance}
)
print("✅ Model loaded successfully!")

# ==========================================================
# 🔹 Dapatkan encoder dari model Siamese
# ==========================================================
try:
    encoder = model.layers[2]  # layer embedding (biasanya di posisi 2)
    print("✅ Encoder layer extracted successfully!")
except Exception as e:
    print(f"⚠️ Gagal mengambil encoder: {e}")
    encoder = None

# ==========================================================
# 🔹 Fungsi bantu: preprocessing gambar
# ==========================================================
def preprocess_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (220, 155))  # sesuaikan ukuran input model
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# ==========================================================
# 🔹 Endpoint: langsung bandingkan dua gambar
# ==========================================================
@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please upload both 'image1' and 'image2'"}), 400

    threshold = float(request.form.get("threshold", 0.5))  # bisa dikirim opsional dari postman

    img1 = preprocess_image(request.files['image1'])
    img2 = preprocess_image(request.files['image2'])

    if encoder is None:
        return jsonify({"error": "Encoder not available"}), 500

    # 1️⃣ Ekstrak embedding dari kedua gambar
    emb1 = encoder.predict(img1)[0]
    emb2 = encoder.predict(img2)[0]

    # 2️⃣ Hitung jarak Euclidean
    distance = float(np.sqrt(np.sum(np.square(emb1 - emb2))))
    match = distance < threshold

    # 3️⃣ Return hasil + embedding (untuk debugging)
    return jsonify({
        "distance": distance,
        "threshold": threshold,
        "match": bool(match),
        "embedding1": emb1.tolist(),
        "embedding2": emb2.tolist()
    })

@app.route('/extract', methods=['POST'])
def extract_embedding():
    if 'image' not in request.files:
        return jsonify({"error": "Please upload 'image'"}), 400

    img = preprocess_image(request.files['image'])

    if encoder is None:
        return jsonify({"error": "Encoder not available"}), 500

    emb = encoder.predict(img)[0]
    return jsonify({
        "embedding": emb.tolist()
    })


# ==========================================================
# 🔹 Endpoint dasar
# ==========================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "✅ Siamese Compare API running.",
        "endpoint": "/compare",
        "usage": "POST form-data: image1, image2, (optional) threshold"
    })

# ==========================================================
# 🔹 Run server
# ==========================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
