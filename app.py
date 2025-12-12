from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================================
# 🔹 Custom Layers (harus didefinisikan ulang!)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom", name="L2Normalize")
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

@tf.keras.utils.register_keras_serializable(package="Custom", name="EuclideanDistance")
class EuclideanDistance(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), 1e-7))

# ==========================================================
# 🔹 PATH MODEL + METRICS
# ==========================================================
EMBED_PATH = "embedding_model_clean.keras"     # ganti sesuai lokasi
METRIC_PATH = "metrics.npz"                    # ganti sesuai lokasi

tf.keras.config.enable_unsafe_deserialization()

# ==========================================================
# 🔹 Load embedding model (No Lambda)
# ==========================================================
print("🔹 Loading embedding model...")
embedding_model = tf.keras.models.load_model(
    EMBED_PATH,
    compile=False,
    safe_mode=False
)
print("✅ Embedding model loaded!")

# ==========================================================
# 🔹 Load threshold metrics
# ==========================================================
if os.path.exists(METRIC_PATH):
    data = np.load(METRIC_PATH)
    eer_threshold = float(data.get("eer_threshold"))
    print(f"🔹 Loaded EER threshold: {eer_threshold}")
else:
    eer_threshold = -0.5237
    print("⚠️ metrics.npz not found — using default eer_threshold =", eer_threshold)

# ==========================================================
# 🔹 Flask init
# ==========================================================
app = Flask(__name__)

# ==========================================================
# 🔹 Preprocessing (IDENTIK training MobileNetV2)
# ==========================================================
def preprocess_inference(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image decode error")

    # --- Step 1: resize (training resize) ---
    img = cv2.resize(img, (220, 155))

    # --- Step 2: convert grayscale → RGB ---
    img_rgb = np.stack([img, img, img], axis=-1).astype("float32")

    # --- Step 3: MobileNet preprocess_input ---
    img_rgb = preprocess_input(img_rgb)

    # Expand dims → (1, H, W, 3)
    img_rgb = np.expand_dims(img_rgb, axis=0)

    return img_rgb

# ==========================================================
# 🔹 Compare Endpoint
# ==========================================================
@app.route("/compare", methods=["POST"])
def compare():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Upload image1 & image2"}), 400

    img1 = preprocess_inference(request.files["image1"])
    img2 = preprocess_inference(request.files["image2"])

    emb1 = embedding_model.predict(img1, verbose=0)[0]
    emb2 = embedding_model.predict(img2, verbose=0)[0]

    distance = np.linalg.norm(emb1 - emb2)
    similarity = -distance

    match = similarity >= eer_threshold

    return jsonify({
        "distance": float(distance),
        "similarity": float(similarity),
        "threshold": float(eer_threshold),
        "match": bool(match)
    })

# ==========================================================
# 🔹 Extract embedding
# ==========================================================
@app.route("/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "Upload image"}), 400

    img = preprocess_inference(request.files["image"])
    emb = embedding_model.predict(img, verbose=0)[0]

    return jsonify({"embedding": emb.tolist()})

# ==========================================================
# 🔹 Root
# ==========================================================
@app.route("/")
def home():
    return jsonify({"message": "Signature Embedding API Running (MobileNetV2)."})


# ==========================================================
# 🔹 Run
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
