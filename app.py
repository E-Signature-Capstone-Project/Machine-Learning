from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

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
EMBED_PATH = "embedding_model_clean.keras"
METRIC_PATH = "metrics.npz"

# ==========================================================
# 🔹 Load embedding model (tanpa Lambda, aman)
# ==========================================================
print("🔹 Loading embedding model...")
embedding_model = tf.keras.models.load_model(
    EMBED_PATH,
    compile=False
)
print("✅ Embedding model loaded!")

# ==========================================================
# 🔹 Load threshold dari metrics.npz
# ==========================================================
if os.path.exists(METRIC_PATH):
    data = np.load(METRIC_PATH)
    eer_threshold = float(data.get("eer_threshold", -0.5250))
    print(f"🔹 Loaded EER threshold: {eer_threshold}")
else:
    eer_threshold = -0.5250
    print("⚠️ metrics.npz not found, default eer_threshold = -0.52")

# ==========================================================
# 🔹 Flask init
# ==========================================================
app = Flask(__name__)

# ==========================================================
# 🔹 Preprocessing (IDENTIK dengan training!)
# ==========================================================
def preprocess_inference(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image decode error")

    # Step 1: blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 2: binarize
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: crop ke kontur terbesar
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        sig = img_bin[y:y + h, x:x + w]
    else:
        sig = img_bin

    target_h, target_w = 155,220
    h0, w0 = sig.shape
    scale = min((target_h - 10) / max(h0, 1), (target_w - 10) / max(w0, 1))

    new_h = max(1, int(h0 * scale))
    new_w = max(1, int(w0 * scale))
    sig = cv2.resize(sig, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 4: padding
    pad_v = target_h - new_h
    top = pad_v // 2
    bottom = pad_v - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    sig = np.pad(sig, ((top, bottom), (left, right)), mode='constant', constant_values=0)

    sig = sig.astype("float32") / 255.0
    sig = np.expand_dims(sig, axis=(0, -1))     # (1, 220, 155, 1)
    return sig

# ==========================================================
# 🔹 Compare Endpoint (pakai distance -> similarity)
# ==========================================================
@app.route("/compare", methods=["POST"])
def compare():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Upload image1 & image2"}), 400

    img1 = preprocess_inference(request.files["image1"])
    img2 = preprocess_inference(request.files["image2"])

    emb1 = embedding_model.predict(img1)[0]
    emb2 = embedding_model.predict(img2)[0]

    # Euclidean distance (training model pakai ini)
    distance = np.linalg.norm(emb1 - emb2)

    # Convert to similarity (harus NEGATIVE distance)
    similarity = -distance

    # Decision using EER threshold
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
    emb = embedding_model.predict(img)[0]

    return jsonify({"embedding": emb.tolist()})

@app.route("/")
def home():
    return jsonify({"message": "Signature Embedding API Running."})

# ==========================================================
# 🔹 Run
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
