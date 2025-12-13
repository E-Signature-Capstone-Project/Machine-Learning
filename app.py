from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================================
# 🔹 Custom Layers (WAJIB sama dengan training)
# ==========================================================
@tf.keras.utils.register_keras_serializable(package="Custom", name="L2Normalize")
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

@tf.keras.utils.register_keras_serializable(package="Custom", name="EuclideanDistance")
class EuclideanDistance(tf.keras.layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(
            tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), 1e-7)
        )

# ==========================================================
# 🔹 CONFIG
# ==========================================================
IMG_H, IMG_W = 155, 220

EMBED_PATH  = "embedding_model_clean.keras"
METRIC_PATH = "metrics.npz"

STRICT_MARGIN = 0.15   # 🔥 penting → bikin sistem tidak permisif

# ==========================================================
# 🔹 Load Embedding Model
# ==========================================================
print("🔹 Loading embedding model...")
embedding_model = tf.keras.models.load_model(
    EMBED_PATH,
    compile=False
)
print("✅ Embedding model loaded")

# ==========================================================
# 🔹 Load Metrics & Threshold
# ==========================================================
if os.path.exists(METRIC_PATH):
    data = np.load(METRIC_PATH)
    eer_threshold = float(data["eer_threshold"])
else:
    eer_threshold = -0.55  # fallback aman

STRICT_THRESHOLD = eer_threshold - STRICT_MARGIN

print("🔹 EER threshold     :", eer_threshold)
print("🔹 STRICT threshold  :", STRICT_THRESHOLD)

# ==========================================================
# 🔹 Flask init
# ==========================================================
app = Flask(__name__)

# ==========================================================
# 🔹 PREPROCESS — IDENTIK DENGAN TRAINING
# ==========================================================
def preprocess_signature_file(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Cannot decode image")

    # ==============================
    # 1. Handle channel
    # ==============================
    if len(img.shape) == 2:
        gray = img

    elif img.shape[2] == 4:
        rgb = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        white = np.ones_like(rgb) * 255
        blended = rgb * alpha[..., None] + white * (1 - alpha[..., None])
        blended = blended.astype(np.uint8)
        gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==============================
    # 2. Denoise
    # ==============================
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ==============================
    # 3. Binarize
    # ==============================
    _, bw = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ==============================
    # 4. Extract signature
    # ==============================
    cnts, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        raise ValueError("No signature found")

    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    sig = bw[y:y+h, x:x+w]

    # ==============================
    # 5. Resize + Center Pad
    # ==============================
    scale = min(
        (IMG_H - 10) / max(h, 1),
        (IMG_W - 10) / max(w, 1)
    )
    new_h, new_w = int(h * scale), int(w * scale)
    sig = cv2.resize(sig, (new_w, new_h))

    canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    top  = (IMG_H - new_h) // 2
    left = (IMG_W - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = sig

    # ==============================
    # 6. Normalize + MobileNet
    # ==============================
    canvas = canvas.astype("float32") / 255.0
    canvas = np.stack([canvas, canvas, canvas], axis=-1)
    canvas = preprocess_input(canvas * 255.0)

    return np.expand_dims(canvas, axis=0)

# ==========================================================
# 🔹 Compare Endpoint (STRICT)
# ==========================================================
@app.route("/compare", methods=["POST"])
def compare():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "image1 and image2 required"}), 400

    img1 = preprocess_signature_file(request.files["image1"])
    img2 = preprocess_signature_file(request.files["image2"])

    emb1 = embedding_model.predict(img1, verbose=0)[0]
    emb2 = embedding_model.predict(img2, verbose=0)[0]

    distance = float(np.linalg.norm(emb1 - emb2))
    similarity = -distance

    match = similarity >= STRICT_THRESHOLD

    return jsonify({
        "distance": distance,
        "similarity": similarity,
        "eer_threshold": eer_threshold,
        "strict_threshold": STRICT_THRESHOLD,
        "match": bool(match)
    })

# ==========================================================
# 🔹 Extract Embedding
# ==========================================================
@app.route("/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    img = preprocess_signature_file(request.files["image"])
    emb = embedding_model.predict(img, verbose=0)[0]

    return jsonify({"embedding": emb.tolist()})

# ==========================================================
# 🔹 Root
# ==========================================================
@app.route("/")
def home():
    return jsonify({
        "message": "Signature Verification API (STRICT MODE)",
        "eer_threshold": eer_threshold,
        "strict_threshold": STRICT_THRESHOLD
    })

# ==========================================================
# 🔹 Run
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
