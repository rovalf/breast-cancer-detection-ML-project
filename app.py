# app.py
from flask import (
    Flask, request, render_template, jsonify,
    redirect, url_for, session, send_file
)
import os, time, datetime as dt, logging
import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

# Silence verbose TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

try:
    import cv2
except Exception:
    cv2 = None  # Grad-CAM skipped if cv2 missing

# =============================================================================
# Flask setup
# =============================================================================
app = Flask(__name__, template_folder="frontend", static_folder="static")
app.secret_key = os.urandom(24)  # new each run → re-login required
app.config["SESSION_PERMANENT"] = False
os.makedirs(app.static_folder, exist_ok=True)

# Demo users
USERS = {
    "admin": generate_password_hash("admin123"),
    "radiologist": generate_password_hash("rad456"),
}

def is_authed() -> bool:
    return "user" in session

# =============================================================================
# Utilities
# =============================================================================
def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,255] uint8."""
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)

def pil_from_dicom(file_storage) -> Image.Image:
    """Convert uploaded DICOM into RGB PIL image."""
    dcm = pydicom.dcmread(file_storage, force=True)
    if "PixelData" not in dcm:
        raise ValueError("Invalid DICOM: missing PixelData")
    px = apply_voi_lut(dcm.pixel_array, dcm)
    if getattr(dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        px = px.max() - px
    return Image.fromarray(_to_uint8(px)).convert("RGB")

def _user_preview_dir():
    """Preview directory per user under /static/previews/"""
    d = os.path.join(app.static_folder, "previews", session["user"])
    os.makedirs(d, exist_ok=True)
    return d

# Keras compatibility patch
class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

class BrightnessJitter(layers.Layer):
    """Custom brightness jitter layer from training."""
    def __init__(self, strength=0.05, **kwargs):
        super().__init__(**kwargs)
        self.strength = float(strength)

    def call(self, x, training=None):
        if training:
            delta = tf.random.uniform((), -self.strength, self.strength)
            return tf.clip_by_value(tf.image.adjust_brightness(x, delta), 0.0, 1.0)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"strength": self.strength})
        return cfg

# =============================================================================
# Model loading
# =============================================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "models/mammo_effnetb2.keras")

def _infer_backbone_size(path: str):
    p = path.lower()
    if "b0" in p: return "B0", (224, 224)
    if "b3" in p: return "B3", (300, 300)
    return "B2", (260, 260)

BACKBONE, DEFAULT_INPUT = _infer_backbone_size(MODEL_PATH)
_loaded_model = None
_input_size = DEFAULT_INPUT

def _build_inference_graph(backbone: str, img_size):
    if backbone == "B0":
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                 input_shape=(img_size[0], img_size[1], 3),
                                                 pooling="avg")
    elif backbone == "B3":
        base = keras.applications.EfficientNetB3(include_top=False, weights="imagenet",
                                                 input_shape=(img_size[0], img_size[1], 3),
                                                 pooling="avg")
    else:
        base = keras.applications.EfficientNetB2(include_top=False, weights="imagenet",
                                                 input_shape=(img_size[0], img_size[1], 3),
                                                 pooling="avg")
    inp = keras.Input(shape=(img_size[0], img_size[1], 3), name="inference_input")
    x = base(inp, training=False)
    x = layers.Dropout(0.4, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out, name=f"effnet_{backbone.lower()}_classifier_infer")

def _try_load_native_or_legacy():
    """Attempt load: .keras → .h5 safe → .h5 unsafe."""
    last_err = None
    abspath = os.path.abspath(MODEL_PATH)
    print(f"[Model] Attempting load from: {abspath}")

    if MODEL_PATH.lower().endswith(".keras"):
        try:
            m = keras.models.load_model(
                MODEL_PATH, compile=False,
                custom_objects={
                    "DepthwiseConv2D": PatchedDepthwiseConv2D,
                    "BrightnessJitter": BrightnessJitter,
                }
            )
            return m, (int(m.inputs[0].shape[1]), int(m.inputs[0].shape[2]))
        except Exception as e:
            last_err = e
            print(f"[Model] .keras load failed: {e}")

    if MODEL_PATH.lower().endswith(".h5"):
        try:
            m = load_model(
                MODEL_PATH, compile=False, safe_mode=True,
                custom_objects={
                    "DepthwiseConv2D": PatchedDepthwiseConv2D,
                    "BrightnessJitter": BrightnessJitter,
                }
            )
            return m, (int(m.inputs[0].shape[1]), int(m.inputs[0].shape[2]))
        except Exception as e:
            last_err = e
            print(f"[Model] .h5 safe_mode=True failed: {e}")

        try:
            m = load_model(
                MODEL_PATH, compile=False, safe_mode=False,
                custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D}
            )
            return m, (int(m.inputs[0].shape[1]), int(m.inputs[0].shape[2]))
        except Exception as e:
            last_err = e
            print(f"[Model] .h5 safe_mode=False failed: {e}")

    raise last_err if last_err else RuntimeError("No compatible model format found")

def get_model():
    """Lazy-load model; rebuild if deserialization fails."""
    global _loaded_model, _input_size
    if _loaded_model is not None:
        return _loaded_model

    t0 = time.time()
    try:
        m, size = _try_load_native_or_legacy()
        _loaded_model, _input_size = m, size
        print(f"[Model] Loaded in {time.time()-t0:.2f}s @ input {_input_size}")
        return _loaded_model
    except Exception as e:
        print(f"[Model] Deserialization failed; fallback: {e}")

    m = _build_inference_graph(BACKBONE, DEFAULT_INPUT)
    try:
        m.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        print("[Model] Weights loaded by name (skip_mismatch=True)")
    except Exception as e2:
        print(f"[Model] load_weights by name failed: {e2}")
        try:
            m.load_weights(MODEL_PATH)
            print("[Model] Weights loaded strictly")
        except Exception as e3:
            print(f"[Model] Strict load_weights failed: {e3}")
    _loaded_model, _input_size = m, DEFAULT_INPUT
    print(f"[Model] Ready (fallback) in {time.time()-t0:.2f}s @ input {_input_size}")
    return _loaded_model

def _preprocess_pil(pil_img: Image.Image):
    """Resize/normalize PIL image for model input."""
    w, h = _input_size
    pil_img = pil_img.convert("RGB").resize((w, h))
    arr = np.asarray(pil_img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# =============================================================================
# Grad-CAM
# =============================================================================
def _get_effnet_base(model):
    for name in ("efficientnetb0", "efficientnetb2", "efficientnetb3"):
        try:
            return model.get_layer(name)
        except Exception:
            pass
    for lyr in model.layers:
        if isinstance(lyr, tf.keras.Model) and "efficientnet" in lyr.name.lower():
            return lyr
    return None

def _copy_weights_by_name(src_model, dst_model):
    src = {l.name: l for l in src_model.layers}
    copied = skipped = 0
    for dl in dst_model.layers:
        sl = src.get(dl.name)
        if sl and sl.weights and dl.weights:
            try:
                dl.set_weights(sl.get_weights())
                copied += 1
            except Exception:
                skipped += 1
        else:
            skipped += 1
    return copied, skipped

def _build_joint_gradcam_model(model, img_size):
    """Build graph mapping input -> (conv_maps, prediction)."""
    base_src = _get_effnet_base(model)
    if base_src is None:
        return None, "EfficientNet base not found"

    feat = tf.keras.applications.EfficientNetB2(
        include_top=False, weights=None,
        input_shape=(img_size[0], img_size[1], 3), pooling=None
    )
    copied, skipped = _copy_weights_by_name(base_src, feat)
    print(f"[GradCAM] weight copy: copied={copied}, skipped={skipped}")

    conv_maps = feat(model.input, training=False)

    try:
        head_dropout = model.get_layer("dropout")
        head_dense = model.get_layer("output")
    except Exception as e:
        return None, f"Missing head layers: {e}"

    gap = tf.keras.layers.GlobalAveragePooling2D(name="gradcam_gap")(conv_maps)
    logits = head_dense(head_dropout(gap, training=False))
    joint = tf.keras.Model(inputs=model.input, outputs=[conv_maps, logits], name="gradcam_joint")
    return joint, None

def save_gradcam(model, input_png_path, output_png_path):
    """Generate Grad-CAM heatmap and save blended overlay."""
    if cv2 is None:
        return False, "OpenCV not installed"

    img = Image.open(input_png_path).convert("RGB").resize(_input_size)
    x = np.expand_dims(np.asarray(img).astype("float32") / 255.0, axis=0)
    xt = tf.convert_to_tensor(x)

    joint, err = _build_joint_gradcam_model(model, _input_size)
    if joint is None:
        return False, f"Grad-CAM setup failed: {err}"

    with tf.GradientTape() as tape:
        conv_out, preds = joint(xt, training=False)
        target = preds[:, 0] if preds.shape[-1] == 1 else preds[:, tf.argmax(preds[0])]
        grads = tape.gradient(target, conv_out)[0]

    fmap = np.asarray(conv_out[0])
    grads = np.asarray(grads)
    weights = grads.mean(axis=(0, 1))
    cam = np.maximum((fmap * weights).sum(axis=-1), 0)
    cam = cam / (cam.max() + 1e-8)

    cam = cv2.resize(cam.astype(np.float32), _input_size)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(np.array(img), 0.6, heat, 0.4, 0)
    Image.fromarray(blend).save(output_png_path)
    return True, None

# =============================================================================
# Routes
# =============================================================================
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("login")) if not is_authed() else redirect(url_for("dashboard"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pwd = request.form.get("password", "")
        if user in USERS and check_password_hash(USERS[user], pwd):
            session["user"] = user
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET"])
def dashboard():
    if not is_authed():
        return redirect(url_for("login"))
    return render_template("index.html")

# ---- Convert DICOM → PNG ----
@app.route("/convert_preview", methods=["POST"])
def convert_preview():
    if not is_authed():
        return jsonify({"error": "Please log in."}), 401
    f = request.files.get("dicom_file")
    if not f or f.filename == "":
        return jsonify({"error": "No DICOM uploaded"}), 400

    try:
        dcm = pydicom.dcmread(f, force=True)
        if "PixelData" not in dcm:
            return jsonify({"error": "Invalid DICOM: missing PixelData"}), 400
        px = apply_voi_lut(dcm.pixel_array, dcm)
        if getattr(dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            px = px.max() - px
        pil = Image.fromarray(_to_uint8(px)).convert("RGB")

        token = uuid.uuid4().hex
        out_dir = _user_preview_dir()
        out_name = f"converted_{token}.png"
        out_path = os.path.join(out_dir, out_name)
        pil.save(out_path, format="PNG")

        preview_url = url_for("static", filename=f"previews/{session['user']}/{out_name}")
        download_url = url_for("download_converted", token=token)
        return jsonify({"preview_url": preview_url, "download_url": download_url, "token": token})
    except Exception as e:
        return jsonify({"error": f"Conversion failed: {e}"}), 500

@app.route("/download_converted/<token>", methods=["GET"])
def download_converted(token):
    if not is_authed():
        return redirect(url_for("login"))
    path = os.path.join(app.static_folder, "previews", session["user"], f"converted_{token}.png")
    if not os.path.exists(path):
        return "Preview not found. Please convert again.", 404
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(path, mimetype="image/png", as_attachment=True,
                     download_name=f"dicom_to_png_{ts}.png")

# ---- Prediction ----
@app.route("/predict", methods=["POST"])
def predict():
    if not is_authed():
        return jsonify({"error": "Please log in to use prediction."}), 401
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    try:
        name = file.filename.lower()
        if name.endswith(".dcm"):
            pil = pil_from_dicom(file)
        else:
            pil = Image.open(file.stream).convert("RGB")

        model = get_model()
        x = _preprocess_pil(pil)

        prob = float(model.predict(x, verbose=0).reshape(-1)[0])
        label = "Malignant" if prob > 0.5 else "Benign"
        conf  = prob if prob > 0.5 else 1 - prob

        os.makedirs(os.path.join(app.static_folder, "previews"), exist_ok=True)
        last_png = os.path.join(app.static_folder, "previews", "last_uploaded.png")
        pil.resize(_input_size).save(last_png)

        grad_png = os.path.join(app.static_folder, "previews", "gradcam.png")
        ok, _ = save_gradcam(model, last_png, grad_png)
        grad_url = url_for("static", filename="previews/gradcam.png") if ok else None

        return jsonify({
            "prediction": label,
            "confidence": f"{conf * 100:.2f}%",
            "image_path": url_for("static", filename="previews/last_uploaded.png"),
            "gradcam_image": grad_url
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route("/evaluation", methods=["GET"])
def evaluation():
    if not is_authed():
        return redirect(url_for("login"))
    return render_template("evaluation.html")

if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"[App] MODEL_PATH={MODEL_PATH} | backbone={BACKBONE} | default_input={DEFAULT_INPUT}")
    app.run(debug=True)
