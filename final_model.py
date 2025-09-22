# final_model.py
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from collections import Counter

# =============================================================================
# Config
# =============================================================================
DATA_ROOT   = "dataset/train"
CLASS_NAMES = ["benign", "malignant"]   # 0=benign, 1=malignant
BACKBONE    = "B2"                      # "B0"(224), "B2"(260), "B3"(300)

# Backbone image size mapping
BACKBONE_IMG = {"B0": (224, 224), "B2": (260, 260), "B3": (300, 300)}
IMG_SIZE     = BACKBONE_IMG[BACKBONE]

BATCH_SIZE      = 8
EPOCHS_WARMUP   = 8
EPOCHS_FINETUNE = 20
SEED            = 42
RECALL_FLOOR    = 0.85
UNFREEZE_LAST_N_LAYERS = 100
LR_WARMUP   = 5e-4
LR_FINETUNE = 5e-6

# Data augmentation strengths
AUG_ROT_FRAC    = 3/360
AUG_TRANS_FRAC  = 0.05
AUG_CONTRAST    = 0.10
AUG_BRIGHTNESS  = 0.05

# Focal loss params
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Make sure output dirs exist
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# Environment / reproducibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel("ERROR")
np.random.seed(SEED); tf.random.set_seed(SEED)

# =============================================================================
# Helpers
# =============================================================================
def parse_patient_id(fname: str) -> str:
    """Extract patient ID from filename."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = re.split(r"[_\-\s]+", stem)
    while parts and parts[0].lower() in ("benign","malignant","mass","calc"):
        parts = parts[1:]
    for p in parts:
        if re.match(r"[A-Za-z]*\d+", p):
            return p
    return parts[0] if parts else stem

def build_index(data_root: str) -> pd.DataFrame:
    """Build dataframe with filepath, label, patient_id."""
    rows = []
    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append((os.path.join(cls_dir, fname), label, parse_patient_id(fname)))
    return pd.DataFrame(rows, columns=["filepath", "label", "patient_id"])

def make_patient_stratified_folds(df: pd.DataFrame, desired_splits=5):
    """Generate stratified folds, grouped by patient_id where possible."""
    uniq_patients = df["patient_id"].nunique()
    if uniq_patients < desired_splits * 2:
        # fallback: redefine patient_id to unique filenames
        df = df.copy()
        df["patient_id"] = df["filepath"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
        uniq_patients = df["patient_id"].nunique()
    n_splits = min(desired_splits, max(2, uniq_patients // 2))
    try:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        folds = [(tr, va) for tr, va in skf.split(df, df["label"], groups=df["patient_id"])]
        return folds, df, n_splits
    except Exception:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return [(tr, va) for tr, va in skf.split(df, df["label"])], df, n_splits

def decode_image(path):
    """Decode image file into a float32 tensor resized to IMG_SIZE."""
    bytestr = tf.io.read_file(path)
    img = tf.io.decode_image(bytestr, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    return img

def make_dataset(paths, labels, training=False):
    """Build a tf.data.Dataset pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load(p, y): return decode_image(p), tf.cast(y, tf.int32)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def tuned_thresholds(y_true, y_prob, recall_floor=0.85):
    """Return (best_f1_threshold, high_recall_threshold)."""
    prec, rec, th = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    t_best = th[np.nanargmax(f1s)] if len(th) else 0.5
    idx = np.where(rec[:-1] >= recall_floor)[0]
    if len(idx):
        j = idx[np.argmax(prec[:-1][idx])]
        return float(t_best), float(th[j])
    return float(t_best), None

# =============================================================================
# Custom Focal Loss
# =============================================================================
@tf.function
def binary_focal_loss(y_true, y_pred, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, eps=1e-7):
    """Binary focal loss for imbalanced classification."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
    w = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
    loss = -w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
    return tf.reduce_mean(loss)

# =============================================================================
# Custom Augmentation Layer
# =============================================================================
class BrightnessJitter(layers.Layer):
    """Random brightness adjustment."""
    def __init__(self, strength=AUG_BRIGHTNESS, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    def call(self, x):
        delta = tf.random.uniform((), -self.strength, self.strength)
        return tf.clip_by_value(tf.image.adjust_brightness(x, delta), 0.0, 1.0)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"strength": self.strength})
        return cfg

# =============================================================================
# Model
# =============================================================================
def build_model():
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(AUG_ROT_FRAC),
        layers.RandomTranslation(AUG_TRANS_FRAC, AUG_TRANS_FRAC),
        layers.RandomContrast(AUG_CONTRAST),
        BrightnessJitter()
    ], name="augment")

    if BACKBONE == "B0":
        base = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling="avg"
        )
    elif BACKBONE == "B2":
        base = keras.applications.EfficientNetB2(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling="avg"
        )
    else:
        base = keras.applications.EfficientNetB3(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling="avg"
        )

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = aug(inputs)
    x = base(x, training=False)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=f"effnet_{BACKBONE.lower()}")

def compile_for(stage, model, lr):
    """Compile with Adam + focal loss + AUC/PRC metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=binary_focal_loss,
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="prc", curve="PR"),
            "accuracy"
        ]
    )

# =============================================================================
# Cross-validation run
# =============================================================================
df = build_index(DATA_ROOT)
folds, df, n_splits = make_patient_stratified_folds(df, desired_splits=5)

for fold, (tr, va) in enumerate(folds, 1):
    tr_df, va_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    train_ds = make_dataset(tr_df["filepath"].values, tr_df["label"].values, training=True)
    val_ds   = make_dataset(va_df["filepath"].values, va_df["label"].values, training=False)

    model = build_model()

    # Locate backbone (EfficientNet submodel)
    base_candidates = [l for l in model.layers if isinstance(l, keras.Model)]
    if not base_candidates:
        raise RuntimeError("Could not locate EfficientNet submodel in the final model.")
    base = base_candidates[0]

    # Warmup training
    base.trainable = False
    compile_for("warmup", model, LR_WARMUP)
    ckpt = f"models/effnet{BACKBONE.lower()}_fold{fold}.keras"
    cbs = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_prc", mode="max",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_prc", mode="max",
                                      patience=4, restore_best_weights=True, verbose=1)
    ]
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_WARMUP, callbacks=cbs, verbose=2)

    # Fine-tuning
    base.trainable = True
    for l in base.layers[:-UNFREEZE_LAST_N_LAYERS]:
        l.trainable = False
    compile_for("finetune", model, LR_FINETUNE)
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_FINETUNE, callbacks=cbs, verbose=2)

    # Save checkpoints
    native_ckpt  = f"models/effnet{BACKBONE.lower()}_fold{fold}.keras"
    weights_ckpt = f"models/effnet{BACKBONE.lower()}_fold{fold}_weights.h5"
    model.save(native_ckpt)
    model.save_weights(weights_ckpt)
    print(f"✅ Saved {native_ckpt}")
    print(f"✅ Saved {weights_ckpt} (weights-only)")
