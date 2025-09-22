# check_model_integrity.py
import os
import numpy as np
from tensorflow import keras

MODEL_PATH = "models/mammo_effnetb2.keras"

print(f"🔎 Checking model at: {MODEL_PATH}")

# =============================================================================
# Load model
# =============================================================================
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit(1)

# =============================================================================
# Inspect model I/O
# =============================================================================
print("Input shape :", model.inputs[0].shape)
print("Output shape:", model.outputs[0].shape)

# =============================================================================
# Run dummy inference
# =============================================================================
# Construct a blank image matching expected input
dummy = np.zeros((1, 260, 260, 3), dtype="float32")
pred = model.predict(dummy, verbose=0)

print("Prediction output:", pred)
print("Output shape from predict:", pred.shape)

# Validate prediction output
if pred.shape == (1, 1) and 0.0 <= float(pred[0][0]) <= 1.0:
    print("✅ Prediction output is valid (probability in [0,1])")
else:
    print("⚠️ Prediction output unexpected:", pred)
