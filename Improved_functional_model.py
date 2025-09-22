
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

# === 1. Prepare Data (No CLAHE) ===
train_dir = 'dataset/train'
img_size = (224, 224)
batch_size = 8

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

print("Train class distribution:", Counter(train_gen.classes))
print("Validation class distribution:", Counter(val_gen.classes))

# === 2. Compute Class Weights ===
class_weights_dict = {0: 2.0, 1: 1.0}  # Manually boosting benign

# === 3. Define CNN Model ===
inputs = Input(shape=(224, 224, 3), name="functional_input")
x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0001))(inputs)
x = MaxPooling2D(2, 2)(x)

x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0001))(x)
x = MaxPooling2D(2, 2)(x)
x = Dropout(0.4)(x)

x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.0001))(x)
x = MaxPooling2D(2, 2)(x)

x = Flatten()(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs, name="cnn_final_regularized")
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === 4. Train Model ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop],
    class_weight=class_weights_dict
)

# === 5. Evaluation ===
val_gen.reset()
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred_labels, target_names=['Benign', 'Malignant']))

cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# === 6. Save Model ===
model.save("cnn_model_final.h5")
print("✅ Model saved as cnn_model_final.h5")
