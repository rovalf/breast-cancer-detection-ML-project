# vgg16_finetune.py
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# 1. Data loading
# =============================================================================
train_dir  = 'dataset/train'
img_size   = (224, 224)
batch_size = 8

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=img_size,
    batch_size=batch_size, class_mode='binary',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    train_dir, target_size=img_size,
    batch_size=batch_size, class_mode='binary',
    subset='validation'
)

# =============================================================================
# 2. Model setup (VGG16 base + custom head)
# =============================================================================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze most layers
base_model.trainable = False

# Optionally unfreeze last few conv layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Build fine-tuned model
inputs = Input(shape=(224, 224, 3), name="vgg_input")
x = base_model(inputs, training=True)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs, name="cnn_model_vgg16_finetuned")

# Compile with conservative learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =============================================================================
# 3. Training
# =============================================================================
print("✅ Starting fine-tuned training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# =============================================================================
# 4. Training plots
# =============================================================================
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("static/training_plot_finetuned.png")
plt.close()

# =============================================================================
# 5. Evaluation
# =============================================================================
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
plt.savefig("static/confusion_matrix_finetuned.png")
plt.close()

# =============================================================================
# 6. Save model
# =============================================================================
model.save("transfer_learning_model.h5")
print("✅ Fine-tuned model saved as transfer_learning_model.h5")
