import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

############################################
# Dataset paths (foggy training dataset)
############################################
FOGGY_TRAIN_DIR = r"D:/ece253/Vehicle-Type-Detection/foggy_dataset/train"
FOGGY_VAL_DIR   = r"D:/ece253/Vehicle-Type-Detection/foggy_dataset/val"

############################################
# Hyperparameters
############################################
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10          # Typical range: 5–20, depending on overfitting
LEARNING_RATE = 1e-4 # Small learning rate recommended for fine-tuning

############################################
# Data augmentation
############################################
train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1/255.0)

train_data = train_gen.flow_from_directory(
    FOGGY_TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    FOGGY_VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

############################################
# Load pretrained model (mobilenet2.h5)
############################################
print("Loading model for fine-tuning...")
model = load_model(r"D:/ece253/Vehicle-Type-Detection/saved_models/mobilenet2.h5")

############################################
# Freeze backbone and fine-tune top layers
############################################
for layer in model.layers[:-20]:   # Freeze all layers except the last 20
    layer.trainable = False

print("Number of trainable parameters:",
      sum([layer.count_params() for layer in model.layers if layer.trainable]))

############################################
# Compile the model
############################################
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

############################################
# Training
############################################
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

############################################
# Save the fine-tuned model
############################################
SAVE_PATH = r"D:/ece253/Vehicle-Type-Detection/saved_models/mobilenet2_finetuned_foggy.h5"
model.save(SAVE_PATH)
print(f"Fine-tuning completed. Model saved to: {SAVE_PATH}")
