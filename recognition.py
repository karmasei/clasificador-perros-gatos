from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
import os

# Cargar Dataset

DATASET_PATH = "dataset/"

num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

# Generador para entrenamiento
# (Con Data Augmentation proporcionado por ChatGPT)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Generador para validación

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Datos de entrenamiento

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=8,
    class_mode=class_mode,
    subset="training",
    shuffle=True
)

print(train_data.classes)
print(train_data.class_indices)

# Datos de validación

val_data = val_datagen.flow_from_directory(   # <-- MODIFICACIÓN AQUÍ
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=8,
    class_mode=class_mode,
    subset="validation",
    shuffle=False
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Modelo CNN

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(32, activation="relu"),

    Dropout(0.2),

    Dense(1, activation="sigmoid")
    if class_mode == "binary"
    else Dense(num_classes, activation="softmax")

])

# Compilación

loss_function = (
    "binary_crossentropy"
    if class_mode == "binary"
    else "categorical_crossentropy"
)

model.compile(
    optimizer="adam",
    loss=loss_function,
    metrics=["accuracy"]
)

# EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Entrenamiento
model.summary()

model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stop],
    verbose=1
)

# Evaluación

test_loss, test_accuracy = model.evaluate(
    val_data,
    verbose=1
)

print(f"\nPrecisión del modelo: {test_accuracy*100:.2f}%")

# Guardar modelo

model.save("image_classifier.keras")
