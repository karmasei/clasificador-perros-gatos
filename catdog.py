from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os

# CONFIGURACIÓN
DATASET_PATH = "dataset/"
MODEL_PATH = "image_classifier.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = sorted(os.listdir(DATASET_PATH))

num_classes = len(CLASS_NAMES)
class_mode = "binary" if num_classes == 2 else "categorical"

# Cargar el modelo una sola vez
model = tf.keras.models.load_model(MODEL_PATH)

# PREPROCESAMIENTO
def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32")
    image = preprocess_input(image)

    return tf.expand_dims(image, axis=0)

# CLASIFICACIÓN
def predict_image(image_path):

    # Verificar archivo
    if not os.path.exists(image_path):
        print(f"❌ Archivo no encontrado:\n{image_path}")
        return

    try:
        img = Image.open(image_path)
        img.verify()

    except (OSError, IOError):
        print(f"❌ Imagen dañada:\n{image_path}")
        return

    # Leer imagen
    img_show = cv2.imread(image_path)

    if img_show is None:
        print("❌ No se pudo leer la imagen.")
        return

    img_display = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

    img = preprocess_image(img_show)

    # Predicción
    prediction = model.predict(img, verbose=0)

    if class_mode == "binary":

        prob_dog = float(prediction[0][0])
        prob_cat = 1 - prob_dog

        if prob_dog >= 0.5:
            predicted_class = "Perro"
            confidence = prob_dog
        else:
            predicted_class = "Gato"
            confidence = prob_cat

    else:

        index = tf.argmax(prediction, axis=-1).numpy()[0]

        predicted_class = CLASS_NAMES[index]
        confidence = prediction[0][index]

        prob_cat = 0
        prob_dog = 0

    # Nivel de confianza
    if confidence >= 0.95:
        estado = "🟢 Muy seguro"

    elif confidence >= 0.80:
        estado = "🟡 Seguro"

    elif confidence >= 0.60:
        estado = "🟠 Poca seguridad"

    else:
        estado = "🔴 Muy dudoso"

    # Mostrar resultados
    print("=" * 45)
    print("        RESULTADO DE LA PREDICCIÓN")
    print("=" * 45)

    print(f"Imagen      : {os.path.basename(image_path)}")
    print()

    print(f"Gato  : {prob_cat:.2%}")
    print(f"Perro : {prob_dog:.2%}")
    print()

    print(f"Predicción : {predicted_class}")
    print(f"Confianza  : {confidence:.2%}")
    print(f"Estado     : {estado}")

    print("=" * 45)

    # Mostrar imagen
    plt.figure(figsize=(6,6))

    plt.imshow(img_display)

    plt.title(
        f"{predicted_class}\n"
        f"Gato: {prob_cat:.1%} | "
        f"Perro: {prob_dog:.1%}\n"
        f"{estado}"
    )

    plt.axis("off")
    plt.show()

predict_image("dataset/dogs/perro1.jpeg")