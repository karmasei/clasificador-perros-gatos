# 🐱🐶 Clasificador Inteligente de Imágenes con TensorFlow

## 📌 Descripción
Este proyecto consiste en un clasificador de imágenes desarrollado con Python y TensorFlow, capaz de identificar si una imagen corresponde a un gato o un perro mediante el uso de una red neuronal basada en Transfer Learning con MobileNetV2.

El modelo fue entrenado utilizando un conjunto de imágenes organizado por categorías y posteriormente optimizado para realizar predicciones sobre nuevas fotografías. Además de indicar la clase detectada, el sistema calcula el porcentaje de confianza de la predicción y clasifica el resultado según su nivel de certeza, permitiendo interpretar con mayor facilidad la fiabilidad del modelo.

Como complemento, el programa valida que la imagen exista y no esté dañada antes de procesarla, realiza el preprocesamiento necesario para la red neuronal y muestra visualmente el resultado junto con los porcentajes de probabilidad para cada categoría.

Este proyecto fue desarrollado con el objetivo de aprender los fundamentos de la visión por computadora, el entrenamiento de modelos de clasificación de imágenes y la implementación de técnicas modernas de aprendizaje profundo mediante Transfer Learning.

## 🔧 Tecnologías
- Python 3
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- Pillow (PIL)
- Matplotlib
- NumPy

## 📝 Funciones 
1. Clasificación automática de gatos y perros.
2. Uso de Transfer Learning con MobileNetV2.
3. Preprocesamiento automático de imágenes.
4. Validación de archivos antes de la predicción.
5. Cálculo del porcentaje de confianza de cada clase.
6. Clasificación del nivel de confianza (Muy seguro, Seguro, Poca seguridad o Muy dudoso).
7. Visualización de la imagen junto con el resultado obtenido.
8. Presentación de probabilidades individuales para cada categoría.
9. Carga del modelo entrenado para realizar inferencias de forma eficiente.

## 👨‍💻 Autor
Karma Sei
