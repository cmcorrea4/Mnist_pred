import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Cargar modelo entrenado
model = tf.keras.models.load_model("mnist_model.h5")

st.title("✍️ Detección de Dígitos Escritos (MNIST)")
st.write("Dibuja un número del 0 al 9 y presiona **Predecir**")

# Canvas para dibujar
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("🔍 Predecir"):
    if canvas_result.image_data is not None:

        # Convertir imagen del canvas a PIL
        image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        ).convert("L")

        # Redimensionar a 28x28 (MNIST)
        image = image.resize((28, 28))

        # Preprocesamiento
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predicción
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        st.subheader(f"📌 Dígito detectado: {digit}")
