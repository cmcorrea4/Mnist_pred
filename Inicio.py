import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

st.title("✍️ Detección de Dígitos Escritos (MNIST)")

# Verificar si el modelo existe
model_path = "mnist_model.h5"

if not os.path.exists(model_path):
    st.error(f"⚠️ No se encontró el modelo en: {model_path}")
    st.info("Por favor, asegúrate de tener el archivo 'mnist_model.h5' en el directorio raíz del proyecto")
    st.stop()

# Cargar modelo con cache para mejorar rendimiento
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
    st.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {str(e)}")
    st.stop()

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
        confidence = np.max(prediction) * 100
        
        st.subheader(f"📌 Dígito detectado: {digit}")
        st.write(f"🎯 Confianza: {confidence:.2f}%")
        
        # Mostrar probabilidades de todos los dígitos
        with st.expander("Ver todas las probabilidades"):
            for i, prob in enumerate(prediction[0]):
                st.write(f"Dígito {i}: {prob*100:.2f}%")
    else:
        st.warning("⚠️ Por favor, dibuja un dígito primero")
