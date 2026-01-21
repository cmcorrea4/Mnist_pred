import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(
    page_title="Detector de Dígitos MNIST",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Detección de Dígitos Escritos a Mano")
st.markdown("**Modelo entrenado con CNN en Google Colab**")

# Cargar modelo con manejo de errores
@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST con manejo robusto de errores"""
    
    # Lista de posibles ubicaciones y formatos del modelo
    model_paths = [
        "mnist_model.h5",
        "mnist_model.keras",
        "mnist_model_savedmodel",
        "mnist_model"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path, compile=True)
                st.success(f"✅ Modelo cargado desde: `{model_path}`")
                return model
            except Exception as e:
                st.warning(f"⚠️ Error al cargar {model_path}: {e}")
                continue
    
    # Si no se encuentra ningún modelo
    st.error("❌ No se encontró el modelo MNIST")
    st.info("""
    **Instrucciones:**
    1. Entrena el modelo en Google Colab usando el código proporcionado
    2. Descarga el archivo `mnist_model.h5`
    3. Súbelo a tu repositorio en la carpeta raíz
    4. Reinicia la aplicación
    """)
    st.stop()

# Cargar modelo
model = load_mnist_model()

# Información del modelo en sidebar
with st.sidebar:
    st.header("ℹ️ Información")
    st.write(f"**TensorFlow:** {tf.__version__}")
    st.write(f"**Parámetros:** {model.count_params():,}")
    
    st.divider()
    
    st.header("💡 Consejos")
    st.markdown("""
    - Dibuja números **grandes** y **centrados**
    - Usa **trazos gruesos**
    - Asegúrate de que el dígito sea **claro**
    - Prueba varios dígitos para ver la precisión
    """)
    
    st.divider()
    
    st.header("🎯 Dataset")
    st.write("**MNIST** - 60,000 imágenes de entrenamiento")
    st.write("Dígitos del 0 al 9")

# Diseño principal
st.write("### 🎨 Dibuja un dígito del 0 al 9")

col1, col2 = st.columns([2, 1])

with col1:
    # Canvas para dibujar
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("### 🎮 Controles")
    
    predict_btn = st.button("🔍 **Predecir**", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Limpiar", use_container_width=True)
    
    if clear_btn:
        st.rerun()

# Predicción
if predict_btn:
    if canvas_result.image_data is not None:
        # Verificar si hay algo dibujado
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ Por favor, dibuja un dígito primero")
        else:
            with st.spinner("🤔 Analizando tu dibujo..."):
                # Convertir imagen del canvas
                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")
                
                # Redimensionar a 28x28
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Preprocesamiento
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predicción
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
            
            # Mostrar resultados
            st.success(f"## 🎯 Dígito detectado: **{digit}**")
            
            # Métricas
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Predicción", digit)
            with col_m2:
                st.metric("Confianza", f"{confidence:.1f}%")
            with col_m3:
                alternative = np.argsort(prediction[0])[-2]
                st.metric("2ª opción", alternative)
            
            # Mostrar imágenes
            st.write("### 🖼️ Procesamiento")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.write("**Tu dibujo original:**")
                st.image(canvas_result.image_data, width=200)
            
            with col_img2:
                st.write("**Imagen procesada (28x28):**")
                st.image(image_resized, width=200)
            
            # Gráfico de barras de probabilidades
            st.write("### 📊 Probabilidades por dígito")
            
            # Crear DataFrame para visualización
            import pandas as pd
            prob_df = pd.DataFrame({
                'Dígito': [str(i) for i in range(10)],
                'Probabilidad (%)': prediction[0] * 100
            })
            
            st.bar_chart(prob_df.set_index('Dígito'))
            
            # Tabla detallada (expandible)
            with st.expander("🔍 Ver detalles de todas las probabilidades"):
                for i, prob in enumerate(prediction[0]):
                    emoji = "🎯" if i == digit else ""
                    st.write(f"{emoji} **Dígito {i}**: {prob*100:.2f}%")
    else:
        st.warning("⚠️ Por favor, dibuja un dígito en el canvas")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Modelo CNN entrenado en MNIST Dataset | Desarrollado con TensorFlow y Streamlit</small>
</div>
""", unsafe_allow_html=True)
