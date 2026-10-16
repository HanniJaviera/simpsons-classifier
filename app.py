"""
Clasificador de personajes de Los Simpson (CNN propia, 10 clases)
App Streamlit para inferencia sobre una imagen subida por el usuario.
"""
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ----------------------------- Configuración ------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "modelo_final.keras"))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", BASE_DIR / "model" / "class_names.json"))

IMG_SIZE = (128, 128)                                   # mismo tamaño del entrenamiento
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.60))  # umbral de baja confianza

st.set_page_config(page_title="Personajes de Los Simpson", page_icon="🍩", layout="centered")


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# ------------------------------- Modelo -----------------------------------
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(model_path: Path, classes_path: Path):
    """Carga el modelo .keras y la lista de clases en el orden del entrenamiento."""
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(classes_path, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    """Replica el pipeline de entrenamiento: RGB -> resize 128x128 -> [0, 1].
    Sin data augmentation (solo se usa en train)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)   # fuerza 3 canales
    arr = tf.image.resize(arr, IMG_SIZE).numpy()             # mismo resize (bilinear) que tf.data
    arr = arr.astype("float32") / 255.0                      # normalización a [0, 1]
    return np.expand_dims(arr, axis=0)                       # (1, 128, 128, 3)


def predict(model, image: Image.Image) -> np.ndarray:
    return model.predict(preprocess(image), verbose=0)[0]


# --------------------------------- UI -------------------------------------
st.title("🍩 Personajes de Los Simpson")
st.write(
    "Sube una imagen y el modelo (CNN propia entrenada desde cero) indicará "
    "el personaje más probable y su nivel de confianza."
)

for required in (MODEL_PATH, CLASSES_PATH):
    if not required.exists():
        st.error(
            f"No se encontró `{required}`. Copia `modelo_final.keras` y "
            "`class_names.json` dentro de la carpeta `model/`."
        )
        st.stop()

model, class_names = load_model(MODEL_PATH, CLASSES_PATH)

uploaded = st.file_uploader(
    "Selecciona una imagen", type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        st.error("No se pudo leer la imagen. Prueba con otro archivo.")
        st.stop()

    st.image(image, caption="Imagen cargada", width="stretch")

    with st.spinner("Clasificando..."):
        probs = predict(model, image)

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    st.subheader("Resultado")
    col1, col2 = st.columns(2)
    col1.metric("Personaje más probable", pretty(class_names[top_idx]))
    col2.metric("Probabilidad", f"{top_prob * 100:.2f}%")

    if top_prob < CONF_THRESHOLD:
        st.warning(
            f"Baja confianza (< {CONF_THRESHOLD:.0%}): la predicción puede ser "
            "poco fiable. Puede ser un personaje atípico, una imagen parcial "
            "o un personaje que el modelo no conoce."
        )

    st.write("**Probabilidad por clase**")
    for i in np.argsort(probs)[::-1]:
        st.write(pretty(class_names[i]))
        st.progress(float(probs[i]), text=f"{probs[i] * 100:.2f}%")

    st.caption(
        "Nota: el modelo solo conoce 10 personajes. Si subes otro tipo de "
        "imagen, igualmente elegirá uno de ellos."
    )
