import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Nutrition Predictor",
    layout="centered"
)

st.title("🥗 Nutrient Prediction from Food Packaging Images Using CNN")
st.write("Upload a food wrapper image to predict nutrition and health status")

# ---------------- MODEL DOWNLOAD ----------------
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE"
MODEL_PATH = "food_nutrition_cnn.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- CLASS LABELS ----------------
# ⚠️ MUST match training folder names exactly
class_labels = [
    'biscuits',
    'chips',
    'chocolate',
    'soft_drinks',
    'juice',
    'snacks'
]

# ---------------- HEALTH LOGIC ----------------
health_status = {
    'biscuits': '❌ Unhealthy',
    'chips': '❌ Unhealthy',
    'chocolate': '❌ Unhealthy',
    'soft_drinks': '❌ Unhealthy',
    'juice': '⚠ Moderately Healthy',
    'snacks': '⚠ Moderately Healthy'
}

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Food Package Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]

    # ---------------- OUTPUT ----------------
    st.subheader("🔍 Prediction Result")
    st.write(f"**Food Category:** {predicted_class}")
    st.write(f"**Health Status:** {health_status[predicted_class]}")

    st.subheader("🍽 Estimated Nutritional Impact")
    if predicted_class in ['chips', 'chocolate', 'biscuits']:
        st.write("- High Calories")
        st.write("- High Sugar & Fat")
    elif predicted_class == 'soft_drinks':
        st.write("- Very High Sugar")
        st.write("- No Essential Nutrients")
    else:
        st.write("- Moderate Calories")
        st.write("- Some Nutritional Value")
