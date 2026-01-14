import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Food Nutrition Predictor", layout="centered")

st.title("🥗 Nutrient Prediction from Food Packaging Images")
st.write("Upload a food wrapper image to check health status")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("food_nutrition_cnn.h5")

model = load_model()

# Class labels (MUST match training folders)
class_labels = [
    'biscuits',
    'chips',
    'chocolate',
    'soft_drinks',
    'juice',
    'snacks'
]

# Health mapping
health_status = {
    'biscuits': '❌ Unhealthy',
    'chips': '❌ Unhealthy',
    'chocolate': '❌ Unhealthy',
    'soft_drinks': '❌ Unhealthy',
    'juice': '⚠ Moderately Healthy',
    'snacks': '⚠ Moderately Healthy'
}

# Image upload
uploaded_file = st.file_uploader("Upload Food Package Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Food Category:** {predicted_class}")
    st.write(f"**Health Status:** {health_status[predicted_class]}")

    # Simple nutrient display
    st.subheader("🍽 Estimated Nutrition")
    if predicted_class in ['chips', 'chocolate', 'biscuits']:
        st.write("- High Calories")
        st.write("- High Sugar / Fat")
    elif predicted_class == 'soft_drinks':
        st.write("- High Sugar")
        st.write("- No Nutritional Value")
    else:
        st.write("- Moderate Calories")
        st.write("- Some Nutritional Value")
