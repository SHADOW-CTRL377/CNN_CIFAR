import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import joblib

model = joblib.load("YOUR_MODEL_NAME.pkl")
# 2. CIFAR-10 class names
classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # 3. Preprocess the image to match CIFAR-10 shape (32x32x3)
    image = image.resize((32, 32))
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 4. Make prediction
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
