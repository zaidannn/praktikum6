import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained model
@st.cache_resource  # Caches the model loading process for faster performance
def load_trained_model():
    model = load_model('final_model.h5')  # Ganti nama file model Anda jika berbeda
    return model

model = load_trained_model()

# Class labels
class_labels = {
    0: "Boxer",
    1: "Dachshund",
    2: "Golden Retriever",
    3: "Poodle",
    4: "Rottweiler"
}

# Dashboard title
st.title("Dog Breed Prediction Dashboard")

# File upload section
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    def preprocess_image(img):
        img = img.resize((224, 224))  # Resize to model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image
        return img_array

    # Convert uploaded file to PIL image
    img = Image.open(uploaded_file)

    # Predict button
    if st.button("Predict"):
        # Preprocess the image
        processed_img = preprocess_image(img)

        # Perform prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Display prediction results
        st.subheader("Prediction Results")
        st.write(f"Predicted Breed: **{class_labels[predicted_class]}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
