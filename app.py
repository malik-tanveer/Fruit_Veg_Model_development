import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Model Page config
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    layout="centered"
)

# Model
MODEL_PATH = "Fruit_Veg_Model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_HEIGHT = 180
IMG_WIDTH = 180

# Return classes names
@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# Load Trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Load the models and labels

try:
    class_names = load_class_names()
    model = load_model()
except Exception as error:
    st.error(f"Could not load model or class names: {error}")
    st.stop()

# UI
st.title("Fruit & Vegetable Classifier")
st.write(
    "Upload an image of a fruit or vegetable "
    "to predict its category using Deep Learning."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Left: Predictions | Right: Image
    right_col,left_col = st.columns([1, 1])

    with right_col:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with left_col:
        st.subheader("Predictions")

        with st.spinner("Analyzing image..."):
            # Resize image
            image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))

            # Convert image to NumPy
            image_array = np.array(image_resized, dtype=np.float32)

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            predictions = model.predict(image_array, verbose=0)

            # Convert logits to probabilities
            probabilities = tf.nn.softmax(predictions[0]).numpy()

            # Get top 3 classes
            top_indices = np.argsort(probabilities)[-3:][::-1]

        for rank, index in enumerate(top_indices, start=1):
            class_name = class_names[int(index)]
            confidence = float(probabilities[index])

            st.markdown(f"**{rank}. {class_name.title()}**")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence * 100:.2f}%")

# Footer
st.markdown(
    """
    <div style="
        text-align: center;
        color: #888888;
        padding: 25px 0 10px 0;
        border-top: 1px solid #303030;
        margin-top: 40px;
        font-size: 13px;
    ">
        Fruit & Vegetable Classification Model
    </div>
    """,
    unsafe_allow_html=True
)