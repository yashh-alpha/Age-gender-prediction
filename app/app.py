import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "models/age_gender_model.keras"
    )

model = load_my_model()


def preprocess_image(image):

    image = image.resize((224, 224))

    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image




st.title("Age & Gender Prediction")

st.write(
    "Upload a face image and predict age and gender."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        img = preprocess_image(
            image
        )

        age_pred, gender_pred = model.predict(
            img,
            verbose=0
        )

        age = int(age_pred[0][0])

        gender_score = gender_pred[0][0]

        gender = (
            "Female"
            if gender_score > 0.5
            else "Male"
        )

        st.success(
            f"Predicted Age: {age}"
        )

        st.success(
            f"Predicted Gender: {gender}"
        )

        st.write(
            f"Gender Confidence: {gender_score:.2f}"
        )