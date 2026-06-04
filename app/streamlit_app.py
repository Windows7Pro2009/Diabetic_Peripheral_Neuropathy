import os
import sys

import streamlit as st

# Allow imports from project root when run via "streamlit run app/streamlit_app.py"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.model_utils import decode_foot_prediction, load_trained_model
from utils.preprocess import preprocess_uploaded_pil_image


MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "dpn_mobilenetv2.keras")


@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


def predict_uploaded_image(model, uploaded_file):
    from PIL import Image

    image = Image.open(uploaded_file)
    processed = preprocess_uploaded_pil_image(image)
    prob = float(model.predict(processed, verbose=0)[0][0])
    label, confidence, prob_neuropathy = decode_foot_prediction(prob)
    return image, label, confidence, prob_neuropathy


def main():
    st.set_page_config(page_title="DPN Detection", page_icon=":hospital:")
    st.title("AI-Based Detection of Diabetic Peripheral Neuropathy")
    st.write("Upload a foot image to classify as Normal or Possible DPN.")

    if not os.path.exists(MODEL_PATH):
        st.warning(
            f"Model file not found at `{MODEL_PATH}`. Train model first using `python train.py`."
        )
        return

    model = get_model()
    uploaded_file = st.file_uploader(
        "Upload foot image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image, label, confidence, prob = predict_uploaded_image(model, uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader("Prediction Result")
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
        st.caption(f"Raw neuropathy probability: {prob:.4f}")


if __name__ == "__main__":
    main()
