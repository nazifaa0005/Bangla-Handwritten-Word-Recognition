"""Streamlit UI for the Assignment 04 Bangla handwritten-word recognition project.

Uses the existing predict_word.py pipeline and trained Keras checkpoint.
Run from the project root with: python -m streamlit run app.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from predict_word import MODEL_PATH, load_character_mapping, predict_word

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
    CANVAS_ERROR = ""
except Exception as exc:
    st_canvas = None
    CANVAS_AVAILABLE = False
    CANVAS_ERROR = str(exc)


st.set_page_config(page_title="Bangla Handwriting Recognition", page_icon="✍️", layout="centered")


@st.cache_resource(show_spinner="Loading the CNN model...")
def get_model():
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def get_mapping():
    mapping = load_character_mapping()
    if len(mapping) != 84:
        raise ValueError(f"Expected 84 character labels, got {len(mapping)}")
    return mapping


def predict_image(gray_image, model, mapping):
    """Reuse CLI inference by temporarily saving a grayscale PNG.

    predict_word() reads a BLACK-on-WHITE image from disk, inverts it,
    and calls the project's existing segment_word()/preprocess_image().
    """
    gray_image = np.asarray(gray_image, dtype=np.uint8)
    # Normalize foreground/background to avoid semi-transparent canvas artefacts.
    _, clean_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(255 - clean_image) < 50:
        raise ValueError("No handwriting detected. Please draw a word first.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        if not cv2.imwrite(str(tmp_path), clean_image):
            raise OSError("Could not prepare the image for prediction")
        return predict_word(tmp_path, model, mapping)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


st.title("✍️ Bangla Handwritten Word Recognition")
st.write("Draw one Bangla word, or upload a handwritten word image. The app predicts character regions and joins them into a word.")

source = st.radio("Input method", ["Draw a word", "Upload an image"], horizontal=True)
gray = None

if source == "Draw a word":
    if CANVAS_AVAILABLE:
        stroke_width = st.slider("Pen thickness", min_value=4, max_value=18, value=9)
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=650,
            drawing_mode="freedraw",
            key="bangla_canvas",
        )
        if canvas.image_data is not None:
            image = np.asarray(canvas.image_data)
            if image.ndim == 3 and image.shape[2] == 4:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            elif image.ndim == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        st.warning("The drawing component could not load. Please use Upload an image instead.")
        st.caption(f"Component error: {CANVAS_ERROR}")
else:
    uploaded = st.file_uploader("Choose a PNG or JPG containing a handwritten word", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            st.error("Could not read the uploaded image.")
        else:
            st.image(gray, caption="Uploaded handwriting", width=650)

if st.button("Recognize word", type="primary"):
    if gray is None:
        st.warning("Please draw a word or upload an image first.")
    else:
        try:
            with st.spinner("Recognizing handwriting..."):
                model = get_model()
                mapping = get_mapping()
                result, predictions = predict_image(gray, model, mapping)
            if not predictions:
                st.warning("No character regions were detected. Try a clearer word.")
            else:
                st.subheader("Predicted Bangla word")
                st.success(result)
                st.caption(f"Detected regions: {len(predictions)}")
                st.subheader("Character predictions")
                st.dataframe(
                    [
                        {
                            "Region": i,
                            "Character": item["character"],
                            "Class ID": item["class_id"],
                            "Confidence": f'{item["confidence"]:.1%}',
                        }
                        for i, item in enumerate(predictions, start=1)
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.info(
    "Prototype limitation: vowel signs, connected handwriting, "
    "and some conjuncts may be recognized incorrectly. "
    "The displayed result is the raw CNN prediction."
)