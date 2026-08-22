import re
import subprocess
import sys
from pathlib import Path

import joblib
import streamlit as st


MODEL_FILE = "kafka_classifier.joblib"
KAFKA_FILE = "ykafka.txt"
NORMAL_FILE = "nkafka.txt"


def train_model_if_needed():
    """
    Train the model automatically if it has not been created yet.
    """
    if not Path(MODEL_FILE).exists():
        st.info("Training classifier for the first time...")

        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            st.error("Model training failed.")
            st.code(result.stderr)
            st.stop()


@st.cache_resource
def load_model():
    train_model_if_needed()
    return joblib.load(MODEL_FILE)


def clean_text(text):
    """
    Basic cleanup for user input.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_prediction(model, text):
    """
    Returns:
        prediction label
        confidence percentage
        Kafkaesque probability
    """
    probabilities = model.predict_proba([text])[0]

    # Class order from the classifier
    classes = model.named_steps["classifier"].classes_

    kafka_index = list(classes).index(1)
    normal_index = list(classes).index(0)

    kafka_probability = probabilities[kafka_index]
    normal_probability = probabilities[normal_index]

    if kafka_probability >= normal_probability:
        label = "KAFKA"
        confidence = kafka_probability
    else:
        label = "NOT KAFKA"
        confidence = normal_probability

    return label, confidence, kafka_probability


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Kafka Classifier",
    page_icon="🪳",
    layout="centered"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 800px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }

    .prediction-label {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
    }

    .confidence-text {
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">Kafkaesque</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Is this sentence Kafkaesque?</div>',
    unsafe_allow_html=True
)

st.write("")

# --------------------------------------------------
# Input
# --------------------------------------------------

user_text = st.text_area(
    "Paste a passage below",
    height=180,
    placeholder=(
        "The door was still closed. Gregor looked at it "
        "for a long time, as though waiting for it to "
        "explain why it had become impossible to open."
    ),
    label_visibility="visible"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = load_model()


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if st.button("Analyze Text", use_container_width=True):

    cleaned_text = clean_text(user_text)

    if not cleaned_text:
        st.warning("Please enter a sentence or passage first.")

    else:
        label, confidence, kafka_probability = make_prediction(
            model,
            cleaned_text
        )

        st.divider()

        st.subheader("Prediction")

        if label == "KAFKA":
            st.markdown(
                '<div class="prediction-label">KAFKA</div>',
                unsafe_allow_html=True
            )

            st.success(
                "This sentence shows kafkaesque characteristics."
            )

        else:
            st.markdown(
                '<div class="prediction-label">NOT KAFKA</div>',
                unsafe_allow_html=True
            )

            st.info(
                "This is a regular sentence."
            )

        st.write("")

        st.subheader("Confidence")

        # Progress bar
        st.progress(int(confidence * 100))

        st.markdown(
            f"### {confidence:.1%}"
        )

        st.caption(
            f"Probability of being Kafkaesque: "
            f"{kafka_probability:.1%}"
        )
