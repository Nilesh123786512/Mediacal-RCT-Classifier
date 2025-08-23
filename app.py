import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import re
from collections import defaultdict

# Corrected decorator for modern TensorFlow versions
@tf.keras.utils.register_keras_serializable()
class UniversalSentenceEncoderLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer to wrap the Universal Sentence Encoder from TensorFlow Hub.
    """
    def __init__(self, **kwargs):
        super(UniversalSentenceEncoderLayer, self).__init__(**kwargs)
        # Load the Hub model once during initialization
        self.hub_layer = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def call(self, inputs):
        # Pass the inputs to the loaded Hub model
        return self.hub_layer(inputs)

    def get_config(self):
        # Implement get_config to make the layer serializable
        return super(UniversalSentenceEncoderLayer, self).get_config()

# Use Streamlit's cache to load the model only once
@st.cache_resource
def load_model():
    """
    Loads the saved Skimlit model from a .keras file.
    """
    try:
        # Load the model with the custom layer
        model = tf.keras.models.load_model("skimlit_tribrid_200k.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the 'skimlit_tribrid.keras' file is in the same directory as this script.")
        return None

# Define the class names and the desired order for display
CLASS_NAMES = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
DISPLAY_ORDER = ["OBJECTIVE", "BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]

def split_chars(text):
    """
    Splits text into a string of space-separated characters.
    """
    return " ".join(list(text))

# --- Streamlit App UI ---

st.set_page_config(page_title="Skimlit RCT Abstract Reader", layout="wide")

st.title("🔬 Skimlit RCT Abstract Reader")
st.info("Paste a medical abstract below. The app will split it into sentences, classify each one's role (e.g., Objective, Methods), and display the results in a structured format.")


# Load the model
model = load_model()

if model:
    # Text area for user input
    input_text = st.text_area(
        "Enter the abstract text here:",
        height=300,
        placeholder="e.g., To investigate the efficacy of a new drug, we conducted a randomized controlled trial. The study involved 200 participants, who were randomly assigned to either the treatment or placebo group. The results showed a significant improvement in the treatment group compared to the placebo group. We conclude that the new drug is an effective treatment for the condition."
    )

    # Button to trigger prediction
    if st.button("Classify Abstract"):
        if not input_text.strip():
            st.warning("Please paste some text into the box above.")
        else:
            with st.spinner("Analyzing the abstract..."):
                # --- START: MODIFIED PREPROCESSING ---

                # 1. Split text into sentences robustly, preserving original text.
                # This regex splits by periods or question marks followed by a space.
                original_sentences = re.split(r'(?<=[.?!])\s+', input_text.strip())
                original_sentences = [s.strip() for s in original_sentences if s.strip()]

                if not original_sentences:
                    st.warning("No sentences found. Please ensure your text contains sentences separated by punctuation.")
                else:
                    # 2. Create a processed version of sentences for the model.
                    # This function replaces any sequence of digits and dots with '@' characters.
                    def replace_numbers(sentence):
                        return re.sub(r'[\d\.]+', lambda m: '@' * len(m.group(0)), sentence)

                    processed_sentences = [replace_numbers(s) for s in original_sentences]
                    
                    # --- END: MODIFIED PREPROCESSING ---

                    total_lines = len(original_sentences)

                    # Create model inputs using the PROCESSED sentences
                    line_numbers_tensor = tf.constant([i for i in range(total_lines)])
                    total_lines_tensor = tf.constant([total_lines - 1] * total_lines)
                    line_numbers_one_hot = tf.one_hot(line_numbers_tensor, depth=15)
                    total_lines_one_hot = tf.one_hot(total_lines_tensor, depth=20)
                    token_inputs = tf.constant(processed_sentences)
                    char_inputs = tf.constant([split_chars(s) for s in processed_sentences])

                    # Make predictions with the correctly shaped, processed inputs
                    try:
                        pred_probs = model.predict(
                            x=(line_numbers_one_hot, total_lines_one_hot, token_inputs, char_inputs)
                        )

                        preds = tf.argmax(pred_probs, axis=1).numpy()
                        pred_confidences = tf.reduce_max(pred_probs, axis=1).numpy()
                        pred_classes = [CLASS_NAMES[p] for p in preds]

                        # Group results, pairing predictions with the ORIGINAL sentences
                        grouped_results = defaultdict(list)
                        for i, sentence in enumerate(original_sentences): # Use original sentences for display
                            grouped_results[pred_classes[i]].append((sentence, pred_confidences[i]))

                        # Display the results
                        st.header("Classification Results")

                        for label in DISPLAY_ORDER:
                            if label in grouped_results:
                                st.markdown(f"**{label}:**")
                                for sentence, confidence in grouped_results[label]:
                                    # Add a period if the original sentence was missing one
                                    if not sentence.endswith(('.', '?', '!')):
                                        sentence += '.'
                                    st.markdown(f"- {sentence.capitalize()} `(Confidence: {confidence:.2%})`")
                                st.markdown("---")

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

else:
    st.error("Model could not be loaded. The application cannot proceed.")