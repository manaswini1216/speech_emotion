import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model("emotion_model.h5")

# Define emotion labels (adjust to your model)
emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'fear', 'surprise']

# Title
st.title("🎙️ Speech Emotion Recognition App")

# Upload audio file
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Preprocess function (adjust according to how you trained)
    def preprocess_audio(file):
        y, sr = librosa.load(file, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled

    # Extract features
    features = preprocess_audio(audio_file)

    # Predict
    prediction = model.predict(np.expand_dims(features, axis=0))
    predicted_label = emotion_labels[np.argmax(prediction)]

    # Show result
    st.markdown("### 🧠 Prediction:")
    st.success(predicted_label)
