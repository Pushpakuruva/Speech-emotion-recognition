import streamlit as st
import torch
import numpy as np
import os
import wave
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.io import wavfile

# Load models
@st.cache_resource
def load_models():
    whisper_model = "openai/whisper-base"
    emotion_model = "j-hartmann/emotion-english-distilroberta-base"
    
    processor = WhisperProcessor.from_pretrained(whisper_model)
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    
    tokenizer = AutoTokenizer.from_pretrained(emotion_model)
    emotion_classifier = AutoModelForSequenceClassification.from_pretrained(emotion_model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper.to(device)
    emotion_classifier.to(device)
    
    return processor, whisper, tokenizer, emotion_classifier, device

processor, whisper_model, tokenizer, emotion_model, device = load_models()

# Emotion labels
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Transcribe audio
def transcribe_audio(audio_path):
    sample_rate, audio = wavfile.read(audio_path)
    
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    if sample_rate != 16000:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)),
            np.arange(0, len(audio)),
            audio
        )
    
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)
        
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Analyze emotion
def analyze_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    probs = probabilities.cpu().numpy()[0]
    
    emotions = {emotion_labels[i]: float(probs[i]) for i in range(len(emotion_labels))}
    dominant_emotion = max(emotions, key=emotions.get)
    
    return dominant_emotion, emotions

# Streamlit UI
st.title("🎤 Speech Emotion Recognition")
st.markdown("Upload an audio file to transcribe speech and analyze emotions.")

uploaded_file = st.file_uploader("Upload your audio file (.wav)", type=["wav"])

if uploaded_file:
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(temp_audio_path)

    st.subheader("Transcription")
    st.success(transcription)

    with st.spinner("Analyzing emotions..."):
        dominant_emotion, emotion_probs = analyze_emotion(transcription)

    st.subheader("Detected Emotion")
    st.info(f"🎭 **Dominant Emotion:** {dominant_emotion}")

    st.subheader("Emotion Probabilities")
    df_emotions = { "Emotion": list(emotion_probs.keys()), "Probability": list(emotion_probs.values()) }
    
    # Plot using Plotly
    fig = px.bar(df_emotions, x="Emotion", y="Probability", color="Emotion", title="Emotion Distribution")
    st.plotly_chart(fig)

    # Clean up temp file
    os.remove(temp_audio_path)
