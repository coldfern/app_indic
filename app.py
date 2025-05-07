import streamlit as st
import whisper

st.title("🔊 Whisper Test")
model = whisper.load_model("tiny")  # use tiny model to reduce memory load
st.success("Model loaded successfully!")
