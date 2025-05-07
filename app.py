import streamlit as st
from transformers import pipeline
from faster_whisper import WhisperModel
import tempfile

@st.cache_resource
def load_model():
    return WhisperModel("medium", compute_type="int8")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

model = load_model()
summarizer = load_summarizer()

st.title("🎙️ Speech to Text + Summary")

audio_file = st.file_uploader("Upload an audio file (.mp3/.wav)", type=["mp3", "wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with st.spinner("Transcribing..."):
        segments, _ = model.transcribe(tmp_path, language="hi")
        transcript = " ".join([seg.text for seg in segments])
        st.success("Transcription Complete!")
        st.write("**Transcript (English):**", transcript)

    if len(transcript.split()) > 10:
        with st.spinner("Summarizing..."):
            summary = summarizer(transcript, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
            st.success("Summary:")
            st.write(summary)
    else:
        st.warning("Transcript too short to summarize.")
