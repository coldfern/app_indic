import streamlit as st
import tempfile
from utils import transcribe_audio, summarize_text, ocr_from_image

st.set_page_config(page_title="Speech & Medical OCR", layout="centered")

st.title("🩺 Speech & Medical Report Analyzer")

option = st.sidebar.selectbox("Choose Feature", ["🎤 Speech to Text", "🖼️ Image to Text (Medical)"])

if option == "🎤 Speech to Text":
    st.header("🎤 Upload or Record Audio")

    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        st.success("Audio uploaded successfully.")
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(tmp_path)
        st.text_area("Transcript", transcript, height=200)

        with st.spinner("Summarizing..."):
            summary = summarize_text(transcript)
        st.text_area("Summary", summary, height=150)

elif option == "🖼️ Image to Text (Medical)":
    st.header("🖼️ Upload Medical Report Image")

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Extracting text..."):
            full_text = ocr_from_image(image_file)
        st.text_area("Extracted Text", full_text, height=200)

        with st.spinner("Summarizing insights..."):
            summary = summarize_text(full_text)
        st.text_area("Summary of Key Insights", summary, height=150)
