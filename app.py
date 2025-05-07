import streamlit as st
from utils import transcribe_audio, summarize_text, ocr_from_image
import tempfile

st.title("🩺 Medical Assistant App")

st.sidebar.header("Choose a feature")
feature = st.sidebar.selectbox("", ["Speech to Text", "Image to Text (OCR)"])

if feature == "Speech to Text":
    st.header("🎤 Speech to Text (Hindi/Hinglish)")
    audio_file = st.file_uploader("Upload your audio file (.wav/.mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(tmp_path)
        st.success("Transcription Complete!")
        st.write("**Transcript (English):**", transcript)

        with st.spinner("Summarizing..."):
            summary = summarize_text(transcript)
        st.write("**Summary:**", summary)

elif feature == "Image to Text (OCR)":
    st.header("🖼️ Image to Text from Medical Reports")
    uploaded_image = st.file_uploader("Upload an image of your medical report", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Report", use_column_width=True)
        with st.spinner("Extracting text..."):
            text = ocr_from_image(uploaded_image)
        st.success("Text Extraction Complete!")
        st.write("**Extracted Text:**", text)

        with st.spinner("Summarizing..."):
            summary = summarize_text(text)
        st.write("**Summary of Insights:**", summary)
