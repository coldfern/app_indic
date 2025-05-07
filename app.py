import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")

summarizer = load_summarizer()

st.title("🧠 Summarizer Test")
text = st.text_area("Enter long text to summarize:")

if text:
    with st.spinner("Summarizing..."):
        summary = summarizer(text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
        st.success("Done!")
        st.write(summary)
