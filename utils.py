from paddleocr import PaddleOCR
from transformers import pipeline
import whisper
import streamlit as st
from PIL import Image
import numpy as np

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

whisper_model = load_whisper()
summarizer = load_summarizer()
ocr_model = load_ocr()

def transcribe_audio(path):
    result = whisper_model.transcribe(path, language='hi')
    return result["text"]

def summarize_text(text):
    if len(text.split()) < 30:
        return "Too short to summarize."
    return summarizer(text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']

def ocr_from_image(image_file):
    img = np.array(Image.open(image_file).convert("RGB"))
    result = ocr_model.ocr(img, cls=True)
    text = " ".join([line[1][0] for line in result[0]])
    return text
