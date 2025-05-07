import whisper
import torch
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import streamlit as st

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")

@st.cache_resource
def load_ocr():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    return processor, model

def transcribe_audio(path):
    model = load_whisper()
    result = model.transcribe(path, language='hi')
    return result["text"]

def summarize_text(text):
    summarizer = load_summarizer()
    if len(text.split()) < 30:
        return "Too short to summarize."
    return summarizer(text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']

def ocr_from_image(image):
    processor, model = load_ocr()
    img = Image.open(image).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
