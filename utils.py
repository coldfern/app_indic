import whisper
import torch
from PIL import Image
import pytesseract
from transformers import pipeline

# Load once
asr_model = whisper.load_model("small")
reader = easyocr.Reader(['en'], gpu=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Summarizer
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=0 if device == "cuda" else -1)

def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path, language='hi')
    return result["text"]

def summarize_text(text):
    if len(text.split()) < 30:
        return "Text too short to summarize."
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def ocr_from_image(image):
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return text
