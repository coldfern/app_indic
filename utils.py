import whisper
import torch
from transformers import pipeline, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper ASR
asr_model = whisper.load_model("small")

# Summarization
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=0 if device == "cuda" else -1)

# TrOCR for OCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path, language='hi')
    return result["text"]

def summarize_text(text):
    if len(text.split()) < 30:
        return "Text too short to summarize."
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def ocr_from_image(image):
    img = Image.open(image).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text
