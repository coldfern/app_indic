from fastapi import FastAPI, UploadFile, File
import whisper
import torch
from transformers import pipeline
from googletrans import Translator
import tempfile

app = FastAPI()
model = whisper.load_model("base")
translator = Translator()
summarizer = pipeline("summarization")

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    original_text = result["text"]

    # Translate Hindi/Hinglish to English
    translated = translator.translate(original_text, dest="en").text

    # Summarize
    summary = summarizer(translated, max_length=60, min_length=20, do_sample=False)[0]['summary_text']

    return {
        "transcript": translated,
        "summary": summary
    }
