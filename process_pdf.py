
import sys
import os
import torch
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
import cv2
import re

from transformers import pipeline, logging
logging.set_verbosity_error()  # suppress model warnings

# Set pytesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
device = 0 if torch.cuda.is_available() else -1

# Load the summarization and question-answering pipelines
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=device)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Function to summarize
def summarize_text(text):
    if len(text.strip()) == 0:
        return "❌ PDF has no readable text."
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summarized = ""
    for chunk in chunks:
        summary = summarizer(chunk, max_length=250, min_length=30, do_sample=False)[0]['summary_text']
        summarized += summary + " "
    return summarized.strip()

# Function to answer a question from PDF text
def answer_question(context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Main
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("⚠ Missing arguments. Usage: python process_pdf.py <pdf_path> <question>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    if not os.path.exists(pdf_path):
        print(f"⚠ File not found: {pdf_path}")
        sys.exit(1)

    context = extract_text_from_pdf(pdf_path)

    # If question is like 'summary', do summary
    if "summary" in question.lower():
        print(summarize_text(context))
    else:
        print(answer_question(context, question))

