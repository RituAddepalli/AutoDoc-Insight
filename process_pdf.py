
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


# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load NLP pipelines (force PyTorch only)
# summarizer = pipeline(
#     "summarization",
#     model="facebook/bart-large-cnn",
#     framework="pt",
#     device=0 if device == "cuda" else -1
# )

# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     framework="pt",
#     device=0 if device == "cuda" else -1
# )

# # Helper: Preprocess images for OCR
# def preprocess_for_ocr(pil_img):
#     img = np.array(pil_img.convert('L'))
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)
#     return Image.fromarray(img)

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     lines = []

#     for page in doc:
#         text = page.get_text("text").strip()
#         if len(text) > 20:
#             lines.extend([line.strip() for line in text.splitlines() if len(line.strip()) > 5])
#         else:
#             pix = page.get_pixmap(dpi=300)
#             img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
#             ocr_text = pytesseract.image_to_string(img, lang="eng")
#             clean_lines = [
#                 line.strip()
#                 for line in ocr_text.splitlines()
#                 if line.strip() and len(line.split()) >= 5 and any(c.isalpha() for c in line)
#             ]
#             lines.extend(clean_lines)

#     final_lines = [line for line in lines if len(line) > 15]
#     return final_lines

# # Filter out code-style lines
# def is_code_line(line):
#     if not line.strip() or len(line.strip()) < 10:
#         return True
#     symbols = ['{', '}', ';', '==', 'def ', 'class ', '=', 'torch.', 'nn.', 'cv2.', 'plt.', '(', ')', '[', ']']
#     if any(sym in line for sym in symbols):
#         return True
#     alpha_ratio = sum(c.isalpha() for c in line) / max(1, len(line))
#     return alpha_ratio < 0.4

# def filter_natural_language(lines):
#     return [line for line in lines if not is_code_line(line)]

# # Summarize text
# def summarize_text(text):
#     if not text.strip():
#         return "No valid natural language text found to summarize."
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     summaries = []
#     for chunk in chunks:
#         result = summarizer(chunk, max_length=250, min_length=60, do_sample=False)
#         summaries.append(result[0]['summary_text'])
#     return "\n\n".join(summaries)

# # Answer a question from text
# def answer_question(text, question):
#     if not text.strip():
#         return "No content found to answer the question."
#     context = text[:4000]  # Keep context short enough
#     result = qa_pipeline({'question': question, 'context': context})
#     return result['answer']

# # ---------- Main Execution ----------
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("❌ Error: Missing PDF path")
#         sys.exit(1)

#     pdf_path = sys.argv[1]
#     if not os.path.isfile(pdf_path):
#         print(f"❌ Error: File not found - {pdf_path}")
#         sys.exit(1)

#     query = sys.argv[2] if len(sys.argv) > 2 else "summarize"

#     # Extract and filter text
#     lines = extract_text_from_pdf(pdf_path)
#     filtered_text = "\n".join(filter_natural_language(lines))

#     # Summarize or answer
#     if "summary" in query.lower() or "summarize" in query.lower():
#         summary = summarize_text(filtered_text)
#         print(summary)
#     else:
#         answer = answer_question(filtered_text, query)
#         print(answer)
































# import sys
# import torch
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
# import numpy as np
# import cv2
# import re

# from transformers import pipeline

# # Set pytesseract path (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# #print(f"Device set to use {device}")

# # Load NLP pipelines
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if device == "cuda" else -1)

# # Helper: Preprocess images for OCR
# def preprocess_for_ocr(pil_img):
#     img = np.array(pil_img.convert('L'))
#     img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)
#     return Image.fromarray(img)

# # Extract text from PDF using text or OCR
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     lines = []

#     for page in doc:
#         text = page.get_text("text").strip()
#         if len(text) > 20:
#             lines.extend([line.strip() for line in text.splitlines() if len(line.strip()) > 5])
#         else:
#             pix = page.get_pixmap(dpi=300)
#             img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
#             ocr_text = pytesseract.image_to_string(img, lang="eng")
#             clean_lines = [
#                 line.strip()
#                 for line in ocr_text.splitlines()
#                 if line.strip() and len(line.split()) >= 5 and any(c.isalpha() for c in line)
#             ]
#             lines.extend(clean_lines)

#     final_lines = [line for line in lines if len(line) > 15]
#     return final_lines

# # Filter natural language
# def is_code_line(line):
#     if not line.strip() or len(line.strip()) < 10:
#         return True
#     symbols = ['{', '}', ';', '==', 'def ', 'class ', '=', 'torch.', 'nn.', 'cv2.', 'plt.', '(', ')', '[', ']']
#     if any(sym in line for sym in symbols):
#         return True
#     alpha_ratio = sum(c.isalpha() for c in line) / max(1, len(line))
#     return alpha_ratio < 0.4

# def filter_natural_language(lines):
#     return [line for line in lines if not is_code_line(line)]

# # Summarization
# def summarize_text(text):
#     if not text.strip():
#         return "No valid natural language text found to summarize."
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     summaries = []
#     for chunk in chunks:
#         result = summarizer(chunk, max_length=250, min_length=60, do_sample=False)
#         summaries.append(result[0]['summary_text'])
#     return "\n\n".join(summaries)

# # Question Answering
# def answer_question(text, question):
#     if not text.strip():
#         return "No content found to answer the question."
#     context = text[:4000]  # Limit to first 4000 characters
#     result = qa_pipeline({'question': question, 'context': context})
#     return result['answer']

# # Main Execution
# # Main Execution
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Error: Missing PDF path")
#         sys.exit(1)

#     pdf_path = sys.argv[1]
#     query = sys.argv[2] if len(sys.argv) > 2 else "summarize"

#     lines = extract_text_from_pdf(pdf_path)
#     filtered_text = "\n".join(filter_natural_language(lines))

#     if query.lower().strip() == "summarize" or "summary" in query.lower():
#         summary = summarize_text(filtered_text)
#         print(summary)
#     else:
#         answer = answer_question(filtered_text, query)
#         print(answer)