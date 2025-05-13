import os
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Extract text from PDF
def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Break text into manageable chunks
def chunk_text(text, chunk_size=3000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Summarize using Gemini Pro
def summarize_text(text):
    model = genai.GenerativeModel("gemini-pro")
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        prompt = f"Summarize this financial report text into 3–5 bullet points:\n\n{chunk}"
        try:
            response = model.generate_content(prompt)
            summaries.append(response.text.strip())
        except Exception as e:
            summaries.append(f"[Error: {e}]")

    return "\n\n".join(summaries)
