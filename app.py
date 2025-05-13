import os
import streamlit as st
import requests
import PyPDF2
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Function to extract text from uploaded PDF
def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to chunk long text
def chunk_text(text, chunk_size=3000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Query Gemini API using SDK
def query_gemini(question: str, context: List[str]) -> str:
    context_text = "\n\n".join(context)
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

# Streamlit App UI
st.set_page_config(page_title="📑 Financial Report Summarizer", layout="wide")
st.title("📊 Financial Report Summarizer (Gemini AI)")
st.write("Upload a 10-K PDF filing and get a concise summary using Google's Gemini LLM.")

uploaded_file = st.file_uploader("Upload 10-K PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Extracting text from PDF..."):
        raw_text = extract_pdf_text(uploaded_file)

    if raw_text:
        st.success("✅ PDF text extraction complete.")
        
        if st.button("🧠 Generate Summary with Gemini"):
            with st.spinner("Generating summary..."):
                chunks = chunk_text(raw_text)
                summary = query_gemini(
                    "Summarize this financial report into 3–5 bullet points for an investor.",
                    chunks
                )
            st.subheader("📝 Summary")
            st.text_area("Gemini Summary Output", summary, height=400)
    else:
        st.error("❌ Could not extract text from this PDF.")

