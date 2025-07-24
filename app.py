import os
import streamlit as st
import PyPDF2
import requests
from dotenv import load_dotenv


# Load environment variables

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b:free"

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def chunk_text(text, max_tokens=1500):
    paragraphs = text.split("\n\n")
    chunks = []
    chunk = ""
    for para in paragraphs:
        if len(chunk + para) < max_tokens:
            chunk += para + "\n\n"
        else:
            chunks.append(chunk.strip())
            chunk = para + "\n\n"
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def ask_deepseek(question, context):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the context from a PDF."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


# Streamlit UI

st.set_page_config(page_title="📄 PDF Chatbot with DeepSeek", layout="wide")
st.title("📄🤖 PDF Chatbot using DeepSeek (via OpenRouter)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# main logic

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(full_text)

    st.success("PDF processed successfully! Ask your question below.")

    question = st.text_input("Ask a question based on the uploaded PDF:")

    if question:
        context = "\n\n".join(chunks[:3])  # Use first few chunks
        with st.spinner("Fetching answer from DeepSeek..."):
            answer = ask_deepseek(question, context)
        st.markdown("### 🤖 Answer:")
        st.write(answer)
