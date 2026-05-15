import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ChatPDF - AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedding_model, groq_client

embedding_model, groq_client = load_models()

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def ask_rag(question, chunks, index):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    distances, indices = index.search(question_embedding, 3)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are a helpful assistant.
Answer ONLY from the context below.
If answer not found say "This topic is not covered in the PDF."

Context:
{context}

Question: {question}
Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, relevant_chunks

with st.sidebar:
    st.title("🤖 ChatPDF")
    st.caption("AI-Powered PDF Assistant")
    st.divider()
    uploaded_file = st.file_uploader("📄 Upload Your PDF", type="pdf")
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Built by Utsav Gaywala 🚀")

st.title("💬 Chat with your PDF")
st.caption("Upload a PDF and ask any question about it!")

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        text = extract_text(uploaded_file)
        chunks = split_chunks(text)
        index, embeddings = build_faiss_index(chunks)

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 File", uploaded_file.name[:20])
    col2.metric("🔢 Chunks", len(chunks))
    col3.metric("✅ Status", "Ready!")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question := st.chat_input("Ask anything about your PDF..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask_rag(question, chunks, index)
                st.write(answer)
                with st.expander("📄 View Sources"):
                    for i, src in enumerate(sources):
                        st.caption(f"Source {i+1}: {src[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a PDF from the sidebar to get started!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("📤 Upload any PDF")
        st.caption("College notes, books, reports — anything!")
    with col2:
        st.success("💬 Ask questions")
        st.caption("Natural language questions about your PDF")
    with col3:
        st.success("🤖 Get answers")
        st.caption("AI answers only from your document!")