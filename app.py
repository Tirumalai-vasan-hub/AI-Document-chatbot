import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from pypdf import PdfReader

# -------------------------
# 1. Load Embedding Model
# -------------------------
@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# 2. Load Question Answering Model
# -------------------------
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# -------------------------
# Helper: Extract text from PDF
# -------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------------
# Helper: Split text
# -------------------------
def split_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# -------------------------
# RAG Search (FAISS)
# -------------------------
def build_faiss_index(embeddings_model, chunks):
    vectors = embeddings_model.encode(chunks)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))
    return index, vectors

def search_similar_chunks(query, embeddings_model, chunks, index, vectors, top_k=3):
    q_vector = embeddings_model.encode([query])
    distances, indices = index.search(np.array(q_vector), top_k)
    return [chunks[i] for i in indices[0]]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Document Chatbot", layout="wide")

st.title("📄 AI Document Q&A Chatbot")
st.markdown("Upload PDFs, Ask Questions, and Get Instant Answers! 🧠")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# PDF Upload
# -------------------------
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")

    # Extract text from all PDFs
    pdf_text = ""
    for uploaded_file in uploaded_files:
        pdf_text += extract_text_from_pdf(uploaded_file) + "\n"

    # Split text into chunks
    chunks = split_text(pdf_text)

    # Load models
    embed_model = load_embeddings_model()
    qa_model = load_qa_model()

    # Build FAISS index
    index, vectors = build_faiss_index(embed_model, chunks)

    # -------------------------
    # User Input
    # -------------------------
    col1, col2 = st.columns([3,1])
    with col1:
        question = st.text_input("Ask your question:")
    with col2:
        get_answer_btn = st.button("Get Answer")

    # -------------------------
    # Answering Questions
    # -------------------------
    if get_answer_btn:
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            retrieved = search_similar_chunks(question, embed_model, chunks, index, vectors)
            context = " ".join(retrieved)
            result = qa_model(question=question, context=context)

            # Save to chat history
            st.session_state.chat_history.append({"question": question, "answer": result["answer"], "context": context})

    # -------------------------
    # Display Chat History
    # -------------------------
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for chat in st.session_state.chat_history[::-1]:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
            with st.expander("Show Relevant Context"):
                st.write(chat["context"])
            st.write("---")

    # -------------------------
    # PDF Summarizer
    # -------------------------
    if st.button("📚 Summarize PDFs"):
        st.info("Generating summary...")
        summary_chunks = split_text(pdf_text, chunk_size=1000, chunk_overlap=100)
        summary = ""
        for chunk in summary_chunks:
            summary += qa_model(question="Summarize this text:", context=chunk)["answer"] + " "
        st.subheader("📚 PDF Summary")
        st.write(summary)

# -------------------------
# Custom CSS Styling
# -------------------------
st.markdown("""
<style>
.stTextInput>div>input {
    height: 50px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)
