import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from config import PRESET_FILES
from src.chain import DEFAULT_MODEL, ask
from src.ingest import process_file

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Ecommerce Chat Support", layout="wide")


# loaded once and shared by all users
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_client(api_key: str):
    return Groq(api_key=api_key)


# cached so the PDF isn't parsed again on every rerun
@st.cache_data
def read_document_text(path: str) -> str:
    if path.endswith(".pdf"):
        return "\n\n".join(page.page_content for page in PyPDFLoader(path).load())
    return Path(path).read_text(encoding="utf-8")


# each upload gets its own temp file so users don't overwrite each other
def save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_knowledge_base(name: str, path: str):
    with st.spinner(f"Indexing {name}..."):
        st.session_state.vectorstore = process_file(path, get_embeddings())
    st.session_state.selected_doc = name
    st.session_state.doc_path = path
    st.session_state.chat_history = []


api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

st.title("Ecommerce Chat Support")

if not api_key:
    st.error("GROQ_API_KEY is not set. Add it to .env locally or to Secrets on Streamlit Cloud.")
    st.stop()

# per-user state
defaults = {
    "chat_history": [],
    "vectorstore": None,
    "selected_doc": None,
    "doc_path": None,
    "last_preset": "None",
    "last_upload": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

left, center, right = st.columns([1, 2, 2])

with left:
    selected_preset = st.selectbox(
        "Choose a preset knowledge base:",
        options=["None"] + list(PRESET_FILES.keys()),
    )
    uploaded_file = st.file_uploader("Or upload your own file (PDF/TXT):", type=["pdf", "txt"])

# only reload when the preset actually changes
if selected_preset != st.session_state.last_preset:
    st.session_state.last_preset = selected_preset
    if selected_preset != "None":
        load_knowledge_base(selected_preset, PRESET_FILES[selected_preset])

# same for uploads
if uploaded_file is not None:
    upload_key = (uploaded_file.name, uploaded_file.size)
    if upload_key != st.session_state.last_upload:
        st.session_state.last_upload = upload_key
        load_knowledge_base(uploaded_file.name, save_upload_to_temp(uploaded_file))
        with left:
            st.success(f"File {uploaded_file.name} uploaded and processed!")

with center:
    if st.session_state.vectorstore is None:
        st.info("Choose a knowledge base on the left to start chatting.")
    else:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        query = st.chat_input("Ask a question...")
        if query:
            with st.spinner("Thinking..."):
                try:
                    _, st.session_state.chat_history = ask(
                        query,
                        st.session_state.vectorstore,
                        get_client(api_key),
                        model,
                        st.session_state.chat_history,
                    )
                except Exception as e:
                    st.error(f"Request to the LLM failed: {e}")
                    st.stop()
            st.rerun()

with right:
    if st.session_state.doc_path:
        st.markdown(f"### 📄 {st.session_state.selected_doc}")
        st.text(read_document_text(st.session_state.doc_path))
    else:
        st.info("Choose a knowledge base to see its content.")