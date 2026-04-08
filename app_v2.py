import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

from src.chain import ask
from src.ingest import process_file
from config import PRESET_FILES


st.set_page_config(
    page_title="Ecommerce Chat Support",
    layout="wide"
)

st.title("Ecommerce Chat Support")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

left, center, right = st.columns([1, 2, 2])

with left:
    selected_preset = st.selectbox("Choose a preset knowledge base:",
                                   options=["None"]+list(PRESET_FILES.keys()))
    uploaded_file = st.file_uploader("Or upload your own file (PDF/TXT):",
                                     type=["pdf", "txt"])

if selected_preset != "None" and selected_preset != st.session_state.selected_doc:
    path = PRESET_FILES[selected_preset]
    st.session_state.vectorstore = process_file(path)
    st.session_state.selected_doc = selected_preset
    st.session_state.chat_history = []

if uploaded_file is not None and uploaded_file.name != st.session_state.selected_doc:
    save_path = f"data/uploads/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.selected_doc = uploaded_file.name
    st.session_state.vectorstore = process_file(save_path)
    st.session_state.chat_history = []
    st.success(f"File {uploaded_file.name} uploaded and processed!")


with center:
    if st.session_state.vectorstore is not None:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        query = st.chat_input("Ask a question...")

        if query:
            with st.spinner("Thinking..."):
                answer, updated_history = ask(query, st.session_state.chat_history)
            st.session_state.chat_history = updated_history
            st.rerun()
    else:
        st.info("Please select a knowledge base from the sidebar to start chatting.")


with right:
    if st.session_state.selected_doc in PRESET_FILES:
        path = PRESET_FILES[st.session_state.selected_doc]
    elif st.session_state.selected_doc is not None:
        path = f"data/uploads/{st.session_state.selected_doc}"
    else:
        path = None

    if path:
        st.markdown("### 📄 Knowledge Base Content")
        if path.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(path)
            pages = loader.load()
            for page in pages:
                st.text(page.page_content)
        else:
            with open(path, "r", encoding="utf-8") as f:
                st.text(f.read())
    else:
        st.info("Select a preset to see content.")