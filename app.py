import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.chain import ask
from src.ingest import process_file


st.set_page_config(
    page_title="Ecommerce Chat Support",
    layout="wide"
)

st.title("Ecommerce Chat Support")
st.sidebar.title("Knowledge Base Settings")


PRESET_FILES = {
    "Electronics FAQ": "data/uploads/electronics_faq.txt",
    "Clothing FAQ": "data/uploads/clothing_faq.txt",
    "Electronics Returns": "data/uploads/electronics_returns.pdf",
    "Clothing Policies": "data/uploads/clothing_policies.pdf"
}

selected_preset = st.sidebar.selectbox(
    "Choose a preset knowledge base:",
    options=["None"] + list(PRESET_FILES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Or upload your own file (PDF/TXT):",
    type=["pdf", "txt"]
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None


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
