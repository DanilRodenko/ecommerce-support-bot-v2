import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader 
from langchain_chroma import Chroma


def load_file(path):
    if path.endswith('.pdf'):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def create_vectorbase(chunks, embeddings): 
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"kb_{uuid.uuid4().hex}",
    )
    return vector_store


def process_file(path, embeddings): 
    documents = load_file(path) 
    chunks = split_document(documents) 
    return create_vectorbase(chunks, embeddings)