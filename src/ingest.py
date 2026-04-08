from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader


def load_file(path):
    if path.endswith('.pdf'):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)

    return loader.load()


def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def create_vectorbase(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory='data/chroma_db'
    )
    return vector_store


def process_file(path):
    loader = load_file(path)
    splitter = split_document(loader)
    vector_base = create_vectorbase(splitter)
    return vector_base