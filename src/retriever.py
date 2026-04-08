from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory='data/chroma_db',
        embedding_function=embeddings,
    )
    return vectorstore


def retrieve_vectorstore(query: str, k: int = 3):
    vectorstore = load_vector_store()
    result = vectorstore.similarity_search(query, k=k)
    return result
