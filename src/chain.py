import os
from groq import Groq
from dotenv import load_dotenv
from src.retriever import retrieve_vectorstore

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask(query: str, chat_history: list = None) -> tuple[str, list]:
    if chat_history is None:
        chat_history = []

    chunks = retrieve_vectorstore(query)
    context = "\n\n".join([doc.page_content for doc in chunks])

    prompt = f"""
        You are a helpful e-commerce customer support assistant.
        Use the following information to answer the customer's questions.
        If you don't know the answer, say so politely.        

        Context: {context}

        Question: {query}

        Answers:"""

    api_messages = chat_history + [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages,
    )

    answer = response.choices[0].message.content

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history