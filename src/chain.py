from src.retriever import retrieve

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_HISTORY_MESSAGES = 10  # last 5 question/answer pairs

SYSTEM_PROMPT = (
    "You are a helpful e-commerce customer support assistant. "
    "Answer ONLY using the provided context. "
    "If the answer is not in the context, say politely that you don't have that information."
)


def ask(query: str, vectorstore, client, model: str = DEFAULT_MODEL, chat_history=None):
    history = list(chat_history) if chat_history else []

    chunks = retrieve(vectorstore, query)
    context = "\n\n".join(doc.page_content for doc in chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    api_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-MAX_HISTORY_MESSAGES:]
        + [{"role": "user", "content": prompt}]
    )

    response = client.chat.completions.create(model=model, messages=api_messages)
    answer = response.choices[0].message.content

    # store the plain question, not the whole prompt with context
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, history