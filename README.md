# E-Commerce Support Chatbot 🛍️

An AI-powered customer support chatbot for e-commerce businesses. Upload your FAQ or product documentation and let the bot handle customer questions automatically.

## Live Demo
[ecommerce-support-bot-v2.streamlit.app](https://ecommerce-support-bot-v2.streamlit.app)

## Features
- Upload your own documents (PDF or TXT)
- Preset knowledge bases for quick demo
- Conversation history within session
- Three-column UI: settings, chat, document preview
- Semantic search powered by ChromaDB and sentence-transformers

## How It Works
1. Select a preset knowledge base or upload your own document
2. The document is split into chunks and indexed in ChromaDB
3. Your question is converted to a vector and matched against the knowledge base
4. Relevant chunks are passed to the LLM which generates a response

## Tech Stack
- LangChain — RAG pipeline orchestration
- ChromaDB — vector store
- sentence-transformers (all-MiniLM-L6-v2) — embeddings
- Groq API (Llama 3.3) — response generation
- Streamlit — user interface

## Setup

1. Clone the repo
```bash
git clone git@github.com:DanilRodenko/ecommerce-support-bot-v2.git
cd ecommerce-support-bot-v2
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create `.env` file
```
GROQ_API_KEY=your_key_here
```

4. Run the app
```bash
python -m streamlit run app_v2.py
```

## Project Structure
```
ecommerce-support-bot-v2/
├── data/
│   └── uploads/          # FAQ and product documents
├── src/
│   ├── ingest.py         # Document loading and indexing
│   ├── retriever.py      # Semantic search
│   └── chain.py          # LLM pipeline with chat history
├── config.py             # Preset files configuration
├── app_v2.py             # Streamlit UI
└── requirements.txt
```

## Author
Danil Rodenko — [github.com/DanilRodenko](https://github.com/DanilRodenko) | [linkedin.com/in/danilrodenko](https://linkedin.com/in/danilrodenko)
