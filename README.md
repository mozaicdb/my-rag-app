# 🤖 RAG PDF Chatbot

A conversational AI chatbot that answers questions 
from your own PDF documents using RAG 
(Retrieval-Augmented Generation).

Built by Moses Iluyemi — MozaicTeck 🇳🇬

---

## 💡 What Problem Does This Solve?

Normal AI like ChatGPT answers from general memory 
and sometimes makes things up (hallucination).

This chatbot reads YOUR specific document and answers 
ONLY from that document. No guessing. No wrong answers.

---

## 🛠️ How It Works

1. PDF is loaded and split into 156 small chunks
2. Each chunk is converted to meaning-numbers (embeddings)
3. All chunks stored permanently in ChromaDB
4. You ask a question
5. System finds the most relevant chunks
6. AI answers ONLY from those chunks

---

## 🚀 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| LangChain | RAG framework |
| ChromaDB | Vector database |
| Groq API | Free LLM (LLaMA 3.3 70B) |
| HuggingFace | Embeddings model |

**Total cost: ₦0 — everything is free**

---

## ⚡ Quick Start
```bash
# Clone the repo
git clone https://github.com/mozaicdb/my-rag-app.git

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install langchain langchain-community
pip install chromadb langchain-groq
pip install sentence-transformers pypdf

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# Run the chatbot
python chat.py
```

---

## 📁 Project Structure
```
my-rag-app/
├── chat.py          # Main chatbot
├── load_pdf.py      # PDF loader
├── store_chunks.py  # Creates embeddings
├── document.pdf     # Source document
└── chroma_db/       # Vector database
```

---

## 🎯 Part of My 90-Day AI Engineering Journey

I am documenting my journey from AI teacher to 
AI Engineer in public — Learning AI with Claude.

Follow along: github.com/mozaicdb

---

*Built with Python · LangChain · ChromaDB · Groq*
```
