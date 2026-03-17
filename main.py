import os
os.environ["TRANSFORMERS_CACHE"] = "/opt/render/project/src/.cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/opt/render/project/src/.cache"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# PART 2 — Setup the kitchen
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
Context: {context}
Question: {question}
""")

chain = prompt | llm | StrOutputParser()

# PART 3 — The order form
class Question(BaseModel):
    question: str

# PART 4 — Open the restaurant doors!
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "🤖 MozaicTeck RAG API is running!"}

@app.post("/ask")
def ask(body: Question):
    docs = retriever.invoke(body.question)
    context = "\n".join([doc.page_content for doc in docs])
    
    response = chain.invoke({
        "question": body.question,
        "context": context
    })
    
    return {"answer": response}