from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Load the vector store
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up Groq as the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context: {context}

Question: {question}
""")

# Build RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
print("🤖 Your PDF chatbot is ready!")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    response = chain.invoke(question)
    print(f"\n🤖 AI: {response}\n")