from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load the vector store
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
Answer the question based on the context and conversation history below.

Context: {context}

Conversation History: {history}

Question: {question}
""")

chain = prompt | llm | StrOutputParser()

print("🤖 Your PDF chatbot is ready!")
print("Type 'quit' to exit\n")

history = []

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    history_text = "\n".join(history)

    response = chain.invoke({
        "question": question,
        "context": context,
        "history": history_text
    })

    history.append(f"You: {question}")
    history.append(f"AI: {response}")

    print(f"\n🤖 AI: {response}\n")