import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1 - Find all PDFs in the documents folder
pdf_files = []
for file in os.listdir("documents"):
    if file.endswith(".pdf"):
        pdf_files.append(file)

print(f"📂 Found {len(pdf_files)} PDF files!")

# Step 2 - Load every PDF one by one
all_pages = []
for pdf in pdf_files:
    path = os.path.join("documents", pdf)
    loader = PyPDFLoader(path)
    pages = loader.load()
    all_pages.extend(pages)
    print(f"✅ Loaded: {pdf} — {len(pages)} pages")

print(f"\n📄 Total pages from all PDFs: {len(all_pages)}")

# Step 3 - Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_pages)
print(f"🔪 Total chunks created: {len(chunks)}")

# Step 4 - Store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
vectorstore.add_documents(chunks)
print(f"\n🎉 All PDFs loaded into ChromaDB successfully!")