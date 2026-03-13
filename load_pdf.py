from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)

print(f"✅ PDF loaded successfully!")
print(f"📄 Total pages: {len(pages)}")
print(f"🔪 Total chunks created: {len(chunks)}")