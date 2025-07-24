from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Better chunking strategy - larger chunks with semantic boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Larger chunks for better context
    chunk_overlap=50,  # Reasonable overlap
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # Respect semantic boundaries
    length_function=len
)
with open('content.txt', 'r', encoding='utf-8') as f:
    content = f.read()
chunks = splitter.split_text(content)

# Prepare LangChain Document objects
docs = [Document(page_content=chunk) for chunk in chunks]

# Use HuggingFaceEmbeddings for BGE-small
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Store in FAISS
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save the FAISS index for later use
vectorstore.save_local("faiss_index_bge_small")

# print(f"Stored {len(docs)} chunks in FAISS index.")