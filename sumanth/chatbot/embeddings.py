from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Chunk by words
word_count = lambda text: len(text.split())
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=word_count)
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