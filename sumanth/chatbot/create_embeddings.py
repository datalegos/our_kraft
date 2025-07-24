from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

def load_clean_content(filename='scraped_content.txt'):
    """Load the cleaned content from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✓ Loaded content from {filename}")
        return content
    except FileNotFoundError:
        print(f"✗ File {filename} not found. Please run improved_scraper.py first.")
        return None

def create_chunks(content, chunk_size=800, chunk_overlap=100):
    """Create optimized chunks from content"""
    # Better chunking strategy for Q&A
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n=== ",  # Page separators
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence breaks
            "! ",      # Exclamation breaks
            "? ",      # Question breaks
            "; ",      # Semicolon breaks
            ", ",      # Comma breaks
            " ",       # Word breaks
            ""         # Character breaks
        ],
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = splitter.split_text(content)
    
    # Filter out very short or repetitive chunks
    filtered_chunks = []
    seen_chunks = set()
    
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 50 and chunk not in seen_chunks:
            # Remove chunks that are mostly navigation/footer content
            if not any(noise in chunk.lower() for noise in [
                'toggle navigation', 'copyright', 'all rights reserved',
                'follow us', 'quick links', 'privacy policy'
            ]):
                filtered_chunks.append(chunk)
                seen_chunks.add(chunk)
    
    print(f"✓ Created {len(filtered_chunks)} unique chunks from {len(chunks)} total chunks")
    return filtered_chunks

def save_chunks(chunks, filename='processed_chunks.txt'):
    """Save chunks to a text file for inspection"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"=== CHUNK {i} ===\n")
            f.write(f"{chunk}\n\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"✓ Saved {len(chunks)} chunks to {filename}")

def create_embeddings(chunks, model_name="BAAI/bge-small-en-v1.5"):
    """Create embeddings and FAISS index"""
    print(f"✓ Loading embedding model: {model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create Document objects with metadata
    docs = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": "DataLegos_website"}
        )
        docs.append(doc)
    
    print(f"✓ Creating FAISS index from {len(docs)} documents...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    # Save the FAISS index
    index_name = "vector_index"
    vectorstore.save_local(index_name)
    print(f"✓ Saved FAISS index to {index_name}")
    
    return vectorstore, embedding_model

def test_retrieval(vectorstore, test_queries=None):
    """Test the retrieval system"""
    if test_queries is None:
        test_queries = [
            "What services does DataLegos offer?",
            "Who are the team members?",
            "What is the Career Catalyst program?",
            "How can I contact DataLegos?",
            "What industries does DataLegos work with?"
        ]
    
    print("\n" + "="*50)
    print("TESTING RETRIEVAL SYSTEM")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        docs = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(docs, 1):
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"{i}. {content_preview}")
        print()

# Main execution
if __name__ == "__main__":
    # Step 1: Load clean content
    content = load_clean_content('scraped_content.txt')
    if not content:
        print("Please run 'python improved_scraper.py' first to generate clean content.")
        exit(1)
    
    # Step 2: Create chunks
    chunks = create_chunks(content, chunk_size=800, chunk_overlap=100)
    
    # Step 3: Save chunks for inspection
    save_chunks(chunks, 'processed_chunks.txt')
    
    # Step 4: Create embeddings
    vectorstore, embedding_model = create_embeddings(chunks)
    
    # Step 5: Test retrieval
    test_retrieval(vectorstore)
    
    print(f"\n✓ Process completed successfully!")
    print(f"✓ Use 'vector_index' in your app.py")
    print(f"✓ Check 'processed_chunks.txt' to see the processed chunks")