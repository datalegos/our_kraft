"""
Create embeddings and FAISS index from documents or scraped content.
Supports multiple document formats and semantic chunking.
"""
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from chatbot.core.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_MIN_CHUNK_SIZE,
    RAG_USE_SEMANTIC_CHUNKING,
    DOCUMENT_INPUT_PATH,
    DOCUMENT_RECURSIVE,
    SCRAPER_CONTENT_FILE,
)
from chatbot.processors.document_processor import DocumentProcessor, SemanticChunker
from chatbot.utils.logger import logger
from chatbot.utils.exceptions import EmbeddingError, ConfigurationError


def word_count(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Text to count words in
    
    Returns:
        Number of words
    """
    return len(text.split())


def create_embeddings_from_documents(
    input_path: str = None,
    recursive: bool = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    min_chunk_size: int = None,
    use_semantic_chunking: bool = None,
    embedding_model_name: str = None,
    output_dir: str = None
) -> None:
    """
    Create embeddings and FAISS index from documents.
    
    Args:
        input_path: Path to document file or directory
        recursive: Process files recursively in directories
        chunk_size: Size of text chunks in words
        chunk_overlap: Overlap between chunks in words
        min_chunk_size: Minimum chunk size to keep
        use_semantic_chunking: Use semantic chunking instead of fixed-size
        embedding_model_name: Name of embedding model
        output_dir: Output directory for FAISS index
    """
    input_path = input_path or DOCUMENT_INPUT_PATH
    recursive = recursive if recursive is not None else DOCUMENT_RECURSIVE
    chunk_size = chunk_size or RAG_CHUNK_SIZE
    chunk_overlap = chunk_overlap or RAG_CHUNK_OVERLAP
    min_chunk_size = min_chunk_size or RAG_MIN_CHUNK_SIZE
    use_semantic_chunking = use_semantic_chunking if use_semantic_chunking is not None else RAG_USE_SEMANTIC_CHUNKING
    embedding_model_name = embedding_model_name or EMBEDDING_MODEL
    output_dir = output_dir or FAISS_INDEX_DIR
    
    try:
        # Process documents
        logger.info(f"Processing documents from: {input_path}")
        processor = DocumentProcessor()
        documents = processor.process_documents(input_path, recursive=recursive)
        
        logger.info(f"Processed {len(documents)} document(s)")
        
        # Chunk documents
        if use_semantic_chunking:
            logger.info("Using semantic chunking...")
            chunker = SemanticChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size
            )
            chunked_docs = chunker.chunk_documents(documents)
        else:
            logger.info(f"Using fixed-size chunking (size={chunk_size}, overlap={chunk_overlap})...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=word_count
            )
            
            chunked_docs = []
            for doc in documents:
                chunks = splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata['chunk_index'] = i
                    chunk_metadata['total_chunks'] = len(chunks)
                    chunked_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        if not chunked_docs:
            raise EmbeddingError("No chunks created from documents")
        
        logger.info(f"Created {len(chunked_docs)} chunks")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        vectorstore = FAISS.from_documents(chunked_docs, embedding_model)
        
        # Save index
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving FAISS index to {output_dir}")
        vectorstore.save_local(output_dir)
        
        avg_chunk_size = sum(word_count(doc.page_content) for doc in chunked_docs) // len(chunked_docs)
        logger.info(f"Successfully created FAISS index with {len(chunked_docs)} documents")
        print(f"[OK] Successfully created FAISS index with {len(chunked_docs)} documents")
        print(f"   Index saved to: {output_dir}")
        print(f"   Chunks: {len(chunked_docs)}")
        print(f"   Average chunk size: {avg_chunk_size} words")
        print(f"   Chunking method: {'Semantic' if use_semantic_chunking else 'Fixed-size'}")
        
    except Exception as e:
        logger.error(f"Error creating embeddings from documents: {e}", exc_info=True)
        raise EmbeddingError(f"Failed to create embeddings: {e}")


def create_embeddings_from_file(
    content_file: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    embedding_model_name: str = None,
    output_dir: str = None
) -> None:
    """
    Create embeddings and FAISS index from a text file (legacy method for scraped content).
    
    Args:
        content_file: Path to content file
        chunk_size: Size of text chunks in words
        chunk_overlap: Overlap between chunks in words
        embedding_model_name: Name of embedding model
        output_dir: Output directory for FAISS index
    """
    content_file = content_file or SCRAPER_CONTENT_FILE
    chunk_size = chunk_size or RAG_CHUNK_SIZE
    chunk_overlap = chunk_overlap or RAG_CHUNK_OVERLAP
    embedding_model_name = embedding_model_name or EMBEDDING_MODEL
    output_dir = output_dir or FAISS_INDEX_DIR
    
    # Validate input file
    content_path = Path(content_file)
    if not content_path.exists():
        raise ConfigurationError(
            f"Content file not found: {content_file}. "
            "Please provide a valid file path."
        )
    
    if content_path.stat().st_size == 0:
        raise ConfigurationError(f"Content file is empty: {content_file}")
    
    try:
        # Read content
        logger.info(f"Reading content from {content_file}")
        with open(content_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            raise ConfigurationError(f"Content file contains no text: {content_file}")
        
        logger.info(f"Content loaded: {len(content)} characters, {word_count(content)} words")
        
        # Split into chunks
        logger.info(f"Splitting content into chunks (size={chunk_size}, overlap={chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=word_count
        )
        chunks = splitter.split_text(content)
        
        if not chunks:
            raise EmbeddingError("No chunks created from content")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Prepare Document objects
        docs = [Document(page_content=chunk, metadata={'source': content_file}) for chunk in chunks]
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        vectorstore = FAISS.from_documents(docs, embedding_model)
        
        # Save index
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving FAISS index to {output_dir}")
        vectorstore.save_local(output_dir)
        
        avg_chunk_size = sum(word_count(c) for c in chunks) // len(chunks)
        logger.info(f"Successfully created FAISS index with {len(docs)} documents")
        print(f"[OK] Successfully created FAISS index with {len(docs)} documents")
        print(f"   Index saved to: {output_dir}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Average chunk size: {avg_chunk_size} words")
        
    except FileNotFoundError as e:
        raise ConfigurationError(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}", exc_info=True)
        raise EmbeddingError(f"Failed to create embeddings: {e}")


def main():
    """Main entry point for creating embeddings."""
    import sys
    
    try:
        # Check if documents directory exists, otherwise use file method
        input_path = Path(DOCUMENT_INPUT_PATH)
        
        if input_path.exists() and (input_path.is_dir() or input_path.is_file()):
            # Use document processor
            create_embeddings_from_documents()
        else:
            # Fall back to file method
            logger.info(f"Document path '{DOCUMENT_INPUT_PATH}' not found, trying file method...")
            create_embeddings_from_file()
            
    except (ConfigurationError, EmbeddingError) as e:
        logger.error(f"Embedding error: {e}")
        print(f"\n[ERROR] Error: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Embedding creation interrupted by user")
        print("\n[WARNING] Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
