"""
Custom exceptions for the chatbot application.
"""

class ChatbotError(Exception):
    """Base exception for chatbot-related errors."""
    pass

class ConfigurationError(ChatbotError):
    """Raised when there's a configuration issue."""
    pass

class EmbeddingError(ChatbotError):
    """Raised when there's an error with embeddings."""
    pass

class ScraperError(ChatbotError):
    """Raised when there's an error with web scraping."""
    pass

class VectorStoreError(ChatbotError):
    """Raised when there's an error with the vector store."""
    pass

class APIError(ChatbotError):
    """Raised when there's an error with external API calls."""
    pass

