"""
Configuration module for the chatbot application.
All settings are loaded from config.yaml file.
"""
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

# Base paths - config.yaml is in project root
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go up to project root
CONFIG_FILE = BASE_DIR / "config.yaml"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Global config dictionary
_config: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml file.
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config.yaml doesn't exist
        yaml.YAMLError: If config.yaml is invalid
    """
    global _config
    
    if _config is not None:
        return _config
    
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}\n"
            "Please create config.yaml with your settings."
        )
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
        
        if _config is None:
            raise ValueError("Configuration file is empty")
        
        return _config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config.yaml: {e}")


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation (e.g., 'openai.api_key').
    
    Args:
        key_path: Dot-separated path to the config value
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    """
    config = load_config()
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


# Load configuration on import
try:
    _loaded_config = load_config()
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    _loaded_config = {}

# OpenAI Configuration
OPENAI_API_KEY: Optional[str] = get_config_value('openai.api_key', '')
OPENAI_MODEL: str = get_config_value('openai.model', 'gpt-4o-mini')
OPENAI_TEMPERATURE: float = float(get_config_value('openai.temperature', 0.7))
OPENAI_MAX_TOKENS: int = int(get_config_value('openai.max_tokens', 1000))

# Embedding Configuration
EMBEDDING_MODEL: str = get_config_value('embedding.model', 'BAAI/bge-small-en-v1.5')
FAISS_INDEX_DIR: str = get_config_value('embedding.faiss_index_dir', 'faiss_index_bge_small')

# RAG Configuration
RAG_K_DOCUMENTS: int = int(get_config_value('rag.k_documents', 3))
RAG_K_FINAL: int = int(get_config_value('rag.k_final', 3))
RAG_CHUNK_SIZE: int = int(get_config_value('rag.chunk_size', 500))
RAG_CHUNK_OVERLAP: int = int(get_config_value('rag.chunk_overlap', 50))
RAG_MIN_CHUNK_SIZE: int = int(get_config_value('rag.min_chunk_size', 50))
RAG_USE_SEMANTIC_CHUNKING: bool = bool(get_config_value('rag.use_semantic_chunking', True))
RAG_USE_RERANKING: bool = bool(get_config_value('rag.use_reranking', True))
RAG_USE_CONTEXT_COMPRESSION: bool = bool(get_config_value('rag.use_context_compression', True))
RAG_MAX_CONTEXT_TOKENS: int = int(get_config_value('rag.max_context_tokens', 2000))
RAG_MAX_CHUNK_TOKENS: int = int(get_config_value('rag.max_chunk_tokens', 200))

# Memory Configuration
MEMORY_MAX_RECENT_MESSAGES: int = int(get_config_value('memory.max_recent_messages', 10))
MEMORY_SUMMARIZATION_THRESHOLD: int = int(get_config_value('memory.summarization_threshold', 20))
MEMORY_STORAGE_DIR: str = get_config_value('memory.storage_dir', 'conversations')

# Routing Configuration
ROUTING_FAQ_FILE: str = get_config_value('routing.faq_file', 'faq_database.json')
ROUTING_SIMILARITY_THRESHOLD: float = float(get_config_value('routing.similarity_threshold', 0.6))
ROUTING_ENABLED: bool = bool(get_config_value('routing.enable_routing', True))

# Cache Configuration
CACHE_DIR: str = get_config_value('cache.cache_dir', 'cache')
CACHE_DEFAULT_TTL_HOURS: int = int(get_config_value('cache.default_ttl_hours', 24))
CACHE_SIMILARITY_THRESHOLD: float = float(get_config_value('cache.similarity_threshold', 0.85))
CACHE_ENABLED: bool = bool(get_config_value('cache.enable_caching', True))

# Token Control Configuration
TOKEN_MAX_INPUT: int = int(get_config_value('token_control.max_input_tokens', 500))
TOKEN_MAX_OUTPUT: int = int(get_config_value('token_control.max_output_tokens', 300))
TOKEN_ENABLE_STREAMING: bool = bool(get_config_value('token_control.enable_streaming', False))
TOKEN_STOP_SEQUENCES: List[str] = get_config_value('token_control.stop_sequences', ["\n\n\n", "---", "###", "END"])

# Analytics Configuration
ANALYTICS_DIR: str = get_config_value('analytics.analytics_dir', 'analytics')
ANALYTICS_ENABLED: bool = bool(get_config_value('analytics.enable_tracking', True))
ANALYTICS_SAVE_INTERVAL: int = int(get_config_value('analytics.save_stats_interval', 100))

# Document Processor Configuration
DOCUMENT_INPUT_PATH: str = get_config_value('document_processor.input_path', 'documents')
DOCUMENT_RECURSIVE: bool = bool(get_config_value('document_processor.recursive', False))

# Scraper Configuration (optional)
SCRAPER_URL: str = get_config_value('scraper.url', '')
SCRAPER_CONFIG_FILE: str = get_config_value('scraper.config_file', 'scraper_config.yaml')
SCRAPER_TIMEOUT: int = int(get_config_value('scraper.timeout', 10))
SCRAPER_MAX_RETRIES: int = int(get_config_value('scraper.max_retries', 3))
SCRAPER_DELAY: float = float(get_config_value('scraper.delay', 1.0))
SCRAPER_CONTENT_FILE: str = get_config_value('scraper.content_file', 'content.txt')
SCRAPER_MAX_PAGES: int = int(get_config_value('scraper.max_pages', 10))

# Application Configuration
APP_TITLE: str = get_config_value('app.title', 'DataLegos Info Genie')
APP_DESCRIPTION: str = get_config_value(
    'app.description',
    'Feel free to ask any queries related to our company :)'
)
APP_HOST: str = get_config_value('app.host', '127.0.0.1')
APP_PORT: int = int(get_config_value('app.port', 7860))
APP_SHARE: bool = bool(get_config_value('app.share', False))

# Logging Configuration
LOG_LEVEL: str = get_config_value('logging.level', 'INFO')
LOG_FILE: str = get_config_value('logging.file', 'chatbot.log')

# Fallback messages
FALLBACK_NO_ANSWER: str = get_config_value(
    'fallbacks.no_answer',
    'I apologize, but I don\'t have enough information to answer your question. '
    'Please contact us directly, and we\'ll be happy to help you with your query.'
)
FALLBACK_ERROR: str = get_config_value(
    'fallbacks.error',
    'I\'m sorry, I encountered an error while processing your request. '
    'Please try again or contact us for assistance.'
)

# System prompt template
SYSTEM_PROMPT_TEMPLATE: str = get_config_value(
    'system_prompt',
    """You are a helpful receptionist at our company.
Your role is to provide accurate information about the company based on the context provided.

Guidelines:
- Answer questions accurately based only on the provided context
- If you don't know the answer, politely say that you have limited knowledge and suggest contacting the company directly
- Be professional, friendly, and helpful
- Do not make up or hallucinate information
- If the question is not related to the company, politely redirect the conversation

Context:
{context}

Question: {query}

Answer:"""
)


def validate_config() -> dict:
    """
    Validate configuration and return any issues found.
    Returns a dictionary with 'errors' and 'warnings' lists.
    """
    issues = {"errors": [], "warnings": []}
    
    try:
        config = load_config()
    except Exception as e:
        issues["errors"].append(f"Failed to load config.yaml: {e}")
        return issues
    
    # Validate OpenAI API key
    if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
        issues["errors"].append(
            "openai.api_key is not set in config.yaml. "
            "Please add your OpenAI API key to the configuration file."
        )
    
    # Validate FAISS index directory
    if not Path(FAISS_INDEX_DIR).exists():
        issues["warnings"].append(
            f"FAISS index directory '{FAISS_INDEX_DIR}' does not exist. "
            "You may need to run embeddings.py first."
        )
    
    # Validate RAG settings
    if RAG_K_DOCUMENTS < 1:
        issues["errors"].append("rag.k_documents must be at least 1")
    
    if RAG_CHUNK_SIZE < 50:
        issues["warnings"].append(
            "rag.chunk_size is very small (< 50). Consider increasing it for better context."
        )
    
    if RAG_CHUNK_OVERLAP < 0:
        issues["errors"].append("rag.chunk_overlap cannot be negative")
    
    if RAG_CHUNK_OVERLAP >= RAG_CHUNK_SIZE:
        issues["errors"].append("rag.chunk_overlap must be less than rag.chunk_size")
    
    # Validate scraper settings
    if SCRAPER_TIMEOUT < 1:
        issues["errors"].append("scraper.timeout must be at least 1 second")
    
    if SCRAPER_MAX_RETRIES < 0:
        issues["errors"].append("scraper.max_retries cannot be negative")
    
    if SCRAPER_DELAY < 0:
        issues["errors"].append("scraper.delay cannot be negative")
    
    # Validate app settings
    if APP_PORT < 1 or APP_PORT > 65535:
        issues["errors"].append("app.port must be between 1 and 65535")
    
    # Validate logging level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if LOG_LEVEL.upper() not in valid_log_levels:
        issues["warnings"].append(
            f"logging.level '{LOG_LEVEL}' is not a standard level. "
            f"Valid levels: {', '.join(valid_log_levels)}"
        )
    
    return issues


def reload_config():
    """
    Reload configuration from config.yaml file.
    Useful for testing or when config file is updated.
    """
    global _config
    _config = None
    load_config()
