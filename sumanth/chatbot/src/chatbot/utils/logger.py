"""
Logging utility for the chatbot application.
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str = "chatbot", log_file: str = None, level: str = None) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Logger name
        log_file: Optional log file path (defaults to config LOG_FILE)
        level: Optional log level (defaults to config LOG_LEVEL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Get config values with lazy import to avoid circular dependencies
    if level is None or log_file is None:
        try:
            from chatbot.core.config import LOG_LEVEL, LOG_FILE
            level = level or LOG_LEVEL
            log_file = log_file or LOG_FILE
        except ImportError:
            # Fallback if config not available
            level = level or "INFO"
            log_file = log_file or "chatbot.log"
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed format)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger instance (lazy initialization)
_logger_instance = None

def get_logger():
    """Get or create the default logger instance."""
    global _logger_instance
    if _logger_instance is None:
        try:
            from chatbot.core.config import LOG_FILE
            _logger_instance = setup_logger(log_file=LOG_FILE)
        except ImportError:
            _logger_instance = setup_logger(log_file="chatbot.log")
    return _logger_instance

# Create logger instance
logger = get_logger()

