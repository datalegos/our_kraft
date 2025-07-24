import re
import unicodedata
from datetime import datetime

def clean_telugu_text(text: str) -> str:
    """Clean and format Telugu/English mixed text content"""
    if not text:
        return ""
    
    # Normalize unicode characters for proper Telugu display
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace but preserve Telugu characters
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove unwanted control characters but keep Telugu unicode range
    # Telugu unicode range: \u0C00-\u0C7F
    text = re.sub(r'[^\w\s\.,!?;:()\-\'""\u0C00-\u0C7F\u200C\u200D]', '', text)
    
    # Remove zero-width characters that might interfere
    text = text.replace('\u200B', '').replace('\u200C', '').replace('\u200D', '')
    
    return text.strip()

def extract_date_from_text(text: str) -> str:
    """Extract date from Telugu/English text or return current date"""
    try:
        # Look for common date patterns
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}\s+\w+\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
                
    except Exception:
        pass
    
    return datetime.now().isoformat()

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text while preserving Telugu characters"""
    if not text or len(text) <= max_length:
        return text
    
    # Find a good breaking point near max_length
    truncated = text[:max_length]
    
    # Try to break at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If space is reasonably close to end
        truncated = truncated[:last_space]
    
    return truncated + "..."

def is_valid_article_title(title: str) -> bool:
    """Check if title is valid (not empty, not just punctuation)"""
    if not title or len(title.strip()) < 5:
        return False
    
    # Remove punctuation and whitespace
    clean_title = re.sub(r'[^\w\u0C00-\u0C7F]', '', title)
    
    return len(clean_title) >= 3