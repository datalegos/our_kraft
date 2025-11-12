"""
Response caching system with similarity matching and expiration.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from chatbot.utils.logger import logger
from chatbot.optimizers.cost_optimizer import TokenCounter


class ResponseCache:
    """
    Cache for query-response pairs with similarity matching.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        default_ttl_hours: int = 24,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize response cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl_hours: Default time-to-live in hours
            similarity_threshold: Similarity threshold for matching (0-1)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.similarity_threshold = similarity_threshold
        self.token_counter = TokenCounter()
        self.cache_index: Dict[str, Dict] = {}
        self.load_cache_index()
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _get_cache_file(self, query_hash: str) -> Path:
        """Get cache file path for query hash."""
        return self.cache_dir / f"{query_hash}.json"
    
    def load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "index.json"
        try:
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
                logger.debug(f"Loaded cache index with {len(self.cache_index)} entries")
        except Exception as e:
            logger.error(f"Error loading cache index: {e}", exc_info=True)
            self.cache_index = {}
    
    def save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "index.json"
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}", exc_info=True)
    
    def check_similarity(self, query: str, cached_queries: List[str]) -> Optional[Tuple[str, float]]:
        """
        Check if query is similar to any cached query.
        
        Args:
            query: Current query
            cached_queries: List of cached query strings
        
        Returns:
            Tuple of (matched_query, similarity_score) or None
        """
        query_lower = query.lower().strip()
        best_match = None
        best_score = 0.0
        
        for cached_query in cached_queries:
            cached_lower = cached_query.lower().strip()
            similarity = SequenceMatcher(None, query_lower, cached_lower).ratio()
            
            if similarity > best_score and similarity >= self.similarity_threshold:
                best_score = similarity
                best_match = cached_query
        
        if best_match:
            logger.debug(f"Similar query found: '{best_match}' (similarity: {best_score:.2f})")
            return (best_match, best_score)
        
        return None
    
    def get_cached_response(
        self,
        query: str,
        ttl_hours: int = None
    ) -> Optional[Dict[str, any]]:
        """
        Get cached response for query.
        
        Args:
            query: User query
            ttl_hours: Optional custom TTL in hours
        
        Returns:
            Cached response data or None
        """
        query_hash = self._hash_query(query)
        cache_file = self._get_cache_file(query_hash)
        
        # Check exact match first
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check expiration
                expires_at = datetime.fromisoformat(cached_data.get('expires_at', ''))
                if datetime.now() < expires_at:
                    logger.debug(f"Cache hit (exact match) for query: {query[:50]}...")
                    return cached_data
                else:
                    # Expired, remove
                    cache_file.unlink()
                    if query_hash in self.cache_index:
                        del self.cache_index[query_hash]
            except Exception as e:
                logger.error(f"Error reading cache file: {e}", exc_info=True)
        
        # Check similarity match
        cached_queries = list(self.cache_index.keys())
        if cached_queries:
            similar = self.check_similarity(query, cached_queries)
            if similar:
                matched_query, similarity = similar
                matched_hash = self._hash_query(matched_query)
                matched_file = self._get_cache_file(matched_hash)
                
                if matched_file.exists():
                    try:
                        with open(matched_file, 'r', encoding='utf-8') as f:
                            cached_data = json.load(f)
                        
                        # Check expiration
                        expires_at = datetime.fromisoformat(cached_data.get('expires_at', ''))
                        if datetime.now() < expires_at:
                            logger.debug(f"Cache hit (similarity match) for query: {query[:50]}...")
                            cached_data['similarity_score'] = similarity
                            cached_data['matched_query'] = matched_query
                            return cached_data
                    except Exception as e:
                        logger.error(f"Error reading similar cache file: {e}", exc_info=True)
        
        return None
    
    def cache_response(
        self,
        query: str,
        response: str,
        ttl_hours: int = None,
        metadata: Dict[str, any] = None
    ):
        """
        Cache query-response pair.
        
        Args:
            query: User query
            response: AI response
            ttl_hours: Optional custom TTL in hours
            metadata: Optional metadata
        """
        query_hash = self._hash_query(query)
        cache_file = self._get_cache_file(query_hash)
        
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
        expires_at = datetime.now() + ttl
        
        cache_data = {
            'query': query,
            'response': response,
            'cached_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat(),
            'ttl_hours': ttl_hours or (self.default_ttl.total_seconds() / 3600),
            'metadata': metadata or {},
            'query_tokens': self.token_counter.estimate_tokens(query),
            'response_tokens': self.token_counter.estimate_tokens(response)
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            # Update index
            self.cache_index[query] = {
                'hash': query_hash,
                'cached_at': cache_data['cached_at'],
                'expires_at': cache_data['expires_at']
            }
            self.save_cache_index()
            
            logger.debug(f"Cached response for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error caching response: {e}", exc_info=True)
    
    def clear_expired(self):
        """Remove expired cache entries."""
        expired_count = 0
        now = datetime.now()
        
        for query, index_data in list(self.cache_index.items()):
            expires_at = datetime.fromisoformat(index_data.get('expires_at', ''))
            if now >= expires_at:
                # Remove cache file
                query_hash = index_data.get('hash')
                if query_hash:
                    cache_file = self._get_cache_file(query_hash)
                    if cache_file.exists():
                        cache_file.unlink()
                
                # Remove from index
                del self.cache_index[query]
                expired_count += 1
        
        if expired_count > 0:
            self.save_cache_index()
            logger.info(f"Cleared {expired_count} expired cache entries")
        
        return expired_count
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        self.clear_expired()
        
        total_entries = len(self.cache_index)
        total_size = sum(
            self._get_cache_file(data.get('hash', '')).stat().st_size
            for data in self.cache_index.values()
            if self._get_cache_file(data.get('hash', '')).exists()
        )
        
        return {
            'total_entries': total_entries,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

