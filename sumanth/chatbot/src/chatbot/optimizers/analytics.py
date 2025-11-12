"""
Analytics and monitoring for token usage, cache performance, and conversation metrics.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from chatbot.utils.logger import logger
from chatbot.optimizers.cost_optimizer import TokenCounter


class AnalyticsTracker:
    """Track and analyze chatbot performance metrics."""
    
    def __init__(self, analytics_dir: str = "analytics"):
        """
        Initialize analytics tracker.
        
        Args:
            analytics_dir: Directory to store analytics data
        """
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        self.token_counter = TokenCounter()
        
        # In-memory stats (reset on restart)
        self.session_stats: Dict[str, Dict] = defaultdict(lambda: {
            'queries': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'faq_hits': 0,
            'ai_calls': 0,
            'total_cost': 0.0,
            'conversation_lengths': []
        })
    
    def track_query(
        self,
        session_id: str,
        query: str,
        response: str,
        tokens_used: int = None,
        prompt_tokens: int = None,
        completion_tokens: int = None,
        cost: float = None,
        cache_hit: bool = False,
        faq_hit: bool = False,
        used_ai: bool = True
    ):
        """
        Track a query and its metrics.
        
        Args:
            session_id: Session identifier
            query: User query
            response: AI response
            tokens_used: Total tokens used
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens
            cost: Estimated cost
            cache_hit: Whether cache was hit
            faq_hit: Whether FAQ was used
            used_ai: Whether AI was called
        """
        stats = self.session_stats[session_id]
        
        stats['queries'] += 1
        
        if tokens_used:
            stats['total_tokens'] += tokens_used
        if prompt_tokens:
            stats['prompt_tokens'] += prompt_tokens
        if completion_tokens:
            stats['completion_tokens'] += completion_tokens
        if cost:
            stats['total_cost'] += cost
        
        if cache_hit:
            stats['cache_hits'] += 1
        else:
            stats['cache_misses'] += 1
        
        if faq_hit:
            stats['faq_hits'] += 1
        
        if used_ai:
            stats['ai_calls'] += 1
        
        # Track conversation length
        query_tokens = self.token_counter.estimate_tokens(query)
        response_tokens = self.token_counter.estimate_tokens(response)
        stats['conversation_lengths'].append(query_tokens + response_tokens)
        
        # Log to file
        self._log_query(session_id, query, response, {
            'tokens_used': tokens_used,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'cost': cost,
            'cache_hit': cache_hit,
            'faq_hit': faq_hit,
            'used_ai': used_ai
        })
    
    def _log_query(
        self,
        session_id: str,
        query: str,
        response: str,
        metrics: Dict[str, Any]
    ):
        """Log query to file."""
        log_file = self.analytics_dir / f"{session_id}_queries.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'query': query,
            'response': response[:200] + '...' if len(response) > 200 else response,
            'metrics': metrics
        }
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error logging query: {e}", exc_info=True)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Statistics dictionary
        """
        stats = self.session_stats.get(session_id, {})
        
        if not stats or stats['queries'] == 0:
            return {
                'queries': 0,
                'total_tokens': 0,
                'average_tokens_per_query': 0,
                'cache_hit_rate': 0.0,
                'faq_hit_rate': 0.0,
                'total_cost': 0.0,
                'average_conversation_length': 0
            }
        
        total_queries = stats['queries']
        cache_requests = stats['cache_hits'] + stats['cache_misses']
        
        return {
            'queries': total_queries,
            'total_tokens': stats['total_tokens'],
            'prompt_tokens': stats['prompt_tokens'],
            'completion_tokens': stats['completion_tokens'],
            'average_tokens_per_query': stats['total_tokens'] / total_queries if total_queries > 0 else 0,
            'cache_hits': stats['cache_hits'],
            'cache_misses': stats['cache_misses'],
            'cache_hit_rate': (stats['cache_hits'] / cache_requests * 100) if cache_requests > 0 else 0.0,
            'faq_hits': stats['faq_hits'],
            'faq_hit_rate': (stats['faq_hits'] / total_queries * 100) if total_queries > 0 else 0.0,
            'ai_calls': stats['ai_calls'],
            'ai_call_rate': (stats['ai_calls'] / total_queries * 100) if total_queries > 0 else 0.0,
            'total_cost': stats['total_cost'],
            'average_cost_per_query': stats['total_cost'] / total_queries if total_queries > 0 else 0.0,
            'average_conversation_length': sum(stats['conversation_lengths']) / len(stats['conversation_lengths']) if stats['conversation_lengths'] else 0
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global statistics across all sessions.
        
        Returns:
            Global statistics dictionary
        """
        if not self.session_stats:
            return {
                'total_sessions': 0,
                'total_queries': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'average_cache_hit_rate': 0.0
            }
        
        total_sessions = len(self.session_stats)
        total_queries = sum(s['queries'] for s in self.session_stats.values())
        total_tokens = sum(s['total_tokens'] for s in self.session_stats.values())
        total_cost = sum(s['total_cost'] for s in self.session_stats.values())
        total_cache_hits = sum(s['cache_hits'] for s in self.session_stats.values())
        total_cache_requests = sum(s['cache_hits'] + s['cache_misses'] for s in self.session_stats.values())
        
        return {
            'total_sessions': total_sessions,
            'total_queries': total_queries,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'average_tokens_per_query': total_tokens / total_queries if total_queries > 0 else 0,
            'average_cost_per_query': total_cost / total_queries if total_queries > 0 else 0.0,
            'total_cache_hits': total_cache_hits,
            'total_cache_requests': total_cache_requests,
            'average_cache_hit_rate': (total_cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0,
            'total_faq_hits': sum(s['faq_hits'] for s in self.session_stats.values()),
            'total_ai_calls': sum(s['ai_calls'] for s in self.session_stats.values())
        }
    
    def save_stats(self, session_id: str = None):
        """
        Save statistics to file.
        
        Args:
            session_id: Optional session ID to save, or None for all
        """
        if session_id:
            stats = self.get_session_stats(session_id)
            stats_file = self.analytics_dir / f"{session_id}_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        else:
            # Save all stats
            global_stats = self.get_global_stats()
            global_file = self.analytics_dir / "global_stats.json"
            with open(global_file, 'w', encoding='utf-8') as f:
                json.dump(global_stats, f, indent=2, ensure_ascii=False)
            
            # Save per-session stats
            for sid in self.session_stats.keys():
                self.save_stats(sid)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data formatted for admin dashboard.
        
        Returns:
            Dashboard data dictionary
        """
        global_stats = self.get_global_stats()
        
        # Get top sessions by queries
        top_sessions = sorted(
            [(sid, self.get_session_stats(sid)) for sid in self.session_stats.keys()],
            key=lambda x: x[1]['queries'],
            reverse=True
        )[:10]
        
        return {
            'global': global_stats,
            'top_sessions': [
                {
                    'session_id': sid,
                    'stats': stats
                }
                for sid, stats in top_sessions
            ],
            'timestamp': datetime.now().isoformat()
        }

