"""
Cost optimization utilities for RAG system.
Includes context compression, re-ranking, and token counting.
"""
from typing import List, Dict, Any
from langchain.docstore.document import Document

from chatbot.utils.logger import logger


class ContextCompressor:
    """
    Compress retrieved context to reduce token usage while preserving key information.
    """
    
    def __init__(self, max_tokens: int = 2000):
        """
        Initialize context compressor.
        
        Args:
            max_tokens: Maximum tokens to keep in compressed context
        """
        self.max_tokens = max_tokens
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 characters).
        
        Args:
            text: Text to estimate
        
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def compress_context(self, documents: List[Document], query: str) -> str:
        """
        Compress context by keeping most relevant parts.
        
        Args:
            documents: List of retrieved documents
            query: User query for relevance filtering
        
        Returns:
            Compressed context string
        """
        if not documents:
            return ""
        
        # Simple compression: prioritize documents with query terms
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            # Score based on query term matches
            score = sum(1 for term in query_terms if term in content_lower)
            scored_docs.append((score, doc))
        
        # Sort by relevance
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Build compressed context
        compressed_parts = []
        total_tokens = 0
        
        for score, doc in scored_docs:
            doc_tokens = self._estimate_tokens(doc.page_content)
            
            if total_tokens + doc_tokens <= self.max_tokens:
                compressed_parts.append(doc.page_content)
                total_tokens += doc_tokens
            else:
                # Truncate if needed
                remaining_tokens = self.max_tokens - total_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    truncated = self._truncate_text(doc.page_content, remaining_tokens)
                    compressed_parts.append(truncated)
                break
        
        return "\n---\n".join(compressed_parts)
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit, trying to preserve sentences."""
        max_chars = max_tokens * 4  # Rough conversion
        
        if len(text) <= max_chars:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        # Use the later of period or newline
        cut_point = max(last_period, last_newline)
        
        if cut_point > max_chars * 0.7:  # Only if we keep at least 70%
            return truncated[:cut_point + 1] + "..."
        
        return truncated + "..."


class DocumentReranker:
    """
    Re-rank documents by relevance to query.
    Simple keyword-based re-ranking (can be enhanced with cross-encoders).
    """
    
    def rerank(self, documents: List[Document], query: str, top_k: int = None) -> List[Document]:
        """
        Re-rank documents by relevance to query.
        
        Args:
            documents: List of documents to re-rank
            query: Query string
            top_k: Return only top K documents (None for all)
        
        Returns:
            Re-ranked list of documents
        """
        if not documents:
            return []
        
        query_terms = set(query.lower().split())
        
        # Score each document
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # Calculate relevance score
            # 1. Term frequency
            term_matches = sum(1 for term in query_terms if term in content_lower)
            
            # 2. Position bonus (terms near start are more important)
            position_bonus = 0
            for term in query_terms:
                pos = content_lower.find(term)
                if pos != -1:
                    # Closer to start = higher bonus
                    position_bonus += max(0, 1 - (pos / len(content_lower)))
            
            # 3. Title/source bonus (if query term in filename)
            source_bonus = 0
            if 'filename' in doc.metadata:
                filename_lower = doc.metadata['filename'].lower()
                source_bonus = sum(1 for term in query_terms if term in filename_lower) * 0.5
            
            total_score = term_matches + position_bonus + source_bonus
            scored_docs.append((total_score, doc))
        
        # Sort by score (descending)
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Return top K
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return [doc for _, doc in scored_docs]


class TokenCounter:
    """Utility for counting tokens in text."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count (rough: 1 token ≈ 4 characters for English).
        
        Args:
            text: Text to count
        
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    @staticmethod
    def estimate_cost(
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o-mini"
    ) -> float:
        """
        Estimate cost in USD (approximate).
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
        
        Returns:
            Estimated cost in USD
        """
        # Approximate pricing (as of 2024, check OpenAI for current rates)
        pricing = {
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
        }
        
        model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
        
        input_cost = prompt_tokens * model_pricing["input"]
        output_cost = completion_tokens * model_pricing["output"]
        
        return input_cost + output_cost


class TokenController:
    """Advanced token control with truncation, stop sequences, and streaming support."""
    
    def __init__(self, max_input_tokens: int = 500, max_output_tokens: int = 300, stop_sequences: List[str] = None):
        """
        Initialize token controller.
        
        Args:
            max_input_tokens: Maximum tokens for input truncation
            max_output_tokens: Maximum tokens for output
            stop_sequences: List of stop sequences (defaults to empty list)
        """
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.token_counter = TokenCounter()
        self.stop_sequences = stop_sequences if stop_sequences is not None else []
    
    def truncate_input(self, text: str, max_tokens: int = None) -> str:
        """
        Truncate input text to fit token limit, preserving sentence boundaries.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (defaults to max_input_tokens)
        
        Returns:
            Truncated text
        """
        max_tokens = max_tokens or self.max_input_tokens
        
        if not text:
            return text
        
        estimated_tokens = self.token_counter.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Calculate max characters (rough estimate)
        max_chars = max_tokens * 4
        
        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        
        # Find last sentence ending
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        last_newline = truncated.rfind('\n')
        
        # Use the latest sentence boundary
        cut_point = max(last_period, last_exclamation, last_question, last_newline)
        
        if cut_point > max_chars * 0.7:  # Only if we keep at least 70%
            return truncated[:cut_point + 1] + "..."
        
        return truncated + "..."
    
    def get_stop_sequences(self) -> List[str]:
        """
        Get stop sequences for API calls.
        
        Returns:
            List of stop sequences
        """
        return self.stop_sequences
    
    def build_optimized_prompt(
        self,
        context: str,
        messages: List[Dict[str, str]],
        query: str,
        system_prompt: str
    ) -> Dict[str, any]:
        """
        Build optimized prompt with token limits and structure.
        
        Args:
            context: Retrieved context
            messages: Conversation messages
            query: Current query
            system_prompt: System prompt template
        
        Returns:
            Optimized prompt structure
        """
        # Truncate query if needed
        query = self.truncate_input(query, self.max_input_tokens)
        
        # Format system prompt with context
        formatted_system = system_prompt.format(context=context, query=query)
        
        # Truncate system prompt if needed (shouldn't happen, but safety check)
        formatted_system = self.truncate_input(formatted_system, self.max_input_tokens * 2)
        
        # Build message structure
        prompt_messages = [{"role": "system", "content": formatted_system}]
        
        # Add conversation history (already truncated by memory manager)
        for msg in messages:
            content = msg.get('content', '')
            # Additional safety truncation
            content = self.truncate_input(content, self.max_input_tokens)
            prompt_messages.append({
                "role": msg.get('role', 'user'),
                "content": content
            })
        
        # Add current query
        prompt_messages.append({"role": "user", "content": query})
        
        return {
            "messages": prompt_messages,
            "max_tokens": self.max_output_tokens,
            "stop": self.stop_sequences
        }
    
    def estimate_prompt_tokens(self, prompt_structure: Dict[str, any]) -> int:
        """
        Estimate total tokens in prompt structure.
        
        Args:
            prompt_structure: Prompt structure from build_optimized_prompt
        
        Returns:
            Estimated token count
        """
        total = 0
        for msg in prompt_structure.get("messages", []):
            total += self.token_counter.estimate_tokens(msg.get("content", ""))
        return total


