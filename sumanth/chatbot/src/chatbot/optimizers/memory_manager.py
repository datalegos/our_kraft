"""
Conversation memory management with sliding window, summarization, and compression.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI

from chatbot.core.config import OPENAI_API_KEY, OPENAI_MODEL
from chatbot.utils.logger import logger
from chatbot.optimizers.cost_optimizer import TokenCounter


class ConversationMemory:
    """
    Manages conversation history with sliding window and summarization.
    """
    
    def __init__(
        self,
        max_recent_messages: int = 10,
        summarization_threshold: int = 20,
        storage_dir: str = "conversations"
    ):
        """
        Initialize conversation memory manager.
        
        Args:
            max_recent_messages: Maximum messages to keep in recent context
            summarization_threshold: Number of messages before summarization
            storage_dir: Directory to store conversation history
        """
        self.max_recent_messages = max_recent_messages
        self.summarization_threshold = summarization_threshold
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.token_counter = TokenCounter()
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    
    def get_recent_messages(
        self,
        messages: List[Dict[str, str]],
        limit: int = None
    ) -> List[Dict[str, str]]:
        """
        Get only the most recent messages (sliding window).
        
        Args:
            messages: Full message history
            limit: Maximum messages to return (defaults to max_recent_messages)
        
        Returns:
            List of recent messages
        """
        limit = limit or self.max_recent_messages
        return messages[-limit:] if len(messages) > limit else messages
    
    def compress_message(self, text: str) -> str:
        """
        Compress message by removing unnecessary whitespace and formatting.
        
        Args:
            text: Message text to compress
        
        Returns:
            Compressed text
        """
        if not text:
            return text
        
        # Remove extra whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        # Join with single space, but preserve paragraph breaks
        compressed = ' '.join(lines)
        
        # Remove multiple spaces
        while '  ' in compressed:
            compressed = compressed.replace('  ', ' ')
        
        return compressed.strip()
    
    def summarize_messages(
        self,
        messages: List[Dict[str, str]],
        session_id: str = None
    ) -> str:
        """
        Summarize old messages when conversation exceeds threshold.
        
        Args:
            messages: Messages to summarize
            session_id: Optional session ID for caching
        
        Returns:
            Summary string
        """
        if not messages or len(messages) < self.summarization_threshold:
            return ""
        
        if not self.openai_client:
            logger.warning("OpenAI client not available, skipping summarization")
            return ""
        
        try:
            # Get messages to summarize (all except recent ones)
            messages_to_summarize = messages[:-self.max_recent_messages]
            
            # Format messages for summarization
            conversation_text = "\n".join([
                f"{msg.get('role', 'user').title()}: {msg.get('content', '')}"
                for msg in messages_to_summarize
            ])
            
            # Compress before summarization
            conversation_text = self.compress_message(conversation_text)
            
            # Create summarization prompt
            summary_prompt = f"""Summarize the following conversation in 2-3 sentences, focusing on:
- Key topics discussed
- Important information shared
- User's main questions or concerns

Conversation:
{conversation_text}

Summary:"""
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            logger.info(f"Summarized {len(messages_to_summarize)} messages into summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing messages: {e}", exc_info=True)
            return ""
    
    def prepare_context(
        self,
        messages: List[Dict[str, str]],
        session_id: str = None
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Prepare conversation context with sliding window and summarization.
        
        Args:
            messages: Full message history
            session_id: Optional session ID
        
        Returns:
            Tuple of (recent_messages, summary)
        """
        if not messages:
            return [], None
        
        summary = None
        
        # Check if summarization is needed
        if len(messages) > self.summarization_threshold:
            summary = self.summarize_messages(messages, session_id)
        
        # Get recent messages
        recent_messages = self.get_recent_messages(messages)
        
        # Compress messages
        compressed_messages = []
        for msg in recent_messages:
            compressed_content = self.compress_message(msg.get('content', ''))
            compressed_messages.append({
                'role': msg.get('role', 'user'),
                'content': compressed_content
            })
        
        return compressed_messages, summary
    
    def save_conversation(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        metadata: Dict[str, Any] = None
    ):
        """
        Save full conversation history to disk.
        
        Args:
            session_id: Unique session identifier
            messages: Full message history
            metadata: Optional metadata
        """
        try:
            file_path = self.storage_dir / f"{session_id}.json"
            
            data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'messages': messages,
                'message_count': len(messages),
                'metadata': metadata or {}
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved conversation {session_id} with {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}", exc_info=True)
    
    def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation history from disk.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Conversation data or None
        """
        try:
            file_path = self.storage_dir / f"{session_id}.json"
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}", exc_info=True)
            return None
    
    def get_conversation_stats(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Get statistics about conversation.
        
        Args:
            messages: Message history
        
        Returns:
            Statistics dictionary
        """
        if not messages:
            return {
                'total_messages': 0,
                'total_tokens': 0,
                'user_messages': 0,
                'assistant_messages': 0
            }
        
        total_tokens = sum(
            self.token_counter.estimate_tokens(msg.get('content', ''))
            for msg in messages
        )
        
        user_messages = sum(1 for msg in messages if msg.get('role') == 'user')
        assistant_messages = sum(1 for msg in messages if msg.get('role') == 'assistant')
        
        return {
            'total_messages': len(messages),
            'total_tokens': total_tokens,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'needs_summarization': len(messages) > self.summarization_threshold
        }

