"""
Optimized AI chatbot application with comprehensive token optimization.
Integrates RAG, memory management, query routing, caching, and analytics.
"""
import uuid
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from pathlib import Path

from chatbot.core.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    RAG_K_DOCUMENTS,
    RAG_K_FINAL,
    RAG_USE_RERANKING,
    RAG_USE_CONTEXT_COMPRESSION,
    RAG_MAX_CONTEXT_TOKENS,
    SYSTEM_PROMPT_TEMPLATE,
    FALLBACK_NO_ANSWER,
    FALLBACK_ERROR,
    APP_TITLE,
    APP_DESCRIPTION,
    APP_HOST,
    APP_PORT,
    APP_SHARE,
    MEMORY_MAX_RECENT_MESSAGES,
    MEMORY_SUMMARIZATION_THRESHOLD,
    MEMORY_STORAGE_DIR,
    ROUTING_FAQ_FILE,
    ROUTING_SIMILARITY_THRESHOLD,
    ROUTING_ENABLED,
    CACHE_DIR,
    CACHE_DEFAULT_TTL_HOURS,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_ENABLED,
    TOKEN_MAX_INPUT,
    TOKEN_MAX_OUTPUT,
    TOKEN_STOP_SEQUENCES,
    ANALYTICS_ENABLED,
    ANALYTICS_SAVE_INTERVAL,
    validate_config,
)
from chatbot.optimizers.cost_optimizer import DocumentReranker, ContextCompressor, TokenCounter, TokenController
from chatbot.optimizers.memory_manager import ConversationMemory
from chatbot.optimizers.query_router import QueryRouter, FAQDatabase
from chatbot.optimizers.response_cache import ResponseCache
from chatbot.optimizers.analytics import AnalyticsTracker
from chatbot.utils.logger import logger
from chatbot.utils.exceptions import ConfigurationError, VectorStoreError, APIError


class OptimizedChatbotApp:
    """Fully optimized chatbot application with all token optimization strategies."""
    
    def __init__(self):
        """Initialize the optimized chatbot application."""
        self._validate_setup()
        self._load_models()
        self._initialize_openai()
        self._initialize_optimization_modules()
        self._initialize_session_tracking()
    
    def _validate_setup(self):
        """Validate configuration and dependencies."""
        issues = validate_config()
        
        if issues["errors"]:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in issues["errors"])
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        if issues["warnings"]:
            for warning in issues["warnings"]:
                logger.warning(warning)
    
    def _load_models(self):
        """Load embedding model and vector store."""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            
            index_path = Path(FAISS_INDEX_DIR)
            if not index_path.exists():
                raise VectorStoreError(
                    f"FAISS index directory '{FAISS_INDEX_DIR}' not found. "
                    "Please run embeddings.py first to create the index."
                )
            
            logger.info(f"Loading FAISS index from: {FAISS_INDEX_DIR}")
            self.vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to load vector store: {e}")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_API_KEY:
            raise ConfigurationError("OPENAI_API_KEY is not set")
        
        try:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"OpenAI client initialized with model: {OPENAI_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}")
    
    def _initialize_optimization_modules(self):
        """Initialize all optimization modules."""
        # Memory management
        self.memory_manager = ConversationMemory(
            max_recent_messages=MEMORY_MAX_RECENT_MESSAGES,
            summarization_threshold=MEMORY_SUMMARIZATION_THRESHOLD,
            storage_dir=MEMORY_STORAGE_DIR
        )
        
        # Query routing
        if ROUTING_ENABLED:
            faq_db = FAQDatabase(faq_file=ROUTING_FAQ_FILE)
            self.query_router = QueryRouter(faq_database=faq_db)
        else:
            self.query_router = None
        
        # Response caching
        if CACHE_ENABLED:
            self.response_cache = ResponseCache(
                cache_dir=CACHE_DIR,
                default_ttl_hours=CACHE_DEFAULT_TTL_HOURS,
                similarity_threshold=CACHE_SIMILARITY_THRESHOLD
            )
        else:
            self.response_cache = None
        
        # Cost optimizers
        self.reranker = DocumentReranker() if RAG_USE_RERANKING else None
        self.compressor = ContextCompressor(max_tokens=RAG_MAX_CONTEXT_TOKENS) if RAG_USE_CONTEXT_COMPRESSION else None
        self.token_counter = TokenCounter()
        self.token_controller = TokenController(
            max_input_tokens=TOKEN_MAX_INPUT,
            max_output_tokens=TOKEN_MAX_OUTPUT,
            stop_sequences=TOKEN_STOP_SEQUENCES
        )
        
        # Analytics
        if ANALYTICS_ENABLED:
            self.analytics = AnalyticsTracker()
        else:
            self.analytics = None
        
        logger.info("All optimization modules initialized")
    
    def _initialize_session_tracking(self):
        """Initialize session tracking."""
        self.sessions: dict = {}  # session_id -> conversation history
    
    def _get_or_create_session(self, session_id: str = None) -> str:
        """Get or create a session ID."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        return session_id
    
    def _convert_gradio_history(self, history) -> list:
        """Convert Gradio history format to internal format."""
        messages = []
        if history:
            for entry in history:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    messages.append({"role": "user", "content": entry[0]})
                    messages.append({"role": "assistant", "content": entry[1]})
        return messages
    
    def process_query(
        self,
        message: str,
        history,
        session_id: str = None
    ) -> str:
        """
        Process a user query with all optimizations.
        
        Args:
            message: User's question
            history: Conversation history from Gradio
            session_id: Optional session ID
        
        Returns:
            Answer string
        """
        if not message or not message.strip():
            return "Please provide a question."
        
        # Get or create session
        session_id = self._get_or_create_session(session_id)
        
        # Convert history
        conversation_history = self._convert_gradio_history(history)
        
        # Step 1: Check cache first
        if self.response_cache:
            cached = self.response_cache.get_cached_response(message)
            if cached:
                response = cached.get('response', '')
                logger.info(f"Cache hit for query: {message[:50]}...")
                
                # Track analytics
                if self.analytics:
                    self.analytics.track_query(
                        session_id, message, response,
                        cache_hit=True, used_ai=False
                    )
                
                return response
        
        # Step 2: Query routing (FAQ/Simple/Complex)
        routing_result = None
        if self.query_router:
            routing_result = self.query_router.route(message)
            
            # If FAQ or simple response available, return it
            if routing_result.get('response'):
                response = routing_result['response']
                logger.info(f"Query routed as '{routing_result['category']}', using pre-computed response")
                
                # Cache the response
                if self.response_cache:
                    self.response_cache.cache_response(message, response)
                
                # Track analytics
                if self.analytics:
                    self.analytics.track_query(
                        session_id, message, response,
                        faq_hit=(routing_result['category'] == 'faq'),
                        used_ai=False
                    )
                
                return response
        
        # Step 3: Check if AI is needed
        if routing_result and not routing_result.get('use_ai', True):
            return FALLBACK_NO_ANSWER
        
        # Step 4: Retrieve relevant context (RAG)
        try:
            initial_k = RAG_K_DOCUMENTS if not RAG_USE_RERANKING else max(RAG_K_DOCUMENTS, RAG_K_FINAL + 2)
            logger.debug(f"Retrieving {initial_k} documents for query: {message[:50]}...")
            docs = self.vectorstore.similarity_search(message, k=initial_k)
            
            if not docs:
                logger.warning("No documents retrieved for query")
                return FALLBACK_NO_ANSWER
            
            # Re-rank if enabled
            if self.reranker:
                logger.debug("Re-ranking documents...")
                docs = self.reranker.rerank(docs, message, top_k=RAG_K_FINAL)
            
            # Build context - use all retrieved documents for comprehensive answers
            # Disable compression to preserve all details (especially for team, address, contact info)
            context = "\n\n--- Document Chunk ---\n\n".join([doc.page_content for doc in docs])
            
            # Add metadata hints if available
            if docs and hasattr(docs[0], 'metadata'):
                sources = set()
                for doc in docs:
                    if 'source' in doc.metadata:
                        sources.add(doc.metadata['source'])
                if sources:
                    context = f"[Sources: {', '.join(sources)}]\n\n{context}"
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return FALLBACK_ERROR
        
        # Step 5: Prepare conversation context with memory management
        recent_messages, summary = self.memory_manager.prepare_context(
            conversation_history,
            session_id
        )
        
        # Step 6: Build optimized prompt
        prompt_structure = self.token_controller.build_optimized_prompt(
            context=context,
            messages=recent_messages,
            query=message,
            system_prompt=SYSTEM_PROMPT_TEMPLATE
        )
        
        # Add summary if available
        if summary:
            prompt_structure["messages"].insert(1, {
                "role": "system",
                "content": f"Previous conversation summary: {summary}"
            })
        
        # Step 7: Call OpenAI API
        try:
            estimated_tokens = self.token_controller.estimate_prompt_tokens(prompt_structure)
            logger.debug(f"Calling OpenAI API (estimated prompt: ~{estimated_tokens} tokens)")
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=prompt_structure["messages"],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=prompt_structure["max_tokens"],
                stop=prompt_structure.get("stop")
            )
            
            answer = response.choices[0].message.content
            
            # Get actual token usage
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                prompt_tokens = estimated_tokens
                completion_tokens = self.token_counter.estimate_tokens(answer)
                total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost
            cost = self.token_counter.estimate_cost(
                prompt_tokens,
                completion_tokens,
                OPENAI_MODEL
            )
            
            logger.info(
                f"Generated answer: {len(answer)} chars, "
                f"{total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens}), "
                f"cost: ${cost:.6f}"
            )
            
            # Step 8: Cache response
            if self.response_cache:
                self.response_cache.cache_response(message, answer)
            
            # Step 9: Update conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": answer})
            self.sessions[session_id] = conversation_history
            
            # Save conversation periodically
            if len(conversation_history) % 10 == 0:
                self.memory_manager.save_conversation(session_id, conversation_history)
            
            # Step 10: Track analytics
            if self.analytics:
                self.analytics.track_query(
                    session_id, message, answer,
                    tokens_used=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    used_ai=True
                )
                
                # Save stats periodically
                if self.analytics.session_stats[session_id]['queries'] % ANALYTICS_SAVE_INTERVAL == 0:
                    self.analytics.save_stats(session_id)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in AI processing: {e}", exc_info=True)
            return FALLBACK_ERROR
    
    def rag_qa(self, message, history):
        """
        Main entry point for Gradio interface.
        
        Args:
            message: User's question
            history: Conversation history from Gradio
        
        Returns:
            Answer string
        """
        return self.process_query(message, history)
    
    def launch(self):
        """Launch the Gradio interface."""
        logger.info(f"Launching optimized chatbot: {APP_TITLE}")
        
        demo = gr.ChatInterface(
            fn=self.rag_qa,
            type="messages",
            title=APP_TITLE,
            description=APP_DESCRIPTION,
        )
        
        logger.info(f"Starting server on {APP_HOST}:{APP_PORT}")
        demo.launch(
            server_name=APP_HOST,
            server_port=APP_PORT,
            share=APP_SHARE,
        )


def main():
    """Main entry point for the application."""
    try:
        app = OptimizedChatbotApp()
        app.launch()
    except (ConfigurationError, VectorStoreError) as e:
        logger.error(f"Failed to start application: {e}")
        print(f"\n[ERROR] Error: {e}\n")
        print("Please check your configuration and try again.")
        return 1
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n[INFO] Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}\n")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
