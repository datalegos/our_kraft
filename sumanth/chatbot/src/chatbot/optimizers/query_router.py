"""
Query classification and routing system.
Routes queries to FAQ database, simple responses, or full AI processing.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

from chatbot.utils.logger import logger
from chatbot.optimizers.cost_optimizer import TokenCounter


class FAQDatabase:
    """Simple FAQ database for instant responses."""
    
    def __init__(self, faq_file: str = "faq_database.json"):
        """
        Initialize FAQ database.
        
        Args:
            faq_file: Path to FAQ JSON file
        """
        self.faq_file = Path(faq_file)
        self.faqs: List[Dict[str, str]] = []
        self.load_faqs()
    
    def load_faqs(self):
        """Load FAQs from file."""
        try:
            if self.faq_file.exists():
                with open(self.faq_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.faqs = data.get('faqs', [])
                logger.info(f"Loaded {len(self.faqs)} FAQs from {self.faq_file}")
            else:
                # Create default FAQ file
                self.create_default_faqs()
        except Exception as e:
            logger.error(f"Error loading FAQs: {e}", exc_info=True)
            self.faqs = []
    
    def create_default_faqs(self):
        """Create default FAQ file with example entries."""
        default_faqs = {
            'faqs': [
                {
                    'question': 'What are your business hours?',
                    'answer': 'Our business hours are Monday to Friday, 9 AM to 5 PM EST.',
                    'keywords': ['hours', 'time', 'open', 'when']
                },
                {
                    'question': 'How can I contact support?',
                    'answer': 'You can contact our support team via email at support@company.com or call us at 1-800-XXX-XXXX.',
                    'keywords': ['contact', 'support', 'help', 'email', 'phone']
                },
                {
                    'question': 'What is your return policy?',
                    'answer': 'We offer a 30-day return policy on all products. Items must be in original condition.',
                    'keywords': ['return', 'refund', 'policy', 'exchange']
                }
            ]
        }
        
        try:
            with open(self.faq_file, 'w', encoding='utf-8') as f:
                json.dump(default_faqs, f, indent=2, ensure_ascii=False)
            self.faqs = default_faqs['faqs']
            logger.info(f"Created default FAQ file with {len(self.faqs)} entries")
        except Exception as e:
            logger.error(f"Error creating default FAQs: {e}", exc_info=True)
    
    def search(self, query: str, threshold: float = 0.6) -> Optional[Dict[str, str]]:
        """
        Search FAQ database for matching question.
        
        Args:
            query: User query
            threshold: Similarity threshold (0-1)
        
        Returns:
            Matching FAQ entry or None
        """
        if not self.faqs:
            return None
        
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        best_match = None
        best_score = 0.0
        
        for faq in self.faqs:
            # Check question similarity
            question_lower = faq.get('question', '').lower()
            question_similarity = SequenceMatcher(None, query_lower, question_lower).ratio()
            
            # Check keyword matches
            keywords = faq.get('keywords', [])
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            keyword_score = keyword_matches / max(len(keywords), 1) if keywords else 0
            
            # Combined score
            score = (question_similarity * 0.7) + (keyword_score * 0.3)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = faq
        
        if best_match:
            logger.debug(f"FAQ match found: {best_match['question']} (score: {best_score:.2f})")
        
        return best_match


class QueryClassifier:
    """Classify queries into FAQ, Simple, or Complex categories."""
    
    def __init__(self, faq_database: FAQDatabase = None):
        """
        Initialize query classifier.
        
        Args:
            faq_database: FAQ database instance
        """
        self.faq_db = faq_database or FAQDatabase()
        self.token_counter = TokenCounter()
        
        # Simple query patterns (greetings, basic questions)
        self.simple_patterns = [
            r'^(hi|hello|hey|greetings)',
            r'^(thanks|thank you|thx)',
            r'^(bye|goodbye|see you)',
            r'^(yes|no|ok|okay|sure)',
            r'^what is your name',
            r'^who are you',
        ]
        
        # Complex query indicators
        self.complex_indicators = [
            'explain', 'describe', 'analyze', 'compare', 'difference',
            'how does', 'why does', 'what if', 'tell me about',
            'detailed', 'comprehensive', 'in depth'
        ]
    
    def classify(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Classify query into category.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (category, response)
            Category: 'faq', 'simple', or 'complex'
            Response: Pre-computed response if FAQ or simple, None if complex
        """
        if not query or not query.strip():
            return 'simple', "I'm here to help! What would you like to know?"
        
        query_lower = query.lower().strip()
        
        # Check FAQ first
        faq_match = self.faq_db.search(query)
        if faq_match:
            return 'faq', faq_match.get('answer', '')
        
        # Check simple patterns
        for pattern in self.simple_patterns:
            if re.match(pattern, query_lower, re.IGNORECASE):
                if re.match(r'^(hi|hello|hey|greetings)', query_lower):
                    return 'simple', "Hello! How can I help you today?"
                elif re.match(r'^(thanks|thank you|thx)', query_lower):
                    return 'simple', "You're welcome! Is there anything else I can help with?"
                elif re.match(r'^(bye|goodbye|see you)', query_lower):
                    return 'simple', "Goodbye! Have a great day!"
                elif re.match(r'^(what is your name|who are you)', query_lower):
                    return 'simple', "I'm an AI assistant here to help answer questions about our company."
        
        # Check for complex indicators
        query_tokens = self.token_counter.estimate_tokens(query)
        has_complex_indicators = any(indicator in query_lower for indicator in self.complex_indicators)
        
        if has_complex_indicators or query_tokens > 50:
            return 'complex', None
        
        # Default to simple for short queries
        if query_tokens < 20:
            return 'simple', None
        
        # Default to complex
        return 'complex', None
    
    def should_use_ai(self, query: str) -> bool:
        """
        Determine if query requires AI processing.
        
        Args:
            query: User query
        
        Returns:
            True if AI is needed, False otherwise
        """
        category, response = self.classify(query)
        return category == 'complex' or (category == 'simple' and response is None)


class QueryRouter:
    """Main query routing system."""
    
    def __init__(self, faq_database: FAQDatabase = None):
        """
        Initialize query router.
        
        Args:
            faq_database: FAQ database instance
        """
        self.classifier = QueryClassifier(faq_database)
        self.faq_db = faq_database or self.classifier.faq_db
    
    def route(self, query: str) -> Dict[str, any]:
        """
        Route query and return routing decision.
        
        Args:
            query: User query
        
        Returns:
            Dictionary with routing information
        """
        category, response = self.classifier.classify(query)
        use_ai = self.classifier.should_use_ai(query)
        
        result = {
            'category': category,
            'use_ai': use_ai,
            'response': response,
            'query': query
        }
        
        logger.debug(f"Query routed as '{category}', use_ai={use_ai}")
        
        return result

