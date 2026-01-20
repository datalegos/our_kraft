import json
import requests
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def generate(self, prompt: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

class OllamaProvider(LLMProvider):
    """Local Ollama provider"""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str, **kwargs) -> str:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 1000)
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=data, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

class LLMService:
    """Service for managing LLM interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['llm']
        self.primary_provider = self._initialize_provider(
            self.config['provider'], 
            self.config
        )
        self.fallback_provider = None
        
        # Initialize fallback if configured
        if self.config.get('fallback_provider'):
            try:
                self.fallback_provider = self._initialize_provider(
                    self.config['fallback_provider'],
                    self.config
                )
            except Exception as e:
                print(f"Warning: Fallback provider initialization failed: {e}")
    
    def _initialize_provider(self, provider_type: str, config: Dict[str, Any]) -> LLMProvider:
        if provider_type == "openai":
            return OpenAIProvider(
                api_key=config['api_key'],
                model=config['model'],
                temperature=config['temperature']
            )
        elif provider_type == "local" or provider_type == "ollama":
            return OllamaProvider(model=config.get('local_model', 'llama3.1:8b'))
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using primary provider with fallback"""
        try:
            return self.primary_provider.generate(prompt, **kwargs)
        except Exception as e:
            print(f"Primary LLM provider failed: {e}")
            
            if self.fallback_provider:
                try:
                    print("Attempting fallback provider...")
                    return self.fallback_provider.generate(prompt, **kwargs)
                except Exception as fallback_error:
                    print(f"Fallback provider also failed: {fallback_error}")
            
            raise Exception(f"All LLM providers failed. Last error: {e}")
    
    def generate_cypher(self, user_query: str, schema_info: str, conversation_context: List[str] = None) -> Dict[str, Any]:
        """Generate Cypher query from natural language"""
        
        context_str = ""
        if conversation_context:
            context_str = f"\nConversation Context:\n" + "\n".join(conversation_context[-3:])
        
        prompt = f"""You are an expert Neo4j Cypher query generator. Convert natural language questions into Cypher queries.

Database Schema:
{schema_info}

{context_str}

User Question: {user_query}

Instructions:
1. Generate a valid Cypher query that answers the user's question
2. Explain your reasoning
3. If the question is ambiguous, make reasonable assumptions
4. Use proper Neo4j syntax and best practices
5. Return response in this JSON format:

{{
    "cypher": "MATCH (n) RETURN n LIMIT 10",
    "explanation": "This query finds...",
    "assumptions": ["Assumed X means Y"],
    "confidence": 0.9
}}

Response:"""

        try:
            response = self.generate(prompt)
            # Try to parse JSON response
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                # Fallback if not JSON
                return {
                    "cypher": response,
                    "explanation": "Generated query",
                    "assumptions": [],
                    "confidence": 0.7
                }
        except json.JSONDecodeError:
            # Extract cypher from text response
            lines = response.split('\n')
            cypher_line = next((line for line in lines if 'MATCH' in line or 'CREATE' in line), "")
            return {
                "cypher": cypher_line,
                "explanation": response,
                "assumptions": [],
                "confidence": 0.5
            }
    
    def analyze_results(self, query: str, results: List[Dict], user_question: str) -> str:
        """Analyze query results and provide insights"""
        
        prompt = f"""Analyze these Neo4j query results and provide insights.

User Question: {user_question}
Query: {query}
Results Count: {len(results)}
Sample Results: {results[:3] if results else "No results"}

Provide:
1. Summary of findings
2. Key insights
3. Suggestions for follow-up questions

Keep response concise and user-friendly."""

        return self.generate(prompt, max_tokens=500)
    
    def suggest_corrections(self, failed_query: str, error_message: str, schema_info: str) -> str:
        """Suggest corrections for failed queries"""
        
        prompt = f"""A Cypher query failed. Suggest corrections.

Failed Query: {failed_query}
Error: {error_message}
Schema: {schema_info}

Provide a corrected query with explanation."""

        return self.generate(prompt, max_tokens=300)