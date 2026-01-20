"""
Simple Autonomous Agent Base Class
Simplified version for demo purposes
"""

import json
from typing import Dict, Any, List
from .llm_service import LLMService

class AutonomousAgent:
    """
    Simple autonomous agent that can make decisions and execute actions
    """
    
    def __init__(self, name: str, role: str, config: Dict[str, Any]):
        self.name = name
        self.role = role
        self.config = config
        self.llm_service = LLMService(config)
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simple autonomous processing
        """
        
        try:
            # Step 1: Understand what to do
            goal = self._understand_request(user_input)
            
            # Step 2: Decide how to do it
            action_plan = self._create_simple_plan(goal, context)
            
            # Step 3: Execute the plan (subclasses must implement this)
            if hasattr(self, '_execute_simple_plan'):
                result = self._execute_simple_plan(action_plan)
            else:
                # Base class fallback - just return the plan
                result = {
                    'status': 'success',
                    'message': f"Plan created: {action_plan['action']} for {goal}",
                    'action_plan': action_plan
                }
            
            # Step 4: Generate response
            response = self._generate_response(result)
            
            return {
                'status': 'success',
                'response': response,
                'goal': goal,
                'action_taken': action_plan.get('action', 'unknown')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'response': f"I encountered an error: {str(e)}",
                'error': str(e)
            }
    
    def _understand_request(self, user_input: str) -> str:
        """Simple request understanding"""
        
        prompt = f"""Understand this user request and state what they want in one clear sentence:

User Request: "{user_input}"

What does the user want? (one sentence):
"""
        
        try:
            goal = self.llm_service.generate(prompt, max_tokens=100)
            return goal.strip()
        except:
            return f"Process user request: {user_input}"
    
    def _create_simple_plan(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a simple action plan"""
        
        goal_lower = goal.lower()
        
        # Simple decision logic
        if any(word in goal_lower for word in ['query', 'find', 'show', 'get', 'how many']):
            return {'action': 'execute_query', 'goal': goal}
        elif any(word in goal_lower for word in ['load', 'import', 'upload']):
            return {'action': 'load_data', 'goal': goal}
        elif any(word in goal_lower for word in ['schema', 'structure', 'tables']):
            return {'action': 'show_schema', 'goal': goal}
        else:
            return {'action': 'execute_query', 'goal': goal}  # Default
    
    def _generate_response(self, result: Dict[str, Any]) -> str:
        """Generate a simple response"""
        
        if result.get('status') == 'success':
            return result.get('message', 'Task completed successfully')
        else:
            return f"Task failed: {result.get('error', 'Unknown error')}"