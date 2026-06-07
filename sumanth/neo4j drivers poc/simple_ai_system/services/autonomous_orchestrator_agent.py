"""
Simple Autonomous Orchestrator Agent
Coordinates agents and services with simple logic
"""

from typing import Dict, Any
from .autonomous_agent import AutonomousAgent

class AutonomousOrchestratorAgent(AutonomousAgent):
    """
    Simple orchestrator that coordinates agents and services
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="OrchestratorAgent",
            role="Coordination Specialist",
            config=config
        )
        
        # Available services
        self.services = {}
    
    def register_service(self, name: str, service_instance):
        """Register a service"""
        self.services[name] = service_instance
    
    def _execute_simple_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute orchestration plan"""
        
        action = plan.get('action')
        goal = plan.get('goal')
        
        try:
            # Simple orchestration logic
            if action == 'execute_query':
                if 'ai_query_service' in self.services:
                    result = self.services['ai_query_service'].process(goal)
                    return {
                        'status': 'success',
                        'message': f"Query executed: {goal}",
                        'result': result
                    }
            
            elif action == 'load_data':
                if 'agentic_data_loader' in self.services:
                    result = self.services['agentic_data_loader'].process(goal)
                    return {
                        'status': 'success',
                        'message': f"Data loading initiated: {goal}",
                        'result': result
                    }
            
            elif action == 'show_schema':
                if 'schema_service' in self.services:
                    result = self.services['schema_service'].process("get_schema")
                    return {
                        'status': 'success',
                        'message': f"Schema retrieved: {goal}",
                        'result': result
                    }
            
            # Default fallback
            return {
                'status': 'success',
                'message': f"Orchestrated action: {action} for goal: {goal}"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Orchestration failed: {str(e)}",
                'error': str(e)
            }
    
    def orchestrate_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simple orchestration method
        """
        return self.process_request(user_input, context)