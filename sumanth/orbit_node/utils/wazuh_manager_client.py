"""
Wazuh Manager REST API Client
Simple client to retrieve agent information from Wazuh Manager
Uses JWT token authentication
"""
import requests
from typing import Dict, Any, Optional


class WazuhManagerClient:
    """Client for interacting with Wazuh Manager REST API with JWT authentication"""
    
    def __init__(self, config: Dict[str, Any]):
        self.host = config['wazuh']['manager']['host']
        self.port = config['wazuh']['manager']['port']
        self.protocol = config['wazuh']['manager']['protocol']
        self.username = config['wazuh']['manager']['username']
        self.password = config['wazuh']['manager']['password']
        self.verify_ssl = config['wazuh']['manager']['verify_ssl']
        self.timeout = config['wazuh']['manager']['timeout']
        
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        self.token = None
        self.session = requests.Session()
    
    def authenticate(self) -> bool:
        """Authenticate and get JWT token"""
        url = f"{self.base_url}/security/user/authenticate"
        
        try:
            response = requests.post(
                url,
                auth=(self.username, self.password),
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract token from response
            if 'data' in data and 'token' in data['data']:
                self.token = data['data']['token']
                # Set up session with token
                self.session.headers.update({
                    'Authorization': f'Bearer {self.token}',
                    'Content-Type': 'application/json'
                })
                return True
            else:
                print("No token received in authentication response")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Authentication failed: {e}")
            return False
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to Wazuh Manager API with JWT token"""
        if not self.token:
            if not self.authenticate():
                raise Exception("Authentication failed. Cannot make request.")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    def get_agents(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """Get agents from Wazuh Manager REST API"""
        # Match reference code exactly - no select parameter
        params = {
            'pretty': 'true',
            'limit': str(limit)
        }
        
        # Add sort if limit is used
        if limit > 0:
            params['sort'] = 'id'
        
        # Add offset if specified
        if offset > 0:
            params['offset'] = str(offset)
        
        return self._make_request('GET', '/agents', params=params)
    
    def get_fim(self, agent_id: str, limit: int = 5000) -> Dict[str, Any]:
        """Get FIM (File Integrity Monitoring) data from Wazuh Manager REST API"""
        params = {
            'pretty': 'true',
            'limit': str(limit)
        }
        endpoint = f'/syscheck/{agent_id}'
        return self._make_request('GET', endpoint, params=params)
    
    def get_host(self, agent_id: str) -> Dict[str, Any]:
        """Get host/OS information from Wazuh Manager REST API"""
        params = {
            'pretty': 'true'
        }
        endpoint = f'/syscollector/{agent_id}/os'
        return self._make_request('GET', endpoint, params=params)
    
    def get_packages(self, agent_id: str, limit: int = 1000) -> Dict[str, Any]:
        """Get packages information from Wazuh Manager REST API"""
        params = {
            'pretty': 'true',
            'limit': str(limit)
        }
        endpoint = f'/syscollector/{agent_id}/packages'
        return self._make_request('GET', endpoint, params=params)
    
    def get_hardware(self, agent_id: str) -> Dict[str, Any]:
        """Get hardware information from Wazuh Manager REST API"""
        params = {
            'pretty': 'true'
        }
        endpoint = f'/syscollector/{agent_id}/hardware'
        return self._make_request('GET', endpoint, params=params)
    
    def get_cpe(self, agent_id: str, limit: int = 1000) -> Dict[str, Any]:
        """Get CPE (Common Platform Enumeration) information from Wazuh Manager REST API"""
        params = {
            'pretty': 'true',
            'limit': str(limit)
        }
        endpoint = f'/syscollector/{agent_id}/packages'
        # Note: CPE data is included in the packages endpoint response
        # Each package may have a 'cpe' field with CPE identifier
        return self._make_request('GET', endpoint, params=params)
    
    def get_groups(self, limit: int = 1000) -> Dict[str, Any]:
        """Get groups information from Wazuh Manager REST API"""
        params = {
            'pretty': 'true',
            'limit': str(limit)
        }
        endpoint = '/groups'
        return self._make_request('GET', endpoint, params=params)
    
    def test_connection(self) -> bool:
        """Test connection to Wazuh Manager by authenticating"""
        return self.authenticate()

