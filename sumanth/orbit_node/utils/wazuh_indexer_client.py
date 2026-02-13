"""
Wazuh Indexer Client
Simple client to retrieve agent information from Wazuh Indexer
"""
import requests
from requests.auth import HTTPBasicAuth
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class WazuhIndexerClient:
    """Client for interacting with Wazuh Indexer API to get agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.host = config['wazuh']['indexer']['host']
        self.port = config['wazuh']['indexer']['port']
        self.protocol = config['wazuh']['indexer']['protocol']
        self.username = config['wazuh']['indexer']['username']
        self.password = config['wazuh']['indexer']['password']
        self.verify_ssl = config['wazuh']['indexer']['verify_ssl']
        self.timeout = config['wazuh']['indexer']['timeout']
        
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        self.auth = HTTPBasicAuth(self.username, self.password)
    
    def _make_request(self, method: str, endpoint: str, body: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to Wazuh Indexer API"""
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.request(
                method=method,
                url=url,
                auth=self.auth,
                headers=headers,
                json=body,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            raise
    
    def get_latest_index(self, pattern: str, days_back: int = 30) -> str:
        """Get the latest index matching pattern (e.g., wazuh-alerts-4.x-YYYY.MM.DD or wazuh-states-vulnerabilities-YYYY.MM.DD)"""
        # Try to list indices and find the latest
        try:
            endpoint = f"/_cat/indices/{pattern}-*?format=json&s=index:desc"
            indices = self._make_request('GET', endpoint)
            if indices and len(indices) > 0:
                return indices[0].get('index', pattern)
        except:
            pass
        
        # Fallback: try current date and go back
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y.%m.%d")
            index = f"{pattern}-{date}"
            try:
                # Test if index exists by doing a small search
                self._make_request('POST', f"/{index}/_search", body={"size": 0, "query": {"match_all": {}}})
                return index
            except:
                continue
        
        # If no date-based index found, return pattern as-is (for wildcard matching)
        return pattern
    
    def search_agents(self, size: int = 1000) -> Dict[str, Any]:
        """Search for agent information from Wazuh Indexer"""
        index = self.get_latest_index("wazuh-alerts-4.x")
        
        # Query to get unique agents using aggregation
        body = {
            "size": 0,
            "aggs": {
                "unique_agents": {
                    "terms": {
                        "field": "agent.id",
                        "size": size
                    },
                    "aggs": {
                        "agent_info": {
                            "top_hits": {
                                "size": 1,
                                "_source": {
                                    "includes": ["agent.id", "agent.name", "agent.ip"]
                                }
                            }
                        }
                    }
                }
            }
        }
        
        endpoint = f"/{index}/_search"
        return self._make_request('POST', endpoint, body=body)
    
    def search_fim(self, agent_id: Optional[str] = None, size: int = 1000) -> Dict[str, Any]:
        """Search FIM data from Wazuh Indexer - essential fields only"""
        index = self.get_latest_index("wazuh-alerts-4.x")
        
        must_clauses = [
            {"term": {"rule.groups": "syscheck"}}
        ]
        
        if agent_id:
            must_clauses.append({"term": {"agent.id": agent_id}})
        
        query = {
            "bool": {
                "must": must_clauses
            }
        }
        
        body = {
            "query": query,
            "size": size,
            "_source": ["agent.id", "agent.name", "syscheck.path", "syscheck.event", "syscheck.md5_after", "timestamp"]
        }
        
        endpoint = f"/{index}/_search"
        return self._make_request('POST', endpoint, body=body)
    
    def search_vulnerabilities(self, agent_id: Optional[str] = None, size: int = 1000) -> Dict[str, Any]:
        """Search vulnerabilities from Wazuh Indexer - uses wazuh-states-vulnerabilities index"""
        # Use the correct index for vulnerabilities (not wazuh-alerts)
        index = self.get_latest_index("wazuh-states-vulnerabilities")
        
        # Simple query like the working reference - match_all with optional agent filter
        if agent_id:
            query = {
                "bool": {
                    "must": [
                        {"match_all": {}},
                        {"term": {"agent.id": agent_id}}
                    ]
                }
            }
        else:
            query = {
                "match_all": {}
            }
        
        body = {
            "size": size,
            "query": query,
            "_source": [
                "agent.id", "agent.name",
                "vulnerability.id", "vulnerability.severity",
                "vulnerability.title", "vulnerability.published",
                "vulnerability.status", "vulnerability.description",
                "vulnerability.cvss",
                "package.name", "package.version",
                "@timestamp", "timestamp"
            ]
        }
        
        endpoint = f"/{index}/_search"
        return self._make_request('POST', endpoint, body=body)
    
    def test_connection(self) -> bool:
        """Test connection to Wazuh Indexer"""
        try:
            self._make_request('GET', '/')
            return True
        except Exception as e:
            print(f"Indexer connection test failed: {e}")
            return False
