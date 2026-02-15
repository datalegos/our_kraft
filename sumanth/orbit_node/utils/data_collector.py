"""
Data Collector - Knowledge Graph Data Collection
Collects data from Wazuh Manager and Indexer based on configuration
Organizes data by agent folders similar to reference structure
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class DataCollector:
    """Base class for data collectors"""
    
    def __init__(self, client, config: Dict[str, Any], collector_name: str):
        self.client = client
        self.config = config
        self.collector_name = collector_name
        self.output_dir = Path(config['collection']['output_dir'])
        self.collector_dir = self.output_dir / collector_name
        self.collector_dir.mkdir(parents=True, exist_ok=True)
        
        # Retry configuration
        collector_config = config.get('collectors', {}).get(collector_name, {})
        self.max_retries = collector_config.get('max_retries', config.get('collection', {}).get('max_retries', 3))
        self.retry_delay = collector_config.get('retry_delay', config.get('collection', {}).get('retry_delay', 5))
    
    def _has_error(self, data: Dict[str, Any]) -> bool:
        """Check if collection data has errors"""
        if 'error' in data:
            return True
        if 'data' in data:
            failed_items = data['data'].get('total_failed_items', 0)
            if failed_items > 0:
                return True
        return False
    
    def _retry_collection(self, collect_func, *args, **kwargs) -> Dict[str, Any]:
        """Retry collection with exponential backoff"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = collect_func(*args, **kwargs)
                
                # Check if collection was successful
                if not self._has_error(result):
                    if attempt > 0:
                        print(f"  ✓ Collection succeeded on retry attempt {attempt + 1}")
                    return result
                else:
                    last_error = result.get('error', 'Unknown error')
                    print(f"  ✗ Collection attempt {attempt + 1} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                print(f"  ✗ Collection attempt {attempt + 1} failed with exception: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"  Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        # All retries failed
        print(f"  ✗ All {self.max_retries} retry attempts failed for {self.collector_name}")
        return {
            "error": f"Collection failed after {self.max_retries} attempts: {last_error}",
            "data": {"affected_items": [], "total_affected_items": 0, "total_failed_items": 1},
            "retries_exhausted": True
        }
    
    def save_data(self, data: Any, filename: str = None, agent_id: str = None) -> str:
        """Save collected data to JSON file, optionally in agent subfolder"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.collector_name}_{timestamp}.json"
        
        # Create agent subfolder if agent_id is provided
        if agent_id:
            agent_dir = self.collector_dir / f"agent_{agent_id}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            filepath = agent_dir / filename
        else:
            filepath = self.collector_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)


class AgentCollector(DataCollector):
    """Collect agent information"""
    
    def __init__(self, client, config: Dict[str, Any], source: str):
        collector_name = f"agents_{source}"
        super().__init__(client, config, collector_name)
        self.source = source
        self.limit = config['collectors']['agents'].get('limit', 1000)
    
    def collect(self) -> Dict[str, Any]:
        """Collect all agents with retry logic"""
        print(f"Collecting agents from Wazuh {self.source.capitalize()}...")
        
        def collect_agents():
            if self.source == "manager":
                data = self.client.get_agents(limit=self.limit)
                result = {**data, 'source': 'manager'}
            else:  # indexer
                data = self.client.search_agents(size=self.limit)
                agents = []
                if 'aggregations' in data and 'unique_agents' in data['aggregations']:
                    for bucket in data['aggregations']['unique_agents']['buckets']:
                        agent_id = bucket['key']
                        agent_info = bucket.get('agent_info', {}).get('hits', {}).get('hits', [])
                        if agent_info:
                            source_data = agent_info[0].get('_source', {})
                            agent_data = source_data.get('agent', {})
                            agents.append({
                                'id': agent_id,
                                'name': agent_data.get('name', ''),
                                'ip': agent_data.get('ip', ''),
                                'os': agent_data.get('os', {})
                            })
                result = {
                    'data': {
                        'affected_items': agents,
                        'total_affected_items': len(agents),
                        'total_failed_items': 0,
                        'failed_items': []
                    },
                    'message': f'All agents information was returned from {self.source}',
                    'error': 0,
                    'source': self.source
                }
            return result
        
        result = self._retry_collection(collect_agents)
        
        filepath = self.save_data(result, "All_Agents.json")
        agent_count = result.get('data', {}).get('total_affected_items', 0)
        
        if not self._has_error(result):
            print(f"  ✓ Successfully collected {agent_count} agents")
        else:
            print(f"  ✗ Failed to collect agents after retries")
        
        print(f"Saved agents data to {filepath}")
        return result


class HostCollector(DataCollector):
    """Collect host/OS information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "host")
        self.limit = config['collectors']['host'].get('limit', 1000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect host data for each agent with retry logic"""
        print(f"Collecting host/OS information for {len(agent_ids)} agents...")
        all_hosts = {}
        failed_agents = []
        
        for agent_id in agent_ids:
            def collect_agent():
                return self.client.get_host(agent_id=agent_id)
            
            result = self._retry_collection(collect_agent)
            
            if not self._has_error(result):
                all_hosts[agent_id] = result
                # Save per agent
                filename = f"Syscollector_OS_Info_{agent_id}.json"
                self.save_data(result, filename, agent_id)
            else:
                print(f"  ✗ Failed to collect host data for agent {agent_id} after retries")
                all_hosts[agent_id] = result
                failed_agents.append(agent_id)
        
        # Save summary
        filepath = self.save_data(all_hosts, "Host_Summary.json")
        
        if failed_agents:
            print(f"  ⚠ Failed to collect host data for {len(failed_agents)} agents: {failed_agents}")
        else:
            print(f"  ✓ Successfully collected host data for all {len(agent_ids)} agents")
        
        print(f"Saved host data to {filepath}")
        return all_hosts


class PackagesCollector(DataCollector):
    """Collect packages information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "packages")
        self.limit = config['collectors']['packages'].get('limit', 1000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect packages data for each agent with retry logic"""
        print(f"Collecting packages information for {len(agent_ids)} agents...")
        all_packages = {}
        failed_agents = []
        
        for agent_id in agent_ids:
            def collect_agent():
                return self.client.get_packages(agent_id=agent_id, limit=self.limit)
            
            result = self._retry_collection(collect_agent)
            
            if not self._has_error(result):
                all_packages[agent_id] = result
                # Save per agent
                filename = f"Syscollector_Packages_{agent_id}.json"
                self.save_data(result, filename, agent_id)
            else:
                print(f"  ✗ Failed to collect packages for agent {agent_id} after retries")
                all_packages[agent_id] = result
                failed_agents.append(agent_id)
        
        # Save summary
        filepath = self.save_data(all_packages, "Packages_Summary.json")
        
        if failed_agents:
            print(f"  ⚠ Failed to collect packages for {len(failed_agents)} agents: {failed_agents}")
        else:
            print(f"  ✓ Successfully collected packages for all {len(agent_ids)} agents")
        
        print(f"Saved packages data to {filepath}")
        return all_packages


class HardwareCollector(DataCollector):
    """Collect hardware information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "hardware")
        self.limit = config['collectors']['hardware'].get('limit', 1000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect hardware data for each agent with retry logic"""
        print(f"Collecting hardware information for {len(agent_ids)} agents...")
        all_hardware = {}
        failed_agents = []
        
        for agent_id in agent_ids:
            def collect_agent():
                return self.client.get_hardware(agent_id=agent_id)
            
            result = self._retry_collection(collect_agent)
            
            if not self._has_error(result):
                all_hardware[agent_id] = result
                # Save per agent
                filename = f"Syscollector_Hardware_{agent_id}.json"
                self.save_data(result, filename, agent_id)
            else:
                print(f"  ✗ Failed to collect hardware for agent {agent_id} after retries")
                all_hardware[agent_id] = result
                failed_agents.append(agent_id)
        
        # Save summary
        filepath = self.save_data(all_hardware, "Hardware_Summary.json")
        
        if failed_agents:
            print(f"  ⚠ Failed to collect hardware for {len(failed_agents)} agents: {failed_agents}")
        else:
            print(f"  ✓ Successfully collected hardware for all {len(agent_ids)} agents")
        
        print(f"Saved hardware data to {filepath}")
        return all_hardware


class GroupsCollector(DataCollector):
    """Collect groups information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "groups")
        self.limit = config['collectors']['groups'].get('limit', 1000)
    
    def collect(self) -> Dict[str, Any]:
        """Collect all groups with retry logic"""
        print("Collecting groups information...")
        
        def collect_groups():
            data = self.client.get_groups(limit=self.limit)
            result = {**data, 'source': 'manager'}
            return result
        
        result = self._retry_collection(collect_groups)
        
        # Save groups data if successful
        if not self._has_error(result) and 'data' in result and 'affected_items' in result['data']:
            for group in result['data']['affected_items']:
                group_name = group.get('name', 'unknown')
                group_dir = self.collector_dir / f"group_{group_name}"
                group_dir.mkdir(parents=True, exist_ok=True)
                filename = f"Group_Agents_{group_name}.json"
                filepath = group_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(group, f, indent=2, ensure_ascii=False)
        
        filepath = self.save_data(result, "Groups_List.json")
        
        if not self._has_error(result):
            print(f"  ✓ Successfully collected groups")
        else:
            print(f"  ✗ Failed to collect groups after retries")
        
        print(f"Saved groups data to {filepath}")
        return result


class FIMCollector(DataCollector):
    """Collect FIM (File Integrity Monitoring) information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "fim")
        self.limit = config['collectors']['fim'].get('limit', 5000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect FIM data for each agent with retry logic"""
        print(f"Collecting FIM information for {len(agent_ids)} agents...")
        all_fim = {}
        failed_agents = []
        
        for agent_id in agent_ids:
            def collect_agent():
                return self.client.get_fim(agent_id=agent_id, limit=self.limit)
            
            result = self._retry_collection(collect_agent)
            
            if not self._has_error(result):
                all_fim[agent_id] = result
                # Save per agent
                filename = f"File_Integrity_Monitoring_{agent_id}.json"
                self.save_data(result, filename, agent_id)
            else:
                print(f"  ✗ Failed to collect FIM for agent {agent_id} after retries")
                all_fim[agent_id] = result
                failed_agents.append(agent_id)
        
        # Save summary
        filepath = self.save_data(all_fim, "FIM_Summary.json")
        
        if failed_agents:
            print(f"  ⚠ Failed to collect FIM for {len(failed_agents)} agents: {failed_agents}")
        else:
            print(f"  ✓ Successfully collected FIM for all {len(agent_ids)} agents")
        
        print(f"Saved FIM data to {filepath}")
        return all_fim


class VulnerabilitiesCollector(DataCollector):
    """Collect vulnerabilities from Indexer"""
    
    def __init__(self, indexer_client, config: Dict[str, Any]):
        super().__init__(indexer_client, config, "vulnerabilities")
        self.size = config['collectors']['vulnerabilities'].get('size', 1000)
    
    def collect(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Collect vulnerabilities from Indexer"""
        print("Collecting vulnerabilities from Wazuh Indexer...")
        all_vulnerabilities = {}
        
        try:
            # Collect for all agents or specific agents
            if agent_ids:
                for agent_id in agent_ids:
                    try:
                        data = self.client.search_vulnerabilities(agent_id=agent_id, size=self.size)
                        items = []
                        if 'hits' in data and 'hits' in data['hits']:
                            for hit in data['hits']['hits']:
                                source = hit.get('_source', {})
                                # Match the working reference structure
                                vuln = source.get('vulnerability', {})
                                package = source.get('package', {})
                                agent = source.get('agent', {})
                                
                                items.append({
                                    'agent_id': agent.get('id', agent_id),
                                    'agent_name': agent.get('name', ''),
                                    'cve': vuln.get('id', ''),  # vulnerability.id is the CVE
                                    'severity': vuln.get('severity', ''),
                                    'title': vuln.get('title', ''),
                                    'description': vuln.get('description', ''),
                                    'published': vuln.get('published', ''),
                                    'status': vuln.get('status', ''),
                                    'cvss': vuln.get('cvss', {}),
                                    'package_name': package.get('name', ''),
                                    'package_version': package.get('version', ''),
                                    'timestamp': source.get('@timestamp', source.get('timestamp', ''))
                                })
                        
                        all_vulnerabilities[agent_id] = {
                            'data': {
                                'affected_items': items,
                                'total_affected_items': len(items)
                            },
                            'source': 'indexer'
                        }
                        # Save per agent
                        filename = f"Vulnerabilities_{agent_id}.json"
                        self.save_data(all_vulnerabilities[agent_id], filename, agent_id)
                    except Exception as e:
                        print(f"Error collecting vulnerabilities for agent {agent_id}: {e}")
                        all_vulnerabilities[agent_id] = {"error": str(e)}
            else:
                # Collect all vulnerabilities
                data = self.client.search_vulnerabilities(size=self.size)
                items = []
                if 'hits' in data and 'hits' in data['hits']:
                    for hit in data['hits']['hits']:
                        source = hit.get('_source', {})
                        # Match the working reference structure
                        vuln = source.get('vulnerability', {})
                        package = source.get('package', {})
                        agent = source.get('agent', {})
                        
                        items.append({
                            'agent_id': agent.get('id', ''),
                            'agent_name': agent.get('name', ''),
                            'cve': vuln.get('id', ''),  # vulnerability.id is the CVE
                            'severity': vuln.get('severity', ''),
                            'title': vuln.get('title', ''),
                            'description': vuln.get('description', ''),
                            'published': vuln.get('published', ''),
                            'status': vuln.get('status', ''),
                            'cvss': vuln.get('cvss', {}),
                            'package_name': package.get('name', ''),
                            'package_version': package.get('version', ''),
                            'timestamp': source.get('@timestamp', source.get('timestamp', ''))
                        })
                
                all_vulnerabilities['all'] = {
                    'data': {
                        'affected_items': items,
                        'total_affected_items': len(items)
                    },
                    'source': 'indexer'
                }
            
            # Save summary
            filepath = self.save_data(all_vulnerabilities, "Vulnerabilities_Summary.json")
            print(f"Saved vulnerabilities data to {filepath}")
            return all_vulnerabilities
            
        except Exception as e:
            print(f"Error collecting vulnerabilities: {e}")
            error_data = {"error": str(e), "source": "indexer"}
            filepath = self.save_data(error_data, "Vulnerabilities_Summary.json")
            return error_data
