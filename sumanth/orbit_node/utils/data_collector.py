"""
Data Collector - Knowledge Graph Data Collection
Collects data from Wazuh Manager and Indexer based on configuration
Organizes data by agent folders similar to reference structure
"""
import json
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
        """Collect all agents"""
        print(f"Collecting agents from Wazuh {self.source.capitalize()}...")
        
        try:
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
            
            filepath = self.save_data(result, "All_Agents.json")
            agent_count = result.get('data', {}).get('total_affected_items', 0)
            print(f"Saved {agent_count} agents to {filepath}")
            return result
            
        except Exception as e:
            print(f"Error collecting agents from {self.source}: {e}")
            error_data = {
                "error": str(e),
                "data": {"affected_items": [], "total_affected_items": 0},
                "source": self.source
            }
            filepath = self.save_data(error_data, "All_Agents.json")
            return error_data


class HostCollector(DataCollector):
    """Collect host/OS information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "host")
        self.limit = config['collectors']['host'].get('limit', 1000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect host data for each agent"""
        print(f"Collecting host/OS information for {len(agent_ids)} agents...")
        all_hosts = {}
        
        for agent_id in agent_ids:
            try:
                data = self.client.get_host(agent_id=agent_id)
                all_hosts[agent_id] = data
                # Save per agent
                filename = f"Syscollector_OS_Info_{agent_id}.json"
                self.save_data(data, filename, agent_id)
            except Exception as e:
                print(f"Error collecting host data for agent {agent_id}: {e}")
                all_hosts[agent_id] = {"error": str(e)}
        
        # Save summary
        filepath = self.save_data(all_hosts, "Host_Summary.json")
        print(f"Saved host data to {filepath}")
        return all_hosts


class PackagesCollector(DataCollector):
    """Collect packages information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "packages")
        self.limit = config['collectors']['packages'].get('limit', 1000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect packages data for each agent"""
        print(f"Collecting packages information for {len(agent_ids)} agents...")
        all_packages = {}
        
        for agent_id in agent_ids:
            try:
                data = self.client.get_packages(agent_id=agent_id, limit=self.limit)
                all_packages[agent_id] = data
                # Save per agent
                filename = f"Syscollector_Packages_{agent_id}.json"
                self.save_data(data, filename, agent_id)
            except Exception as e:
                print(f"Error collecting packages for agent {agent_id}: {e}")
                all_packages[agent_id] = {"error": str(e)}
        
        # Save summary
        filepath = self.save_data(all_packages, "Packages_Summary.json")
        print(f"Saved packages data to {filepath}")
        return all_packages


class HardwareCollector(DataCollector):
    """Collect hardware information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "hardware")
        self.limit = config['collectors']['hardware'].get('limit', 1000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect hardware data for each agent"""
        print(f"Collecting hardware information for {len(agent_ids)} agents...")
        all_hardware = {}
        
        for agent_id in agent_ids:
            try:
                data = self.client.get_hardware(agent_id=agent_id)
                all_hardware[agent_id] = data
                # Save per agent
                filename = f"Syscollector_Hardware_{agent_id}.json"
                self.save_data(data, filename, agent_id)
            except Exception as e:
                print(f"Error collecting hardware for agent {agent_id}: {e}")
                all_hardware[agent_id] = {"error": str(e)}
        
        # Save summary
        filepath = self.save_data(all_hardware, "Hardware_Summary.json")
        print(f"Saved hardware data to {filepath}")
        return all_hardware


class GroupsCollector(DataCollector):
    """Collect groups information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "groups")
        self.limit = config['collectors']['groups'].get('limit', 1000)
    
    def collect(self) -> Dict[str, Any]:
        """Collect all groups"""
        print("Collecting groups information...")
        
        try:
            data = self.client.get_groups(limit=self.limit)
            result = {**data, 'source': 'manager'}
            
            # Save groups data
            if 'data' in data and 'affected_items' in data['data']:
                for group in data['data']['affected_items']:
                    group_name = group.get('name', 'unknown')
                    group_dir = self.collector_dir / f"group_{group_name}"
                    group_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"Group_Agents_{group_name}.json"
                    filepath = group_dir / filename
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(group, f, indent=2, ensure_ascii=False)
            
            filepath = self.save_data(result, "Groups_List.json")
            print(f"Saved groups data to {filepath}")
            return result
            
        except Exception as e:
            print(f"Error collecting groups: {e}")
            error_data = {"error": str(e), "source": "manager"}
            filepath = self.save_data(error_data, "Groups_List.json")
            return error_data


class FIMCollector(DataCollector):
    """Collect FIM (File Integrity Monitoring) information"""
    
    def __init__(self, manager_client, config: Dict[str, Any]):
        super().__init__(manager_client, config, "fim")
        self.limit = config['collectors']['fim'].get('limit', 5000)
    
    def collect(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect FIM data for each agent"""
        print(f"Collecting FIM information for {len(agent_ids)} agents...")
        all_fim = {}
        
        for agent_id in agent_ids:
            try:
                data = self.client.get_fim(agent_id=agent_id, limit=self.limit)
                all_fim[agent_id] = data
                # Save per agent
                filename = f"File_Integrity_Monitoring_{agent_id}.json"
                self.save_data(data, filename, agent_id)
            except Exception as e:
                print(f"Error collecting FIM for agent {agent_id}: {e}")
                all_fim[agent_id] = {"error": str(e)}
        
        # Save summary
        filepath = self.save_data(all_fim, "FIM_Summary.json")
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
