#!/usr/bin/env python3
"""
Advanced Wazuh API Demo Client
Executes HTTP GET requests with dynamic agent discovery based on YAML configuration.
"""

import json
import os
import requests
import yaml
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_output_directory():
    """Create timestamped output directory with agent subfolders."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "manager").mkdir(exist_ok=True)
    (output_dir / "static").mkdir(exist_ok=True)
    (output_dir / "agents").mkdir(exist_ok=True)
    (output_dir / "groups").mkdir(exist_ok=True)
    
    return output_dir, timestamp


def get_jwt_token(base_url, username, password):
    """Get JWT token for authentication."""
    try:
        response = requests.post(
            f"{base_url}/security/user/authenticate",
            auth=(username, password),
            verify=False,
            timeout=30
        )
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('data', {}).get('token')
        else:
            print(f"Failed to get JWT token: HTTP {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting JWT token: {e}")
        return None


def discover_agents(session, base_url, agent_config):
    """Discover available agents based on configuration filters."""
    print("🔍 Discovering agents...")
    
    try:
        params = {"pretty": "true", "limit": "1000"}
        
        # Apply filters from configuration
        if "agent_filters" in agent_config and agent_config["agent_filters"]:
            filters = agent_config["agent_filters"]
            print(f"   Applying filters: {filters}")
            for key, value in filters.items():
                params[key] = value
        else:
            print("   No filters applied - getting all agents")
        
        print(f"   Request params: {params}")
        
        response = session.get(
            f"{base_url}/agents",
            params=params,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            agents = data.get('data', {}).get('affected_items', [])
            total_agents = data.get('data', {}).get('total_affected_items', 0)
            
            print(f"   Total agents in system: {total_agents}")
            print(f"   Agents matching filters: {len(agents)}")
            
            # Limit number of agents if specified
            max_agents = agent_config.get('max_agents', 0)
            if max_agents > 0:
                print(f"   Limiting to first {max_agents} agents")
                agents = agents[:max_agents]
            
            print(f"✅ Found {len(agents)} agents")
            for agent in agents:
                status = agent.get('status', 'unknown')
                last_seen = agent.get('lastKeepAlive', 'never')
                print(f"   - Agent {agent['id']}: {agent['name']} (Status: {status}, Last seen: {last_seen})")
            
            return agents
        else:
            print(f"❌ Failed to discover agents: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Error discovering agents: {e}")
        return []


def discover_groups(session, base_url):
    """Discover available groups."""
    print("🔍 Discovering groups...")
    
    try:
        response = session.get(
            f"{base_url}/groups",
            params={"pretty": "true", "limit": "100"},
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            groups = data.get('data', {}).get('affected_items', [])
            print(f"✅ Found {len(groups)} groups")
            for group in groups:
                print(f"   - Group: {group['name']}")
            return groups
        else:
            print(f"❌ Failed to discover groups: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error discovering groups: {e}")
        return []


def execute_request(name, url, params, session, results, output_subdir=None):
    """Execute a single HTTP GET request and store result."""
    endpoint = urlparse(url).path
    
    print(f"Executing: {name}")
    print(f"URL: {url}")
    
    try:
        response = session.get(url, params=params, verify=False, timeout=30)
        
        result = {
            "endpoint": endpoint,
            "name": name,
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "error": None if response.status_code == 200 else f"HTTP {response.status_code}",
            "output_subdir": output_subdir
        }
        
        if response.status_code == 200:
            try:
                result["response"] = response.json()
            except json.JSONDecodeError:
                result["response"] = response.text
                result["error"] = "Invalid JSON response"
        
        results.append(result)
        
        status_symbol = "✅" if response.status_code == 200 else "❌"
        print(f"{status_symbol} Status: {response.status_code}")
        print("-" * 50)
        
    except requests.exceptions.RequestException as e:
        result = {
            "endpoint": endpoint,
            "name": name,
            "status_code": None,
            "success": False,
            "error": str(e),
            "output_subdir": output_subdir
        }
        results.append(result)
        print(f"❌ Error: {e}")
        print("-" * 50)


def categorize_endpoint(endpoint):
    """Categorize endpoint based on its path."""
    if endpoint == "/":
        return "General / Manager"
    elif endpoint.startswith("/manager"):
        return "General / Manager"
    elif endpoint.startswith("/agents") and "/stats" in endpoint:
        return "Agent Stats"
    elif endpoint.startswith("/agents") and "/config" in endpoint:
        return "Agent Config"
    elif endpoint.startswith("/agents"):
        return "Agents / Assets"
    elif endpoint.startswith("/groups"):
        return "Groups"
    elif endpoint.startswith("/syscollector"):
        return "Inventory / Syscollector"
    elif endpoint.startswith("/vulnerability"):
        return "Vulnerability Detection"
    elif endpoint.startswith("/syscheck"):
        return "File Integrity Monitoring"
    elif endpoint.startswith("/rootcheck"):
        return "Rootcheck"
    elif endpoint.startswith("/sca"):
        return "Security Configuration Assessment"
    elif endpoint.startswith("/rules") or endpoint.startswith("/decoders"):
        return "Rules & Decoders"
    elif endpoint.startswith("/mitre"):
        return "MITRE ATT&CK"
    elif endpoint.startswith("/cluster"):
        return "Cluster"
    elif endpoint.startswith("/security"):
        return "Security"
    else:
        return "Other"


def generate_endpoint_summary(results, output_dir, timestamp):
    """Generate endpoint summary in the requested format."""
    categories = {}
    for result in results:
        category = categorize_endpoint(result["endpoint"])
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    summary_lines = []
    summary_lines.append(f"/* ===== WAZUH API ENDPOINTS SUMMARY - {timestamp} ===== */")
    summary_lines.append(f"/* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} */")
    summary_lines.append(f"/* Total Endpoints Tested: {len(results)} */")
    summary_lines.append("")
    
    category_order = [
        "General / Manager", "Agents / Assets", "Agent Stats", "Agent Config",
        "Groups", "Inventory / Syscollector", "Vulnerability Detection",
        "File Integrity Monitoring", "Rootcheck", "Security Configuration Assessment",
        "Rules & Decoders", "MITRE ATT&CK", "Cluster", "Security", "Other"
    ]
    
    for category in category_order:
        if category in categories:
            summary_lines.append(f"/* ===== {category} ===== */")
            for result in categories[category]:
                status_symbol = "✓" if result["success"] else "✗"
                error_info = f" // {result['error']}" if result["error"] else ""
                summary_lines.append(f'"{result["endpoint"]}", // {status_symbol} {result["name"]}{error_info}')
            summary_lines.append("")
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    summary_lines.append("/* ===== EXECUTION STATISTICS ===== */")
    summary_lines.append(f"/* Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%) */")
    summary_lines.append(f"/* Failed: {failed}/{len(results)} ({failed/len(results)*100:.1f}%) */")
    
    summary_file = output_dir / f"wazuh_endpoints_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n📄 Endpoint summary saved to: {summary_file}")
    return summary_file


def main():
    """Main execution function."""
    config = load_config("config.yaml")
    
    output_dir, timestamp = create_output_directory()
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    # Get JWT token
    auth_config = config["auth"]
    base_url = config["base_url"]
    
    print("🔐 Getting JWT token...")
    token = get_jwt_token(base_url, auth_config["username"], auth_config["password"])
    
    if not token:
        print("❌ Failed to get JWT token. Please check your credentials.")
        return
    
    print("✅ JWT token obtained successfully")
    print("=" * 60)
    
    # Create session with JWT token
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    })
    
    results = []
    
    # Execute static requests
    print("📋 Executing static requests...")
    for request in config["static_requests"]:
        url = f"{base_url}{request['endpoint']}"
        params = request.get('params', {})
        
        # Determine output subdirectory
        if request['endpoint'].startswith('/manager'):
            subdir = "manager"
        else:
            subdir = "static"
            
        execute_request(request['name'], url, params, session, results, subdir)
    
    # Discover and execute agent-specific requests
    if config.get("agent_discovery", {}).get("enabled", False):
        print("\n" + "=" * 60)
        print("🔍 Agent discovery is enabled")
        agents = discover_agents(session, base_url, config["agent_discovery"])
        print(f"🔍 Discovered {len(agents)} agents")
        
        # Debug: Check if agent_specific_requests exists
        if "agent_specific_requests" in config:
            print(f"✅ Found {len(config['agent_specific_requests'])} agent-specific request templates")
        else:
            print("❌ No agent_specific_requests found in config!")
            return
        
        if agents and "agent_specific_requests" in config:
            print(f"\n📋 Executing agent-specific requests for {len(agents)} agents...")
            
            for agent in agents:
                agent_id = agent["id"]
                agent_name = agent["name"]
                agent_status = agent.get("status", "unknown")
                
                print(f"\n--- Processing Agent {agent_id} ({agent_name}) - Status: {agent_status} ---")
                
                # Create agent-specific directory
                agent_dir = output_dir / "agents" / f"agent_{agent_id}_{agent_name.replace(' ', '_')}"
                agent_dir.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {agent_dir}")
                
                for request in config["agent_specific_requests"]:
                    endpoint = request['endpoint'].replace('{agent_id}', agent_id)
                    url = f"{base_url}{endpoint}"
                    params = request.get('params', {})
                    name = f"{request['name']} ({agent_id})"
                    
                    execute_request(name, url, params, session, results, f"agents/agent_{agent_id}_{agent_name.replace(' ', '_')}")
        else:
            if not agents:
                print("❌ No agents discovered")
            if "agent_specific_requests" not in config:
                print("❌ No agent_specific_requests in config")
    else:
        print("❌ Agent discovery is disabled")
    
    # Discover and execute group-specific requests
    if "group_specific_requests" in config:
        print("\n" + "=" * 60)
        groups = discover_groups(session, base_url)
        
        if groups:
            print(f"\n📋 Executing group-specific requests for {len(groups)} groups...")
            
            for group in groups:
                group_name = group["name"]
                
                # Create group-specific directory
                group_dir = output_dir / "groups" / f"group_{group_name}"
                group_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\n--- Group {group_name} ---")
                
                for request in config["group_specific_requests"]:
                    endpoint = request['endpoint'].replace('{group_id}', group_name)
                    url = f"{base_url}{endpoint}"
                    params = request.get('params', {})
                    name = f"{request['name']} ({group_name})"
                    
                    execute_request(name, url, params, session, results, f"groups/group_{group_name}")
    
    # Save individual JSON files organized by subdirectory
    print(f"\n💾 Saving individual response files...")
    for result in results:
        if result["success"] and "response" in result:
            # Determine output path
            if result.get("output_subdir"):
                output_path = output_dir / result["output_subdir"]
            else:
                output_path = output_dir / "static"
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Clean filename
            filename = f"{result['name'].replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')}.json"
            output_file = output_path / filename
            
            with open(output_file, 'w') as f:
                json.dump(result["response"], f, indent=2)
    
    # Generate summary
    summary_file = generate_endpoint_summary(results, output_dir, timestamp)
    
    print(f"\n🎉 All requests completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"   📂 Manager data: {output_dir}/manager/")
    print(f"   📂 Static data: {output_dir}/static/")
    print(f"   📂 Agent data: {output_dir}/agents/")
    print(f"   📂 Group data: {output_dir}/groups/")
    print(f"📄 Summary file: {summary_file}")


if __name__ == "__main__":
    main()