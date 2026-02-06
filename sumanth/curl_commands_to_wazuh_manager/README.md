# Wazuh API Demo Client

An advanced Python program that executes comprehensive HTTP GET requests against the Wazuh API with dynamic agent discovery and JWT authentication.

## Features

- **Fully config-driven** - YAML-based configuration with no hardcoded values
- **JWT Authentication** - Secure token-based authentication
- **Dynamic Agent Discovery** - Automatically discovers and queries all agents
- **Organized Output Structure** - Results categorized by manager, agents, groups, and static data
- **Comprehensive Coverage** - Queries 55+ different API endpoints across all categories
- **Detailed Reporting** - Generates endpoint summary with success/failure statistics
- **Flexible Filtering** - Configurable agent and data retrieval limits
- **Error Handling** - Graceful handling of failed requests with detailed logging
- **Vulnerability Tracking** - CVE detection and Windows hotfix monitoring
- **SCA Policy Checks** - Automated security configuration assessment with detailed checks
- **Network Protocol Stats** - Complete network visibility including protocol-level metrics

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Update `config.yaml` with your Wazuh API details:
   - Change `base_url` to your Wazuh manager URL
   - Update authentication credentials in the `auth` section
   - Adjust data retrieval limits and filters as needed

## Usage

Run the program:
```bash
python wazuh_api_demo.py
```

## Configuration

The `config.yaml` file contains several main sections:

### Authentication
- **auth**: JWT authentication configuration with username/password

### Data Retrieval Settings
- **data_retrieval**: Configurable limits for packages, processes, ports, etc.
- **agent_discovery**: Settings for automatic agent discovery and filtering

### Request Categories
- **static_requests**: Manager info, rules, decoders, security settings
- **agent_specific_requests**: Per-agent data like syscollector, FIM, rootcheck
- **group_specific_requests**: Group configurations and member lists

## Output Structure

Results are saved in timestamped directories with organized subdirectories:

```
output/
├── 20240203_143022/
│   ├── manager/
│   │   ├── Manager_Info.json
│   │   ├── Manager_Configuration.json
│   │   └── Manager_Stats.json
│   ├── static/
│   │   ├── All_Agents.json
│   │   ├── Rules_List.json
│   │   └── Security_Users.json
│   ├── agents/
│   │   ├── agent_001_WebServer/
│   │   │   ├── Syscollector_OS_Info.json
│   │   │   ├── Syscollector_Packages.json
│   │   │   └── File_Integrity_Monitoring.json
│   │   └── agent_002_DatabaseServer/
│   │       └── ...
│   ├── groups/
│   │   ├── group_default/
│   │   └── group_webservers/
│   └── wazuh_endpoints_summary_20240203_143022.txt
```

Each execution creates a new directory with format: `YYYYMMDD_HHMMSS`

## Key Endpoints Covered

### Manager & System
- Root API info, manager status, configuration, and statistics
- Manager logs and log summaries
- Manager files and API configuration
- Cluster status, configuration, nodes, and health
- Security configuration, users, roles, and policies
- Task status monitoring

### Agents & Assets  
- Agent discovery with filtering capabilities
- Agent summaries, status, and statistics
- Outdated agents and agents without groups
- Agent configuration and daemon statistics
- Syscollector data (OS, hardware, packages, processes, network, protocols, hotfixes)

### Security Monitoring
- File Integrity Monitoring (FIM) data
- Rootcheck results
- Security Configuration Assessment (SCA) with detailed policy checks
- Vulnerability detection and CVE tracking
- MITRE ATT&CK techniques and tactics

### Rules & Intelligence
- Detection rules and decoders
- CDB lists and list files
- Group configurations and memberships

### Network Monitoring
- Network interfaces and addresses
- Open ports and connections
- Network protocol statistics (TCP/UDP/ICMP)

## Example Configuration

```yaml
auth:
  type: basic
  username: wazuh-wui
  password: your_password

base_url: https://your-wazuh-manager:55000

data_retrieval:
  get_full_data: true
  max_packages: 10000
  max_processes: 5000

agent_discovery:
  enabled: true
  max_agents: 0  # 0 = get all agents
  agent_filters:
    # status: active  # Optional filters
    # os.platform: linux

static_requests:
  - name: Manager Info
    endpoint: /manager/info
    params:
      pretty: true

agent_specific_requests:
  - name: Syscollector OS Info
    endpoint: /syscollector/{agent_id}/os
    params:
      pretty: true
```
```

## Advanced Features

### Dynamic Agent Discovery
The tool automatically discovers all agents in your Wazuh environment and executes agent-specific queries for each one. You can filter agents by status, OS platform, or group membership.

### Comprehensive Data Collection
- **Manager Data**: System status, configuration, statistics, and cluster information
- **Agent Inventory**: Hardware, software packages, running processes, and network interfaces  
- **Security Monitoring**: File integrity monitoring, rootcheck results, and security assessments
- **Intelligence**: Detection rules, decoders, and MITRE ATT&CK mappings

### Organized Output
Results are automatically organized into logical subdirectories (manager, agents, groups, static) with descriptive filenames and a comprehensive summary report.

## Notes

- Uses JWT authentication for secure API access
- SSL verification is disabled for demo purposes (enable in production)
- Failed requests are logged but don't stop execution
- Configurable limits prevent overwhelming the API with large datasets
- Generates detailed endpoint summary with success/failure statistics
- Supports filtering and limiting data retrieval for performance optimization

## Dependencies

- `requests==2.31.0` - HTTP library for API calls
- `PyYAML==6.0.1` - YAML configuration file parsing