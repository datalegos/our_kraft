# Module: utils

## Purpose
This module provides utility functions and client classes for interacting with Wazuh Manager and Indexer APIs, data collection, conversion, and vulnerability detection. It serves as a shared utilities library for the project.

## Key Components
- `wazuh_manager_client.py`: REST API client for Wazuh Manager with JWT authentication
  - Handles authentication, agent information retrieval, and various Wazuh Manager endpoints
- `wazuh_indexer_client.py`: REST API client for Wazuh Indexer with HTTP Basic Auth
  - Retrieves agent information and data from Wazuh Indexer
- `data_collector.py`: Base class and utilities for collecting data from Wazuh services
  - Organizes collected data by agent folders
  - Handles data saving and folder structure management
- `csv_converter.py`: Utility functions for converting JSON files to CSV format
  - Flattens nested dictionaries
  - Maintains folder structure during conversion
- `vulnerability_detector.py`: Vulnerability detection utilities (if implemented)

## Dependencies
- **External Libraries**: `requests` (HTTP client), `json`, `csv`, `pathlib`
- **Config Files**: 
  - `../config/config.yaml`: Wazuh Manager and Indexer connection settings
- **Other Modules**: None (standalone utility module)
- **Data Sources**: 
  - Input: Data from Wazuh Manager/Indexer APIs
  - Output: JSON/CSV files in `../collected_data/` directory

## Entry Points
- `WazuhManagerClient`: Main client for Wazuh Manager API
  - `authenticate()`: Get JWT token
  - `get_agents()`: Retrieve all agents
  - Various endpoint methods for Wazuh Manager operations
- `WazuhIndexerClient`: Main client for Wazuh Indexer API
  - `get_agents()`: Retrieve agents from indexer
  - Query methods for indexer data
- `DataCollector`: Base class for data collection operations
  - `save_data()`: Save collected data to files
  - Folder structure management
- `json_to_csv_simple()`: Convert JSON to CSV format
- `flatten_dict()`: Flatten nested dictionaries

## Configuration
- **Wazuh Manager Config**: Configured via `config/config.yaml`
  - Manager host, port, protocol
  - Authentication credentials (username, password)
  - SSL verification settings
  - Timeout configurations
- **Wazuh Indexer Config**: Configured via `config/config.yaml`
  - Indexer host, port, protocol
  - HTTP Basic Auth credentials
  - SSL verification settings

## Data Flow
1. **Client Initialization**: Load configuration from `config/config.yaml`
2. **Authentication**: 
   - Manager: JWT token authentication
   - Indexer: HTTP Basic Auth
3. **Data Collection**: 
   - Make API requests to Wazuh services
   - Organize data by agent folders
   - Save to JSON files in `collected_data/` directory
4. **Data Conversion**: 
   - Convert JSON files to CSV format
   - Maintain folder structure
   - Flatten nested data structures

## Usage Example
```python
from utils.wazuh_manager_client import WazuhManagerClient
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize client
client = WazuhManagerClient(config)

# Authenticate
if client.authenticate():
    # Get agents
    agents = client.get_agents()
    print(f"Found {len(agents)} agents")
```

## Notes
- All HTTP requests include proper error handling and timeout management
- SSL verification can be configured per client
- Data collection maintains the same folder structure as reference implementations
- CSV conversion utilities handle nested JSON structures

