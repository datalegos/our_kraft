#!/usr/bin/env python3
"""
Wazuh Indexer CPE Fetcher
Fetches unique CPE (Common Platform Enumeration) information from Wazuh Indexer (OpenSearch).
"""

import yaml
import json
import urllib3
from opensearchpy import OpenSearch
from pathlib import Path
from datetime import datetime

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def connect_to_indexer(config):
    """Create OpenSearch client connection."""
    indexer_config = config.get('indexer', {})
    
    # Check for required fields
    if not indexer_config:
        print("No 'indexer' configuration found in config.yaml")
        return None

    host = indexer_config.get('url', 'https://localhost:9200')
    username = indexer_config.get('username')
    password = indexer_config.get('password')
    verify_certs = indexer_config.get('verify_certs', False)

    if '<' in username or '<' in password:
        print("Please update config.yaml with actual username and password.")
        return None

    print(f"Connecting to Wazuh Indexer at {host}...")
    
    try:
        # Extract host and port from URL if necessary, but opensearch-py handles URLs well
        # If the URL has protocol, we can pass it directly.
        # However, opensearch-py expects a list of hosts or a single host dict/string.
        
        # Parse the host string to handle http/https
        if "://" in host:
            protocol, rest = host.split("://")
            host_addr = rest
            use_ssl = (protocol == "https")
        else:
            host_addr = host
            use_ssl = True

        client = OpenSearch(
            hosts=[host],
            http_auth=(username, password),
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False
        )
        
        # Test connection
        info = client.info()
        print(f"Connected to {info['version']['distribution']} {info['version']['number']}")
        return client
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def fetch_cpe_data(client, index_pattern="wazuh-alerts-*", size=1000):
    """
    Fetch unique CPEs from vulnerability alerts.
    Aggregates data.vulnerability.package.cpe field.
    """
    print(f"Searching for CPEs in {index_pattern}...")
    
    # Aggregation query to get unique CPEs
    query = {
        "size": 0,  # We don't need the documents, just the aggregation
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "data.vulnerability.package.cpe"}}
                ],
                "filter": [
                    {"range": {"timestamp": {"gte": "now-7d/d"}}}  # Last 7 days by default
                ]
            }
        },
        "aggs": {
            "unique_cpes": {
                "terms": {
                    "field": "data.vulnerability.package.cpe",
                    "size": size
                }
            }
        }
    }
    
    try:
        response = client.search(
            body=query,
            index=index_pattern
        )
        
        buckets = response['aggregations']['unique_cpes']['buckets']
        print(f"Found {len(buckets)} unique CPEs (Last 7 days)")
        
        results = []
        for bucket in buckets:
            cpe = bucket['key']
            count = bucket['doc_count']
            results.append({"cpe": cpe, "count": count})
            
        return results
        
    except Exception as e:
        print(f"Error executing search: {e}")
        return []

def save_results(results):
    """Save results to file."""
    if not results:
        print("No results to save.")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output_cpe")
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f"cpe_list_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {filename}")
    
    # Also save as simple text list
    txt_filename = output_dir / f"cpe_list_{timestamp}.txt"
    with open(txt_filename, 'w') as f:
        for item in results:
            f.write(f"{item['cpe']}\n")
            
    print(f"Text list saved to {txt_filename}")

def main():
    config = load_config("config.yaml")
    
    client = connect_to_indexer(config)
    if not client:
        return
        
    # Get index pattern from config or default
    indexer_config = config.get('indexer', {})
    index_pattern = indexer_config.get('cpe_index_pattern', 'wazuh-alerts-*')
    limit = indexer_config.get('cpe_search_limit', 1000)
    
    results = fetch_cpe_data(client, index_pattern, limit)
    
    if results:
        print("===== Top 10 CPEs Found =====")
        for item in results[:10]:
            print(f"- {item['cpe']} ({item['count']} occurrences)")
        
        save_results(results)

if __name__ == "__main__":
    main()
