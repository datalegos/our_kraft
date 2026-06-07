# ORBIT - Package Change Alert Watcher

Monitors Wazuh Indexer for package installation/update/removal alerts and logs them to JSON.

## Prerequisites

1. **Wazuh 4.14.2** (or compatible) running in Docker
2. **Custom rules** (100401-100403) added to Wazuh Manager
3. **Python 3.7+** installed
4. **Network access** to Wazuh Indexer (port 9200)

## Quick Start

### 1. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 2. Create Configuration File

Copy the example config and edit it with your Wazuh server details:

```cmd
copy config.example.py config.py
```

Then edit `config.py` with your actual values:

```python
INDEXER_HOST     = "192.168.1.100"          # Your Wazuh server IP
INDEXER_PORT     = 9200                      # Default port
INDEXER_USER     = "admin"                   # Default username
INDEXER_PASSWORD = "YourActualPassword"      # Your indexer password
```

The script will automatically load all settings from `config.py`.

### 3. Run the Watcher

```cmd
python orbit_package_alert_watcher.py
```

The script will:
- Poll the Wazuh Indexer every 30 seconds
- Look for package change alerts (rules 100401, 100402, 100403)
- Write events to `orbit_package_changes.json`
- Track the last check time in `orbit_last_check.txt`

### 4. Stop the Watcher

Press `CTRL+C` to stop gracefully.

## Configuration Options

All settings are defined in `config.py`:

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `INDEXER_HOST` | Yes | - | Your Wazuh server IP address |
| `INDEXER_PASSWORD` | Yes | - | Indexer password |
| `INDEXER_PORT` | No | 9200 | Wazuh Indexer port |
| `INDEXER_USER` | No | admin | Indexer username |
| `POLL_INTERVAL_SECONDS` | No | 30 | Seconds between checks |
| `OUTPUT_FILE` | No | orbit_package_changes.json | Output file path |
| `RULE_IDS_TO_WATCH` | No | ["100401", "100402", "100403"] | Rule IDs to monitor |
| `WAZUH_ALERTS_INDEX` | No | wazuh-alerts-* | Index pattern to search |

## Rule IDs

- **100401**: Package installed
- **100402**: Package updated
- **100403**: Package removed

## Output Format

Each event in `orbit_package_changes.json` contains:

```json
{
  "event_timestamp": "2026-02-23T10:12:00Z",
  "agent_id": "001",
  "agent_name": "web-server",
  "agent_ip": "192.168.1.50",
  "action": "INSTALLED",
  "package_name": "curl",
  "package_version": "7.81.0",
  "package_vendor": "Ubuntu",
  "package_arch": "amd64",
  "package_format": "deb",
  "package_size": 392,
  "rule_id": "100401",
  "rule_level": 7
}
```

## Troubleshooting

### Missing Configuration

```
ERROR: config.py not found!
```

**Solution:**
```cmd
copy config.example.py config.py
```
Then edit `config.py` with your Wazuh server details.

### Connection Errors

```
✗ Cannot connect to Wazuh Indexer
```

**Solutions:**
- Verify Wazuh Docker container is running: `docker ps`
- Check port 9200 is accessible: `telnet YOUR_IP 9200`
- Verify firewall rules allow connections
- Confirm INDEXER_HOST in `config.py` is correct

### Authentication Errors

```
✗ Indexer query failed! HTTP 401
```

**Solutions:**
- Verify INDEXER_USER and INDEXER_PASSWORD in `config.py` are correct
- Check Wazuh Indexer credentials in your Docker setup

### No Alerts Found

```
→ No new package change alerts found.
```

**Possible causes:**
- Custom rules not added to Wazuh Manager
- No package changes on monitored agents
- Timestamp filter too restrictive (delete `orbit_last_check.txt` to reset)

## Files Generated

- `orbit_package_changes.json` - All detected package events
- `orbit_last_check.txt` - Last poll timestamp (for state tracking)

## Notes

- The script uses the **Wazuh Indexer API** (port 9200), not the Wazuh Server API (port 55000)
- SSL verification is disabled for self-signed certificates
- First run looks back 5 minutes to catch recent events
- Events are appended to the JSON file (it grows over time)

## Version

- Wazuh: 4.14.2 (Docker)
- Python: 3.7+
