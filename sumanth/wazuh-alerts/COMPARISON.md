# Script Comparison: orbit_package_alert_watcher.py vs package_alert_watcher.py

## Summary

The **package_alert_watcher.py** is the UPDATED and IMPROVED version. It should be used going forward.

## Key Improvements in package_alert_watcher.py

### 1. Correct Field Mapping for Windows Agents ✅

**OLD (orbit_package_alert_watcher.py):**
```python
package_field = data_field.get("package", {})  # ❌ Wrong for Windows
```

**NEW (package_alert_watcher.py):**
```python
program = data.get("program", {})  # ✅ Correct for Windows agents
```

**Why this matters:**
- Windows agents send package data in `data.program` field
- Linux agents send package data in `data.package` field
- The old script only worked for Linux, leaving Windows package details empty
- The new script correctly uses `data.program` for Windows

### 2. Better Field Names

**OLD:** Uses generic "package_*" field names
```json
{
  "package_name": "",
  "package_version": "",
  "package_vendor": ""
}
```

**NEW:** Uses accurate "program_*" field names matching Windows reality
```json
{
  "program_name": "7-Zip 26.00 (x64)",
  "program_version": "26.00",
  "program_vendor": "Igor Pavlov",
  "program_location": "C:\\Program Files\\7-Zip\\",
  "program_install_time": "2026/02/23 17:44:03"
}
```

### 3. Additional Useful Fields

The new script captures more details:
- `program_location` - Installation path
- `program_install_time` - When it was installed
- Better descriptions and metadata

### 4. Configuration System

Both scripts now use the same `config.py` system:
- No hardcoded credentials
- Easy to update settings
- Validates required fields on startup

## Evidence from Your Real Alert

From your actual Wazuh alert (captured in logs):

```json
{
  "data": {
    "type": "dbsync_packages",
    "operation_type": "INSERTED",
    "program": {                    ← Windows uses "program" NOT "package"
      "name": "7-Zip 26.00 (x64)",
      "version": "26.00",
      "vendor": "Igor Pavlov",
      "architecture": "x86_64",
      "format": "win",
      "install_time": "2026/02/23 17:44:03",
      "location": "C:\\Program Files\\7-Zip\\"
    }
  }
}
```

## Test Results

### Old Script Output (orbit_package_alert_watcher.py)
```json
{
  "package_name": "",           ← Empty!
  "package_version": "",        ← Empty!
  "package_vendor": "",         ← Empty!
  "action": "INSTALLED"
}
```

### New Script Output (package_alert_watcher.py)
```json
{
  "program_name": "7-Zip 26.00 (x64)",      ← Populated!
  "program_version": "26.00",                ← Populated!
  "program_vendor": "Igor Pavlov",           ← Populated!
  "program_location": "C:\\Program Files\\7-Zip\\",
  "action": "INSTALLED"
}
```

## Recommendation

✅ **USE: package_alert_watcher.py** (the new one)
❌ **RETIRE: orbit_package_alert_watcher.py** (the old one)

## Current Status

Both scripts now:
- Load configuration from `config.py`
- Use your existing Wazuh server: 172.27.122.220
- Poll every 30 seconds
- Write to `orbit_package_changes.json`
- Track state in `orbit_last_check.txt`

The new script is running and working correctly. It's currently polling and will capture package details properly when new installations/updates/removals occur on your Windows agent.
