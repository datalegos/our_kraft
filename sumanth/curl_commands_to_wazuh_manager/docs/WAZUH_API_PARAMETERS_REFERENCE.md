# Wazuh API v4.14.2 - GET Endpoints Parameters Reference

## Common Parameters (Available for most endpoints)

### **Pagination & Sorting**
- `limit` (integer): Maximum number of items to return (default: 500, max: 100000)
- `offset` (integer): First item to return (default: 0)
- `sort` (string): Sort criteria. Format: `field` or `+field` (ascending) or `-field` (descending)

### **Filtering & Search**
- `search` (string): Search term to look for in the response
- `select` (string): Select specific fields to return (comma-separated)
- `q` (string): Query using Wazuh Query Language (WQL)
- `pretty` (boolean): Pretty print JSON response (true/false)

### **Date Filtering**
- `older_than` (string): Filter items older than specified time (e.g., "7d", "1h", "30m")
- `newer_than` (string): Filter items newer than specified time

---

## Endpoint-Specific Parameters

### **1. Root API Info (`/`)**
```
Parameters: pretty
```

### **2. Manager Endpoints (`/manager/*`)**

#### `/manager/info`
```
Parameters: pretty, select
```

#### `/manager/configuration`
```
Parameters: pretty, section, field, raw
- section: Configuration section to retrieve
- field: Specific field within section
- raw: Return raw configuration (true/false)
```

#### `/manager/status`
```
Parameters: pretty
```

#### `/manager/stats` | `/manager/stats/hourly` | `/manager/stats/weekly`
```
Parameters: pretty, date
- date: Specific date for stats (YYYY-MM-DD format)
```

### **3. Agents Endpoints (`/agents/*`)**

#### `/agents`
```
Parameters: 
- pretty, limit, offset, sort, search, select, q
- status: Agent status (active, pending, never_connected, disconnected)
- os.platform: OS platform filter
- os.name: OS name filter  
- os.version: OS version filter
- manager: Manager name filter
- version: Agent version filter
- group: Group name filter
- node_name: Node name filter
- name: Agent name filter
- ip: Agent IP filter
- registerIP: Registration IP filter
- older_than, newer_than: Date filters
```

#### `/agents/summary` | `/agents/summary/status`
```
Parameters: pretty
```

### **4. Groups Endpoint (`/groups`)**
```
Parameters:
- pretty, limit, offset, sort, search, q
- hash: Group hash filter
```

### **5. Rules Endpoint (`/rules`)**
```
Parameters:
- pretty, limit, offset, sort, search, select, q
- rule_ids: Specific rule IDs (comma-separated)
- filename: Rules filename filter
- relative_dirname: Relative directory filter
- status: Rule status (enabled, disabled)
- group: Rule group filter
- level: Rule level filter (0-16)
- pci_dss: PCI DSS requirement filter
- gpg13: GPG13 requirement filter
- gdpr: GDPR requirement filter
- hipaa: HIPAA requirement filter
- nist_800_53: NIST 800-53 requirement filter
- tsc: TSC requirement filter
- mitre: MITRE ATT&CK technique filter
```

### **6. Decoders Endpoint (`/decoders`)**
```
Parameters:
- pretty, limit, offset, sort, search, select, q
- decoder_names: Specific decoder names (comma-separated)
- filename: Decoder filename filter
- relative_dirname: Relative directory filter
- status: Decoder status (enabled, disabled)
```

### **7. MITRE Endpoints (`/mitre/*`)**

#### `/mitre/techniques`
```
Parameters:
- pretty, limit, offset, sort, search, select, q
- technique_ids: Specific technique IDs
- tactic: Tactic filter
- platform: Platform filter
```

#### `/mitre/tactics`
```
Parameters:
- pretty, limit, offset, sort, search, select, q
- tactic_ids: Specific tactic IDs
```

### **8. Cluster Endpoint (`/cluster/status`)**
```
Parameters: pretty
```

### **9. Security Endpoints (`/security/*`)**

#### `/security/config`
```
Parameters: pretty
```

#### `/security/users`
```
Parameters:
- pretty, limit, offset, sort, search, q
- user_ids: Specific user IDs (comma-separated)
```

#### `/security/roles`
```
Parameters:
- pretty, limit, offset, sort, search, q
- role_ids: Specific role IDs (comma-separated)
```

#### `/security/policies`
```
Parameters:
- pretty, limit, offset, sort, search, q
- policy_ids: Specific policy IDs (comma-separated)
```

---

## Parameter Examples

### **Basic Usage**
```
?pretty=true&limit=10&offset=0
```

### **Filtering Agents**
```
?status=active&os.platform=linux&limit=20&sort=-lastKeepAlive
```

### **Searching Rules**
```
?search=ssh&level=5&group=authentication&pretty=true
```

### **Complex Query (WQL)**
```
?q=level>5;group=web&select=id,description,level&sort=+level
```

### **Date Filtering**
```
?newer_than=7d&older_than=1d&pretty=true
```

---

## Wazuh Query Language (WQL) Examples

### **Operators**
- `=` : Equal
- `!=` : Not equal
- `>` : Greater than
- `<` : Less than
- `~` : Like (regex)

### **Logical Operators**
- `;` : AND
- `,` : OR
- `!` : NOT

### **Examples**
```
q=level>5;group=web                    # Level > 5 AND group = web
q=status=active,status=pending         # Status = active OR pending  
q=name~ubuntu;!group=default           # Name contains ubuntu AND NOT in default group
q=level>3;(group=ssh,group=web)        # Level > 3 AND (group=ssh OR group=web)
```

---

## Response Format

All endpoints return JSON in this format:
```json
{
  "data": {
    "affected_items": [...],
    "total_affected_items": 0,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "Success message",
  "error": 0
}
```

## Notes

1. **Boolean Parameters**: Use `true`/`false` (lowercase)
2. **Date Formats**: Use ISO 8601 format or relative time (7d, 1h, 30m)
3. **Multiple Values**: Use comma-separated values for arrays
4. **Case Sensitivity**: Parameter names and values are case-sensitive
5. **URL Encoding**: Encode special characters in parameter values