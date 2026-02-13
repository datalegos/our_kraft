# Wazuh 4.14.2 Endpoints Reference

This document describes all endpoints used for data collection, their parameters, and the information retrieved from each endpoint.

## ⚠️ Important Notice

**This documentation is based on the actual implementation in this codebase and may not reflect all parameters available in the official Wazuh API documentation.**

- **Wazuh Version:** 4.14.2 (deployed via Docker as per [Wazuh Docker deployment guide](https://documentation.wazuh.com/current/deployment-options/docker/wazuh-container.html))
- **Deployment Method:** Docker (single-node or multi-node stack)
- **Config Note:** The `config.yaml` file shows version 4.12.2, but the actual running Wazuh instance is 4.14.2
- **Documentation Status:** Based on implementation, not fully verified against official Wazuh 4.14.2 API documentation
- **Recommendation:** Please verify all endpoints and parameters against the [official Wazuh 4.14.2 API documentation](https://documentation.wazuh.com/current/user-manual/api/index.html)
- **Parameters Listed:** Only parameters actually used in the implementation are documented here. Additional parameters may be available in the official API.
- **Official Documentation:** For complete API reference, see [Wazuh API Reference](https://documentation.wazuh.com/current/user-manual/api/reference.html)

## Overview

The data collection system uses **both Wazuh Manager REST API and Wazuh Indexer** (OpenSearch API):
- **Wazuh Manager REST API** (Port 55000): Used for agents, host/OS, packages, hardware, groups, and FIM data
- **Wazuh Indexer** (Port 9200): Used for vulnerabilities data

**Why Dual Approach?**
- Manager API provides real-time agent and syscollector data
- Indexer provides vulnerability data from dedicated vulnerability index
- Each source is used for its optimal data type

---

## Authentication

### Wazuh Manager REST API
- **Method:** JWT Token Authentication
- **Endpoint:** `POST /security/user/authenticate`
- **Authentication:** HTTP Basic Auth (username/password)
- **Response:** JWT token in `data.token`
- **Usage:** Token must be included in `Authorization: Bearer {token}` header for all subsequent requests

### Wazuh Indexer
- **Method:** HTTP Basic Authentication
- **Credentials:** Username and password in HTTP Basic Auth header
- **Usage:** Included in all requests automatically

---

## Wazuh Manager REST API Endpoints

**Base URL:** `https://{host}:55000`  
**Protocol:** HTTPS (configurable)  
**Authentication:** JWT Token (Bearer token in Authorization header)

### 1. Authentication Endpoint

**Endpoint:** `POST /security/user/authenticate`  
**Purpose:** Authenticate and obtain JWT token

**Request:**
- **Method:** POST
- **Headers:** 
  - `Content-Type: application/json`
- **Authentication:** HTTP Basic Auth (username/password)
- **Body:** None

**Response:**
```json
{
  "data": {
    "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

**Usage:** Token must be used in `Authorization: Bearer {token}` header for all Manager API requests.

---

### 2. Agents Collection

**Endpoint:** `GET /agents`  
**Purpose:** Retrieve all registered agents (assets)

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pretty` | string | No | `true` | Format JSON response (true/false) |
| `limit` | integer | No | 1000 | Maximum number of results to return |
| `offset` | integer | No | 0 | Number of results to skip (for pagination) |
| `sort` | string | No | `id` | Sort field (e.g., "id", "-id" for descending) |
| `select` | string | No | - | Comma-separated list of fields to return (not used in current implementation) |
| `q` | string | No | - | Query string for filtering |
| `status` | string | No | - | Filter by agent status (active, disconnected, etc.) |
| `group` | string | No | - | Filter by group name |

**Example Request:**
```
GET /agents?pretty=true&limit=1000&sort=id&offset=0
```

**Response Data:**
```json
{
  "data": {
    "affected_items": [
      {
        "id": "000",
        "name": "agent-name",
        "ip": "192.168.1.100",
        "status": "active",
        "os": {
          "platform": "linux",
          "version": "Ubuntu 20.04"
        },
        "version": "4.12.2",
        "dateAdd": "2024-01-10T08:00:00Z"
      }
    ],
    "total_affected_items": 5,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "All agents information was returned",
  "error": 0
}
```

**Graph Model Mapping:**
- Node: `Asset/WAgent` (CrownJewel label)
- Properties: `asset_id`, `asset_name`

---

### 3. Host/OS Information

**Endpoint:** `GET /syscollector/{agent_id}/os`  
**Purpose:** Retrieve operating system details from syscollector

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Agent ID (e.g., "000", "001") |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pretty` | string | No | `true` | Format JSON response (true/false) |

**Note:** This endpoint does NOT accept `limit` parameter.

**Example Request:**
```
GET /syscollector/000/os?pretty=true
```

**Response Data:**
```json
{
  "data": {
    "affected_items": [
      {
        "os_name": "Ubuntu",
        "os_version": "20.04.5 LTS",
        "os_major": "20",
        "os_minor": "04",
        "os_codename": "focal",
        "os_platform": "ubuntu",
        "os_uname": "Linux",
        "os_arch": "x86_64",
        "hostname": "hostname",
        "scan_time": "2024-01-15T10:30:00Z"
      }
    ],
    "total_affected_items": 1,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "Syscollector information was returned",
  "error": 0
}
```

**Graph Model Mapping:**
- Node: `HOST`
- Properties: OS name, version, platform, architecture, hostname

---

### 4. Packages Collection

**Endpoint:** `GET /syscollector/{agent_id}/packages`  
**Purpose:** Retrieve installed packages/software from syscollector

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Agent ID (e.g., "000", "001") |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pretty` | string | No | `true` | Format JSON response (true/false) |
| `limit` | integer | No | 1000 | Maximum number of results to return |
| `offset` | integer | No | 0 | Number of results to skip (for pagination) |
| `sort` | string | No | - | Sort field |
| `q` | string | No | - | Query string for filtering |

**Example Request:**
```
GET /syscollector/000/packages?pretty=true&limit=1000
```

**Response Data:**
```json
{
  "data": {
    "affected_items": [
      {
        "name": "package-name",
        "version": "1.2.3",
        "architecture": "amd64",
        "format": "deb",
        "vendor": "vendor-name",
        "description": "Package description",
        "install_time": "2024-01-10T08:00:00Z",
        "location": "/path/to/package"
      }
    ],
    "total_affected_items": 500,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "Syscollector information was returned",
  "error": 0
}
```

**Graph Model Mapping:**
- Node: `Software/Package`
- Properties: Package name, version, architecture, vendor, install time

---

### 5. Hardware Information

**Endpoint:** `GET /syscollector/{agent_id}/hardware`  
**Purpose:** Retrieve hardware details from syscollector

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Agent ID (e.g., "000", "001") |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pretty` | string | No | `true` | Format JSON response (true/false) |

**Note:** This endpoint does NOT accept `limit` parameter.

**Example Request:**
```
GET /syscollector/000/hardware?pretty=true
```

**Response Data:**
```json
{
  "data": {
    "affected_items": [
      {
        "board_serial": "ABC123",
        "cpu_name": "Intel Core i7",
        "cpu_cores": 8,
        "cpu_mhz": 2400,
        "ram_total": 16384,
        "ram_free": 8192,
        "ram_usage": 50,
        "scan_time": "2024-01-15T10:30:00Z"
      }
    ],
    "total_affected_items": 1,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "Syscollector information was returned",
  "error": 0
}
```

**Graph Model Mapping:**
- Node: `Hardware`
- Properties: CPU details, RAM, board serial, scan time

---

### 6. Groups Collection

**Endpoint:** `GET /groups`  
**Purpose:** Retrieve all agent groups

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pretty` | string | No | `true` | Format JSON response (true/false) |
| `limit` | integer | No | 1000 | Maximum number of results to return |
| `offset` | integer | No | 0 | Number of results to skip (for pagination) |
| `sort` | string | No | - | Sort field |
| `q` | string | No | - | Query string for filtering |

**Example Request:**
```
GET /groups?pretty=true&limit=1000
```

**Response Data:**
```json
{
  "data": {
    "affected_items": [
      {
        "name": "default",
        "configuration": {
          "disabled": false
        },
        "count": 5
      }
    ],
    "total_affected_items": 1,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "All groups information was returned",
  "error": 0
}
```

**Graph Model Mapping:**
- Node: `AssetGroup`
- Properties: Group name, agent count

---

### 7. File Integrity Monitoring (FIM)

**Endpoint:** `GET /syscheck/{agent_id}`  
**Purpose:** Retrieve FIM (File Integrity Monitoring) data

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Agent ID (e.g., "000", "001") |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pretty` | string | No | `true` | Format JSON response (true/false) |
| `limit` | integer | No | 5000 | Maximum number of results to return |
| `offset` | integer | No | 0 | Number of results to skip (for pagination) |
| `sort` | string | No | - | Sort field |
| `q` | string | No | - | Query string for filtering |
| `file` | string | No | - | Filter by file path |
| `type` | string | No | - | Filter by file type (file, registry) |

**Example Request:**
```
GET /syscheck/000?pretty=true&limit=5000
```

**Response Data:**
```json
{
  "data": {
    "affected_items": [
      {
        "file": "/etc/filebeat/wazuh-template.json",
        "type": "file",
        "size": 84275,
        "perm": "rw-------",
        "uid": "0",
        "gid": "0",
        "user_name": "root",
        "group_name": "root",
        "md5": "db12ab2b4db38f907016b776c6808aeb",
        "sha1": "206dd802dbcc29e4abe695a96d3e58f0074256d5",
        "sha256": "c6e30822c67c10f7e777cb51926e261d8b2c3a941c4ffcf83325f700c1c8802f",
        "mtime": "2026-01-29T09:21:28+00:00",
        "inode": 2392172,
        "date": "2026-01-29T09:21:28+00:00",
        "changes": 1
      }
    ],
    "total_affected_items": 200,
    "total_failed_items": 0,
    "failed_items": []
  },
  "message": "Syscheck information was returned",
  "error": 0
}
```

**Graph Model Mapping:**
- Node: `File`
- Properties: File path, hash values, permissions, modification time

---

## Wazuh Indexer Endpoints (OpenSearch API)

**Base URL:** `https://{host}:9200`  
**Protocol:** HTTPS (configurable)  
**Authentication:** HTTP Basic Auth  
**Content-Type:** `application/json`

All endpoints use the OpenSearch Query DSL format (JSON in request body).

### Indices Used:
- `wazuh-alerts-4.x-*` - Security alerts, FIM events
- `wazuh-monitoring-*` - Agent monitoring data, groups
- `wazuh-states-*` - Syscollector snapshots (OS, packages, hardware)
- `wazuh-states-vulnerabilities-*` - Vulnerability data (dedicated index)

---

### 8. Vulnerabilities Collection

**Endpoint:** `POST /wazuh-states-vulnerabilities-*/_search`  
**Purpose:** Retrieve vulnerability data from Wazuh Indexer

**Request Body Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `size` | integer | No | 1000 | Maximum number of results to return |
| `from` | integer | No | 0 | Number of results to skip (for pagination) |
| `query` | object | Yes | - | OpenSearch Query DSL object |
| `_source` | array/boolean | No | true | Fields to include in response (array of field names or true for all) |
| `sort` | array | No | - | Sort criteria (e.g., `[{"@timestamp": {"order": "desc"}}]`) |

**Query DSL Structure:**
```json
{
  "query": {
    "bool": {
      "must": [
        {"match_all": {}},
        {"term": {"agent.id": "000"}}  // Optional: filter by agent
      ]
    }
  }
}
```

**Query Parameters:**
- `match_all`: Match all documents
- `term`: Exact match filter (e.g., `{"term": {"agent.id": "000"}}`)
- `bool.must`: All conditions must match
- `bool.should`: At least one condition must match
- `bool.must_not`: Conditions must not match

**Example Request:**
```json
{
  "size": 1000,
  "query": {
    "bool": {
      "must": [
        {"match_all": {}},
        {"term": {"agent.id": "000"}}
      ]
    }
  },
  "_source": [
    "agent.id",
    "agent.name",
    "vulnerability.id",
    "vulnerability.severity",
    "vulnerability.title",
    "vulnerability.published",
    "vulnerability.status",
    "vulnerability.description",
    "vulnerability.cvss",
    "package.name",
    "package.version",
    "@timestamp"
  ]
}
```

**Response Data:**
```json
{
  "hits": {
    "total": {
      "value": 150,
      "relation": "eq"
    },
    "hits": [
      {
        "_index": "wazuh-states-vulnerabilities-4.x-2024.01.15",
        "_source": {
          "agent": {
            "id": "000",
            "name": "agent-name"
          },
          "vulnerability": {
            "id": "CVE-2024-1234",
            "severity": "High",
            "title": "Vulnerability Title",
            "description": "Vulnerability description",
            "published": "2024-01-10T00:00:00Z",
            "status": "Active",
            "cvss": {
              "score": 7.5,
              "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
            }
          },
          "package": {
            "name": "package-name",
            "version": "1.2.3"
          },
          "@timestamp": "2024-01-15T10:30:00Z"
        }
      }
    ]
  }
}
```

**Important Notes:**
- Uses dedicated `wazuh-states-vulnerabilities-*` index (NOT `wazuh-alerts-*`)
- CVE ID is stored in `vulnerability.id` (not `vulnerability.cve`)
- Timestamp field is `@timestamp` (not `timestamp`)
- Package data is at top level (not nested in `data.vulnerability`)

**Graph Model Mapping:**
- Node: `Vulnerability`
- Properties: CVE ID, severity, score, package, version, published date

---

### 9. Index Discovery (Helper Endpoint)

**Endpoint:** `GET /_cat/indices/{pattern}-*?format=json&s=index:desc`  
**Purpose:** Find the latest index matching a pattern

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pattern` | string | Yes | Index pattern (e.g., "wazuh-alerts-4.x", "wazuh-states-vulnerabilities") |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `format` | string | No | `json` | Response format (json, yaml, text) |
| `s` | string | No | - | Sort criteria (e.g., "index:desc" for descending) |

**Example Request:**
```
GET /_cat/indices/wazuh-states-vulnerabilities-*?format=json&s=index:desc
```

**Response Data:**
```json
[
  {
    "index": "wazuh-states-vulnerabilities-4.x-2024.01.15",
    "health": "green",
    "status": "open",
    "uuid": "abc123...",
    "pri": "1",
    "rep": "1",
    "docs.count": "1500",
    "store.size": "2.5mb"
  }
]
```

---

## Data Collection Summary

| Collector | Source | Endpoint | Key Parameters |
|-----------|--------|----------|----------------|
| Agents | Manager | `GET /agents` | `limit`, `offset`, `sort`, `pretty` |
| Host/OS | Manager | `GET /syscollector/{agent_id}/os` | `pretty` (no `limit`) |
| Packages | Manager | `GET /syscollector/{agent_id}/packages` | `limit`, `offset`, `pretty` |
| Hardware | Manager | `GET /syscollector/{agent_id}/hardware` | `pretty` (no `limit`) |
| Groups | Manager | `GET /groups` | `limit`, `offset`, `pretty` |
| FIM | Manager | `GET /syscheck/{agent_id}` | `limit`, `offset`, `pretty` |
| Vulnerabilities | Indexer | `POST /wazuh-states-vulnerabilities-*/_search` | `size`, `from`, `query`, `_source` |

---

## Common Query Parameters (Manager API)

### Pagination
- `limit`: Maximum number of results (default: varies by endpoint)
- `offset`: Number of results to skip (default: 0)

### Formatting
- `pretty`: Format JSON response (`true`/`false`, default: `true`)

### Filtering
- `q`: Query string for filtering (varies by endpoint)
- `sort`: Sort field (e.g., `id`, `-id` for descending)

### Common Filter Parameters (varies by endpoint)
- `status`: Filter by status
- `group`: Filter by group name
- `file`: Filter by file path (FIM)
- `type`: Filter by type

---

## Common Query DSL Parameters (Indexer)

### Pagination
- `size`: Maximum number of results (default: 10)
- `from`: Number of results to skip (default: 0)

### Field Selection
- `_source`: Array of field names to include, or `true` for all fields, or `false` to exclude all

### Sorting
- `sort`: Array of sort criteria
  ```json
  "sort": [
    {"@timestamp": {"order": "desc"}},
    {"_score": {"order": "asc"}}
  ]
  ```

### Query Types
- `match_all`: Match all documents
- `term`: Exact match
- `match`: Full-text search
- `bool`: Boolean query (must, should, must_not, filter)
- `range`: Range queries
- `exists`: Check if field exists
- `wildcard`: Pattern matching

---

## Notes

1. **Wazuh Version:** 4.14.2 (deployed via Docker)
2. **Deployment:** Docker container deployment following [Wazuh Docker deployment guide](https://documentation.wazuh.com/current/deployment-options/docker/wazuh-container.html)
3. **Config Note:** The `config.yaml` file shows version 4.12.2, but the actual running Wazuh instance is 4.14.2
4. **Documentation Source:** This document is based on the actual implementation in the codebase. For complete and authoritative parameter documentation, refer to the [official Wazuh 4.14.2 API documentation](https://documentation.wazuh.com/current/user-manual/api/index.html)
5. **Dual Connection:** Uses both Manager REST API and Indexer
4. **Authentication:** 
   - Manager: JWT token (obtained via `/security/user/authenticate`)
   - Indexer: HTTP Basic Auth
5. **Pagination:** 
   - Manager: Use `limit` and `offset` query parameters
   - Indexer: Use `size` and `from` in request body
6. **Indices:** Four main index patterns:
   - `wazuh-alerts-*` - Security events, alerts, FIM
   - `wazuh-monitoring-*` - Agent monitoring and status data
   - `wazuh-states-*` - Syscollector snapshots (OS, packages, hardware)
   - `wazuh-states-vulnerabilities-*` - Vulnerability data (dedicated index)
7. **Query Format:** 
   - Manager: REST API with query parameters
   - Indexer: OpenSearch Query DSL (JSON in request body)
8. **Error Handling:** All endpoints return error information in response if request fails
9. **Parameter Completeness:** The parameters listed are those used in the current implementation. The official Wazuh API may support additional parameters not documented here. Please refer to official documentation for complete parameter lists.

---

## Graph Model Relationships

Based on the knowledge graph model, the following relationships are established:

- `Asset/WAgent` → `RUNS_SOFTWARE` → `Software/Package`
- `Asset/WAgent` → `LOCATED_IN` → `AssetGroup`
- `Asset/WAgent` → `HAS_RISK` → `RiskScore`
- `Asset/WAgent` → `HAS_SENSITIVITY` → `SeverityProfile`
- `Asset/WAgent` → (relationship) → `ScanEvent`
- `Asset/WAgent` → (relationship) → `HOST`
- `Asset/WAgent` → (relationship) → `Hardware`
- `Asset/WAgent` → (relationship) → `File`
- `Software/Package` → `HAS_VULNERABILITY` → `Vulnerability`
- `AssetGroup` → `HAS_VULNERABILITY` → `Vulnerability`
- `HOST` → (relationship) → `Vulnerability`
- `Hardware` → (relationship) → `Vulnerability`
- `File` → (relationship) → `Vulnerability`
- `ScanEvent` → `PRODUCED` → `SeverityProfile`
