# ORBIT.Node Knowledge Graph Schema
## Production-Ready Local Knowledge Graph for Member Security Operations

---

## Document Overview

This document defines the **ORBIT.Node Knowledge Graph** - a self-contained, plug-and-play graph database that models the complete security posture of a single member institution.

**Scope**: ORBIT.Node only (local member environment)
**Data Source**: Wazuh Manager APIs and telemetry
**Integration**: Native Wazuh UI for all operations
**Privacy**: All sensitive data remains within member perimeter

---

## Design Intent

The Node Knowledge Graph serves as the **Local Reality** - a comprehensive, queryable model of:
- Asset inventory (hosts, software, hardware, network)
- Security findings (vulnerabilities, compliance, file integrity, rootcheck)
- Sensitivity classification (Crown Jewels via Presidio scanning)
- Action card correlation and execution state
- Temporal changes for drift detection

**Key Characteristics**:
- **Plug-and-Play**: Pre-packaged schema frozen to specific Wazuh version
- **Incremental Updates**: Delta-based updates, no full rebuilds
- **Wazuh-Native**: All observability through Wazuh Manager UI
- **Privacy-Preserving**: No data leaves Node without anonymization
- **Audit-Ready**: Complete traceability for all state changes

---

## Assumptions

### Technical Assumptions
1. **Wazuh Version**: Fixed to v4.14.2+ (schema frozen per ORBIT.Node release)
2. **Data Source**: 100% Wazuh APIs - no external dependencies
3. **Graph Database**: Embedded lightweight graph (Neo4j, ArangoDB, or similar)
4. **Storage**: Local to member environment, no cloud dependencies
5. **Presidio**: Local deployment for PII/PCI detection

### Operational Assumptions
1. **Wazuh Authority**: Wazuh Manager is the single UI and control plane
2. **No Custom UI**: All ORBIT operations visible through Wazuh dashboards
3. **Immutable Audit**: All state changes are append-only logged
4. **Human Execution**: All defensive actions require analyst approval
5. **Snapshot-Based**: Temporal snapshots enable drift detection

---

## ORBIT.Node Use Cases Supported

This schema directly supports:
- **N1**: Local Site Profile & Knowledge Graph Creation
- **N2**: Privacy Scanning & Crown Jewel Classification
- **N3**: Subscription Generation & Optimization
- **N4**: Action Proposal Injection into Wazuh
- **N5**: Human-in-the-Loop Execution via Active Response
- **N6**: Pending Action Assignment (Ticket Workflow)
- **N7**: Differential Profile Updates
- **N8**: Completion Feedback & Status Synchronization

---

## Node Entities (Node Types)

### Infrastructure & Asset Entities

### 1. WazuhManager

**Purpose**: Represents the local Wazuh Manager instance

**Key Properties**:
- `manager_id` (string, unique): Wazuh Manager UUID
- `name` (string): Manager hostname
- `version` (string): Wazuh version (e.g., "v4.14.2")
- `type` (string): Deployment type ("server", "worker", "master")
- `installation_path` (string): Installation directory
- `max_agents` (string): Agent capacity
- `openssl_support` (boolean): OpenSSL availability
- `node_name` (string): Cluster node name (if clustered)
- `created_at` (timestamp): Entity creation time
- `updated_at` (timestamp): Last update time

**Maps to Wazuh Data**: `/manager/info` endpoint
**ORBIT Use Case**: N1 - Local Site Profile

---

### 2. Agent

**Purpose**: Represents a monitored endpoint (server, workstation, container)

**Key Properties**:
- `agent_id` (string, unique): Wazuh agent ID (e.g., "001", "002")
- `name` (string): Agent hostname
- `ip` (string): IP address
- `register_ip` (string): Registration IP
- `status` (string): Connection status ("active", "disconnected", "never_connected", "pending")
- `status_code` (integer): Numeric status (0=active, 3=disconnected)
- `version` (string): Wazuh agent version
- `os_name` (string): Operating system name
- `os_platform` (string): OS platform ("windows", "linux", "darwin", "amzn")
- `os_version` (string): OS version
- `os_arch` (string): Architecture ("x86_64", "arm64")
- `os_codename` (string, nullable): OS codename
- `os_major` (string): OS major version
- `os_minor` (string, nullable): OS minor version
- `os_build` (string, nullable): OS build number
- `date_add` (timestamp): Agent registration date
- `last_keep_alive` (timestamp): Last heartbeat
- `disconnection_time` (timestamp, nullable): When agent disconnected
- `node_name` (string): Cluster node managing this agent
- `config_sum` (string): Configuration checksum
- `merged_sum` (string): Merged configuration checksum
- `group_config_status` (string): Sync status ("synced", "not_synced")
- `created_at` (timestamp): Entity creation time
- `updated_at` (timestamp): Last update time

**Maps to Wazuh Data**: `/agents` endpoint
**ORBIT Use Case**: N1 - Local Site Profile

---

### 3. AgentGroup

**Purpose**: Logical grouping of agents for configuration and policy management

**Key Properties**:
- `group_id` (string, unique): Generated from group name
- `name` (string, unique): Group name (e.g., "default", "webservers", "databases")
- `agent_count` (integer): Number of agents in group
- `config_sum` (string): Group configuration checksum
- `merged_sum` (string): Merged configuration checksum
- `created_at` (timestamp): Entity creation time
- `updated_at` (timestamp): Last update time

**Maps to Wazuh Data**: `/groups` endpoint
**ORBIT Use Case**: N1 - Local Site Profile

---

### 4. DataSnapshot

**Purpose**: Point-in-time capture of the environment for drift detection

**Key Properties**:
- `snapshot_id` (string, unique): Timestamp-based ID (e.g., "20260206_152708")
- `collection_time` (timestamp): When snapshot was collected
- `status` (string): Collection status ("complete", "partial", "failed")
- `agent_count` (integer): Number of agents captured
- `endpoint_count` (integer): Number of API endpoints queried
- `change_summary` (json): Summary of changes from previous snapshot
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: Output folder structure
**ORBIT Use Case**: N7 - Differential Profile Updates

---

### Hardware & Network Entities

### 5. HardwareProfile

**Purpose**: Physical/virtual hardware characteristics of an agent

**Key Properties**:
- `profile_id` (string, unique): agent_id + scan_id
- `cpu_name` (string): CPU model
- `cpu_cores` (integer): Number of cores
- `cpu_mhz` (integer): CPU frequency
- `ram_total_kb` (integer): Total RAM in KB
- `ram_free_kb` (integer): Free RAM in KB
- `ram_usage_percent` (integer): RAM usage percentage
- `board_serial` (string, nullable): Motherboard serial
- `scan_id` (integer): Syscollector scan ID
- `scan_time` (timestamp): When hardware was scanned
- `is_current` (boolean): Whether this is the latest profile
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscollector/{agent_id}/hardware`
**ORBIT Use Case**: N1 - Local Site Profile

---

### 6. NetworkInterface

**Purpose**: Network interface on an agent

**Key Properties**:
- `interface_id` (string, unique): agent_id + interface_name
- `name` (string): Interface name (e.g., "eth0", "Wi-Fi")
- `adapter` (string, nullable): Adapter description
- `type` (string): Interface type ("ethernet", "wireless", "loopback")
- `state` (string): Interface state ("up", "down")
- `mac` (string): MAC address
- `mtu` (integer): Maximum transmission unit
- `tx_packets` (integer): Transmitted packets
- `rx_packets` (integer): Received packets
- `tx_bytes` (integer): Transmitted bytes
- `rx_bytes` (integer): Received bytes
- `tx_errors` (integer): Transmission errors
- `rx_errors` (integer): Reception errors
- `tx_dropped` (integer): Dropped transmissions
- `rx_dropped` (integer): Dropped receptions
- `scan_time` (timestamp): When interface was scanned
- `is_active` (boolean): Whether interface is currently active
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscollector/{agent_id}/netif`
**ORBIT Use Case**: N1 - Local Site Profile

---

### 7. NetworkAddress

**Purpose**: IP address assigned to a network interface

**Key Properties**:
- `address_id` (string, unique): agent_id + interface + address
- `address` (string): IP address
- `netmask` (string): Network mask
- `broadcast` (string, nullable): Broadcast address
- `proto` (string): Protocol ("ipv4", "ipv6")
- `scan_time` (timestamp): When address was scanned
- `is_primary` (boolean): Whether this is the primary address
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscollector/{agent_id}/netaddr`
**ORBIT Use Case**: N1 - Local Site Profile

---

### 8. OpenPort

**Purpose**: Open network port on an agent

**Key Properties**:
- `port_id` (string, unique): agent_id + protocol + local_ip + local_port
- `protocol` (string): Protocol ("tcp", "udp", "tcp6", "udp6")
- `local_ip` (string): Local IP address
- `local_port` (integer): Local port number
- `remote_ip` (string, nullable): Remote IP address
- `remote_port` (integer, nullable): Remote port number
- `state` (string): Connection state ("listening", "established", "time_wait", "close_wait")
- `pid` (integer, nullable): Process ID
- `process_name` (string, nullable): Process name
- `scan_time` (timestamp): When port was scanned
- `is_listening` (boolean): Whether port is in listening state
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscollector/{agent_id}/ports`
**ORBIT Use Case**: N1 - Local Site Profile, Attack Surface Analysis

---

### 9. Process

**Purpose**: Running process on an agent

**Key Properties**:
- `process_id` (string, unique): agent_id + pid + scan_time
- `pid` (integer): Process ID
- `name` (string): Process name
- `state` (string): Process state ("running", "sleeping", "stopped", "zombie")
- `ppid` (integer): Parent process ID
- `egroup` (string): Effective group
- `euser` (string): Effective user
- `fgroup` (string): File system group
- `priority` (integer): Process priority
- `nice` (integer): Nice value
- `size` (integer): Memory size
- `vm_size` (integer): Virtual memory size
- `resident` (integer): Resident memory
- `share` (integer): Shared memory
- `start_time` (timestamp): Process start time
- `pgrp` (integer): Process group
- `session` (integer): Session ID
- `nlwp` (integer): Number of threads
- `tgid` (integer): Thread group ID
- `tty` (integer): Terminal
- `processor` (integer): CPU number
- `cmd` (string): Command line
- `argvs` (string, nullable): Arguments
- `scan_time` (timestamp): When process was scanned
- `is_running` (boolean): Whether process is currently running
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscollector/{agent_id}/processes`
**ORBIT Use Case**: N1 - Local Site Profile

---

### 10. Package

**Purpose**: Installed software package on an agent

**Key Properties**:
- `package_id` (string, unique): agent_id + name + version + architecture
- `name` (string): Package name
- `version` (string): Package version
- `architecture` (string): Package architecture
- `format` (string): Package format ("deb", "rpm", "msi", "pkg", "exe")
- `vendor` (string, nullable): Package vendor
- `install_time` (timestamp, nullable): Installation timestamp
- `description` (string, nullable): Package description
- `size` (integer, nullable): Package size
- `priority` (string, nullable): Package priority
- `section` (string, nullable): Package section/category
- `location` (string, nullable): Installation location
- `scan_time` (timestamp): When package was scanned
- `is_current` (boolean): Whether package is currently installed
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscollector/{agent_id}/packages`
**ORBIT Use Case**: N1 - Local Site Profile, N3 - Subscription Generation

---

### 11. Hotfix

**Purpose**: Windows hotfix/patch installed on an agent

**Key Properties**:
- `hotfix_id` (string, unique): agent_id + hotfix_code
- `hotfix` (string): Hotfix identifier (e.g., "KB5012345")
- `scan_time` (timestamp): When hotfix was scanned
- `is_current` (boolean): Whether hotfix is currently installed
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscollector/{agent_id}/hotfixes`
**ORBIT Use Case**: N1 - Local Site Profile

---

### Security Finding Entities

### 12. FileIntegrityEvent

**Purpose**: File or registry change detected by FIM

**Key Properties**:
- `event_id` (string, unique): agent_id + file_path + date + hash
- `file_path` (string): Full file or registry path
- `type` (string): Event type ("file", "registry_key", "registry_value")
- `event_type` (string): Change type ("added", "modified", "deleted")
- `date` (timestamp): When change was detected
- `changes` (integer): Number of changes
- `size` (integer, nullable): File size
- `permissions` (string, nullable): File permissions
- `uid` (string, nullable): User ID
- `gid` (string, nullable): Group ID
- `user_name` (string, nullable): User name
- `group_name` (string, nullable): Group name
- `inode` (integer, nullable): Inode number
- `md5` (string, nullable): MD5 hash
- `sha1` (string, nullable): SHA1 hash
- `sha256` (string, nullable): SHA256 hash
- `arch` (string, nullable): Architecture for registry ("[x32]", "[x64]")
- `value_name` (string, nullable): Registry value name
- `value_type` (string, nullable): Registry value type
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/syscheck/{agent_id}`
**ORBIT Use Case**: N1 - Local Site Profile, Security Monitoring

---

### 13. SCAPolicy

**Purpose**: Security Configuration Assessment policy (e.g., CIS Benchmark)

**Key Properties**:
- `policy_id` (string, unique): Policy identifier (e.g., "cis_win11_enterprise")
- `name` (string): Policy name
- `description` (text): Policy description
- `references` (string): Reference URLs
- `total_checks` (integer): Total number of checks in policy
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/sca/{agent_id}` policy metadata
**ORBIT Use Case**: N1 - Local Site Profile, Compliance Monitoring

---

### 14. SCAScan

**Purpose**: Single execution of an SCA policy on an agent

**Key Properties**:
- `scan_id` (string, unique): agent_id + policy_id + start_scan
- `start_scan` (timestamp): Scan start time
- `end_scan` (timestamp): Scan end time
- `score` (integer): Compliance score percentage
- `pass_count` (integer): Number of passed checks
- `fail_count` (integer): Number of failed checks
- `invalid_count` (integer): Number of invalid checks
- `hash_file` (string): Policy file hash
- `is_latest` (boolean): Whether this is the most recent scan
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/sca/{agent_id}` scan results
**ORBIT Use Case**: N1 - Local Site Profile, Compliance Monitoring

---

### 15. RootcheckFinding

**Purpose**: Rootkit or system anomaly detected by rootcheck

**Key Properties**:
- `finding_id` (string, unique): agent_id + event + date_first
- `event` (string): Finding description
- `status` (string): Finding status ("outstanding", "solved")
- `old_day` (timestamp): First detection date
- `date_first` (timestamp): First occurrence
- `date_last` (timestamp): Last occurrence
- `is_resolved` (boolean): Whether finding is resolved
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: `/rootcheck/{agent_id}`
**ORBIT Use Case**: N1 - Local Site Profile, Security Monitoring

---

### ORBIT-Specific Entities

### 16. SensitivityClassification

**Purpose**: Privacy/sensitivity assessment of an asset (Crown Jewel classification)

**Key Properties**:
- `classification_id` (string, unique): Generated ID
- `asset_type` (string): Type of asset ("agent", "database", "directory", "file")
- `asset_identifier` (string): Reference to asset (agent_id, db_name, path)
- `sensitivity_score` (float): Computed sensitivity score (0.0-1.0)
- `is_crown_jewel` (boolean): Whether asset is classified as Crown Jewel
- `pii_types_detected` (array): Types of PII found (["ssn", "credit_card", "email"])
- `pci_detected` (boolean): Whether PCI data was detected
- `phi_detected` (boolean): Whether PHI data was detected
- `detection_confidence` (float): Presidio confidence score
- `scan_method` (string): How scanned ("database_sample", "file_sample", "metadata")
- `sample_size` (integer): Number of samples analyzed
- `scanned_at` (timestamp): When sensitivity scan occurred
- `classification_rationale` (text): Why this classification was assigned
- `created_at` (timestamp): Entity creation time
- `updated_at` (timestamp): Last update time

**Maps to Wazuh Data**: ORBIT-generated, injected into Wazuh as custom events
**ORBIT Use Case**: N2 - Privacy Scanning & Crown Jewel Classification

---

### 17. TechnologySubscription

**Purpose**: Node's declared interest in specific technologies for Core intelligence

**Key Properties**:
- `subscription_id` (string, unique): Generated ID
- `technology_name` (string): Technology name (e.g., "Apache HTTP Server")
- `technology_category` (string): Category ("os", "web_server", "database", "middleware")
- `cpe` (string, nullable): Common Platform Enumeration
- `has_crown_jewels` (boolean): Whether Crown Jewels use this technology
- `asset_count` (integer): Number of local assets with this technology
- `subscription_status` (string): Status ("active", "pending", "deprecated")
- `subscribed_at` (timestamp): When subscription was created
- `last_updated` (timestamp): Last subscription update
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: Derived from Package and Agent inventory
**ORBIT Use Case**: N3 - Subscription Generation & Optimization

---

### 18. ActionCard

**Purpose**: Defensive recommendation received from ORBIT.Core

**Key Properties**:
- `action_card_id` (string, unique): Core-provided action card ID
- `threat_id` (string): Associated threat (CVE, Campaign, TTP)
- `threat_name` (string): Threat name
- `threat_severity` (string): Severity level
- `affected_technology` (string): Target technology
- `recommended_action` (string): Action type ("isolate", "patch", "block", "investigate", "monitor")
- `action_details` (text): Detailed action guidance
- `threat_score` (float): Core-computed threat score
- `priority_rank` (integer): Core-assigned priority
- `digital_signature` (string): Core Guardian signature
- `received_at` (timestamp): When Node received the card
- `state` (string): Node-side state ("received", "correlated", "injected", "acknowledged", "executed", "completed", "rejected")
- `created_at` (timestamp): Entity creation time
- `updated_at` (timestamp): Last state update

**Maps to Wazuh Data**: Received from Core, injected into Wazuh as alerts
**ORBIT Use Case**: N4 - Action Proposal Injection into Wazuh

---

### 19. ActionCorrelation

**Purpose**: Links Action Cards to specific local assets

**Key Properties**:
- `correlation_id` (string, unique): Generated ID
- `action_card_id` (string): Related action card
- `affected_agent_ids` (array): List of affected agent IDs
- `crown_jewel_involved` (boolean): Whether Crown Jewels are affected
- `urgency_level` (string): Node-computed urgency ("critical", "high", "medium", "low")
- `wazuh_alert_id` (string, nullable): Injected Wazuh alert ID
- `wazuh_severity` (integer): Wazuh alert severity level
- `correlation_rationale` (text): Why these assets are affected
- `correlated_at` (timestamp): When correlation was performed
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: ORBIT-generated correlation logic
**ORBIT Use Case**: N4 - Action Proposal Injection into Wazuh

---

### 20. PendingAction

**Purpose**: Investigation or multi-step action requiring analyst assignment

**Key Properties**:
- `pending_action_id` (string, unique): Generated ID
- `action_card_id` (string, nullable): Related action card (if applicable)
- `wazuh_alert_id` (string): Associated Wazuh alert ID
- `title` (string): Action title
- `description` (text): Investigation guidance
- `complexity_reason` (string): Why this requires investigation
- `affected_agent_ids` (array): Impacted agents
- `crown_jewel_involved` (boolean): Crown Jewel involvement
- `assigned_to` (string, nullable): Analyst or team assigned
- `status` (string): Status ("open", "acknowledged", "in_progress", "resolved", "false_positive", "escalated")
- `created_at` (timestamp): When pending action was created
- `acknowledged_at` (timestamp, nullable): When acknowledged
- `resolved_at` (timestamp, nullable): When resolved
- `resolution_notes` (text, nullable): Analyst notes
- `updated_at` (timestamp): Last update time

**Maps to Wazuh Data**: Injected as Wazuh alerts, state tracked via Wazuh lifecycle
**ORBIT Use Case**: N6 - Pending Action Assignment (Ticket Workflow)

---

### 21. ExecutionRecord

**Purpose**: Record of defensive action execution attempt

**Key Properties**:
- `execution_id` (string, unique): Generated ID
- `action_card_id` (string): Related action card
- `pending_action_id` (string, nullable): Related pending action (if applicable)
- `agent_id` (string): Target agent
- `action_type` (string): Action executed ("isolate", "patch", "block", "terminate_process")
- `execution_method` (string): How executed ("active_response", "manual", "script")
- `active_response_script` (string, nullable): Script name if Active Response
- `initiated_by` (string): Analyst identifier
- `authorization_method` (string): How authorized ("wazuh_ui", "cli", "api")
- `outcome` (string): Result ("success", "failure", "partial", "timeout", "error")
- `outcome_details` (text): Detailed outcome description
- `error_message` (text, nullable): Error details if failed
- `execution_duration_seconds` (integer, nullable): How long execution took
- `initiated_at` (timestamp): When execution started
- `completed_at` (timestamp, nullable): When execution completed
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: Wazuh Active Response logs
**ORBIT Use Case**: N5 - HITL Execution via Active Response, N8 - Completion Feedback

---

### 22. ProfileDelta

**Purpose**: Incremental change detected in local environment

**Key Properties**:
- `delta_id` (string, unique): Generated ID
- `delta_type` (string): Type of change ("agent_added", "agent_removed", "software_installed", "software_removed", "version_changed", "config_changed")
- `entity_type` (string): What changed ("agent", "package", "group", "configuration")
- `entity_identifier` (string): ID of changed entity
- `change_details` (json): Before/after details
- `detected_at` (timestamp): When change was detected
- `synchronized_to_core` (boolean): Whether delta was sent to Core
- `synchronized_at` (timestamp, nullable): When synchronized
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: Derived from Wazuh inventory change events
**ORBIT Use Case**: N7 - Differential Profile Updates

---

### 23. OutcomeFeedback

**Purpose**: Aggregated execution outcome for Core feedback

**Key Properties**:
- `feedback_id` (string, unique): Generated ID
- `action_card_id` (string): Related action card
- `action_type` (string): Action executed
- `outcome` (string): Result ("success", "failure", "partial", "ignored")
- `time_to_mitigation_minutes` (integer, nullable): Duration to complete
- `crown_jewel_involved` (boolean): Crown Jewel involvement
- `execution_count` (integer): Number of execution attempts
- `affected_asset_count` (integer): Number of assets affected (anonymized)
- `feedback_sent_to_core` (boolean): Whether sent to Core
- `sent_at` (timestamp, nullable): When sent to Core
- `created_at` (timestamp): Entity creation time

**Maps to Wazuh Data**: Aggregated from ExecutionRecord entities
**ORBIT Use Case**: N8 - Completion Feedback & Status Synchronization

---
## Node Relationships (Edges)

### Infrastructure Relationships

#### 1. MANAGES
- **From**: WazuhManager → Agent
- **Direction**: WazuhManager → Agent
- **Cardinality**: 1-to-N (one manager manages many agents)
- **Purpose**: Links agents to their Wazuh manager
- **Properties**:
  - `managed_since` (timestamp): When management started
  - `node_name` (string): Cluster node name
- **Why**: Establishes manager-agent hierarchy

---

#### 2. BELONGS_TO_GROUP
- **From**: Agent → AgentGroup
- **Direction**: Agent → AgentGroup
- **Cardinality**: N-to-N (agents can belong to multiple groups)
- **Purpose**: Links agents to configuration groups
- **Properties**:
  - `joined_at` (timestamp): When agent joined group
  - `config_status` (string): Configuration sync status
- **Why**: Groups determine policies and configurations

---

#### 3. CAPTURED_IN_SNAPSHOT
- **From**: Agent/Package/Process → DataSnapshot
- **Direction**: Entity → DataSnapshot
- **Cardinality**: N-to-N (entities appear in multiple snapshots)
- **Purpose**: Links entities to temporal snapshots
- **Properties**:
  - `captured_at` (timestamp): Capture time
  - `entity_state` (string): State at capture time
- **Why**: Enables temporal analysis and drift detection

---

### Hardware & Network Relationships

#### 4. HAS_HARDWARE
- **From**: Agent → HardwareProfile
- **Direction**: Agent → HardwareProfile
- **Cardinality**: 1-to-N (one agent has multiple profiles over time)
- **Purpose**: Links agents to hardware characteristics
- **Properties**:
  - `scanned_at` (timestamp): Scan time
  - `is_current` (boolean): Whether this is latest profile
- **Why**: Tracks hardware inventory and changes

---

#### 5. HAS_INTERFACE
- **From**: Agent → NetworkInterface
- **Direction**: Agent → NetworkInterface
- **Cardinality**: 1-to-N (one agent has multiple interfaces)
- **Purpose**: Links agents to network interfaces
- **Properties**:
  - `scanned_at` (timestamp): Scan time
  - `is_active` (boolean): Whether interface is active
- **Why**: Network topology and connectivity mapping

---

#### 6. HAS_ADDRESS
- **From**: NetworkInterface → NetworkAddress
- **Direction**: NetworkInterface → NetworkAddress
- **Cardinality**: 1-to-N (one interface has multiple addresses)
- **Purpose**: Links interfaces to IP addresses
- **Properties**:
  - `assigned_at` (timestamp): Assignment time
  - `is_primary` (boolean): Primary address flag
- **Why**: Maps IP addresses to interfaces

---

#### 7. HAS_OPEN_PORT
- **From**: Agent → OpenPort
- **Direction**: Agent → OpenPort
- **Cardinality**: 1-to-N (one agent has multiple ports)
- **Purpose**: Links agents to open network ports
- **Properties**:
  - `scanned_at` (timestamp): Scan time
  - `is_listening` (boolean): Listening state
- **Why**: Attack surface analysis

---

#### 8. RUNS_PROCESS
- **From**: Agent → Process
- **Direction**: Agent → Process
- **Cardinality**: 1-to-N (one agent runs multiple processes)
- **Purpose**: Links agents to running processes
- **Properties**:
  - `scanned_at` (timestamp): Scan time
  - `is_running` (boolean): Current running state
- **Why**: Process inventory and anomaly detection

---

#### 9. LISTENS_ON_PORT
- **From**: Process → OpenPort
- **Direction**: Process → OpenPort
- **Cardinality**: 1-to-N (one process can listen on multiple ports)
- **Purpose**: Links processes to ports they're listening on
- **Properties**:
  - `bound_at` (timestamp): Binding time
- **Why**: Identifies which processes expose network services

---

#### 10. HAS_PACKAGE
- **From**: Agent → Package
- **Direction**: Agent → Package
- **Cardinality**: 1-to-N (one agent has multiple packages)
- **Purpose**: Links agents to installed software
- **Properties**:
  - `installed_at` (timestamp): Installation time
  - `scanned_at` (timestamp): Scan time
  - `is_current` (boolean): Currently installed flag
- **Why**: Software inventory for vulnerability and compliance

---

#### 11. HAS_HOTFIX
- **From**: Agent → Hotfix
- **Direction**: Agent → Hotfix
- **Cardinality**: 1-to-N (one agent has multiple hotfixes)
- **Purpose**: Links Windows agents to installed patches
- **Properties**:
  - `scanned_at` (timestamp): Scan time
  - `is_current` (boolean): Currently installed flag
- **Why**: Patch status tracking

---

### Security Finding Relationships

#### 12. HAS_FIM_EVENT
- **From**: Agent → FileIntegrityEvent
- **Direction**: Agent → FileIntegrityEvent
- **Cardinality**: 1-to-N (one agent has multiple FIM events)
- **Purpose**: Links agents to file/registry changes
- **Properties**:
  - `detected_at` (timestamp): Detection time
  - `severity` (string): Event severity
- **Why**: Tracks unauthorized file system changes

---

#### 13. SCANNED_WITH_POLICY
- **From**: Agent → SCAScan
- **Direction**: Agent → SCAScan
- **Cardinality**: 1-to-N (one agent has multiple scans)
- **Purpose**: Links agents to SCA scan executions
- **Properties**:
  - `scanned_at` (timestamp): Scan time
  - `is_latest` (boolean): Latest scan flag
- **Why**: Compliance posture tracking

---

#### 14. USES_POLICY
- **From**: SCAScan → SCAPolicy
- **Direction**: SCAScan → SCAPolicy
- **Cardinality**: N-to-1 (many scans use one policy)
- **Purpose**: Links scans to policy/benchmark
- **Properties**:
  - `policy_version` (string): Policy version at scan time
- **Why**: Identifies compliance framework assessed

---

#### 15. HAS_ROOTCHECK_FINDING
- **From**: Agent → RootcheckFinding
- **Direction**: Agent → RootcheckFinding
- **Cardinality**: 1-to-N (one agent has multiple findings)
- **Purpose**: Links agents to rootkit detections
- **Properties**:
  - `detected_at` (timestamp): Detection time
  - `is_resolved` (boolean): Resolution status
- **Why**: Tracks potential compromises

---

### ORBIT-Specific Relationships

#### 16. HAS_SENSITIVITY_CLASSIFICATION
- **From**: Agent → SensitivityClassification
- **Direction**: Agent → SensitivityClassification
- **Cardinality**: 1-to-N (one agent has multiple classifications over time)
- **Purpose**: Links agents to Crown Jewel classifications
- **Properties**:
  - `classified_at` (timestamp): Classification time
  - `is_current` (boolean): Current classification flag
- **Why**: Identifies high-value assets for prioritization

---

#### 17. GENERATES_SUBSCRIPTION
- **From**: Package → TechnologySubscription
- **Direction**: Package → TechnologySubscription
- **Cardinality**: N-to-1 (many packages contribute to one subscription)
- **Purpose**: Links installed software to technology subscriptions
- **Properties**:
  - `contributed_at` (timestamp): When package contributed to subscription
- **Why**: Derives subscriptions from actual asset inventory

---

#### 18. TARGETS_AGENT
- **From**: ActionCard → Agent
- **Direction**: ActionCard → Agent
- **Cardinality**: N-to-N (action cards can target multiple agents)
- **Purpose**: Links action cards to affected agents
- **Properties**:
  - `relevance_score` (float): How relevant to this agent
  - `correlated_at` (timestamp): Correlation time
- **Why**: Enables agent-specific action card queries

---

#### 19. CORRELATED_AS
- **From**: ActionCard → ActionCorrelation
- **Direction**: ActionCard → ActionCorrelation
- **Cardinality**: 1-to-1 (one action card has one correlation)
- **Purpose**: Links action cards to local correlation analysis
- **Properties**:
  - `correlated_at` (timestamp): Correlation time
- **Why**: Captures Node-side relevance assessment

---

#### 20. INJECTED_AS_ALERT
- **From**: ActionCorrelation → PendingAction
- **Direction**: ActionCorrelation → PendingAction (or direct Wazuh alert)
- **Cardinality**: 1-to-1 (one correlation produces one alert/action)
- **Purpose**: Links correlations to Wazuh-visible actions
- **Properties**:
  - `injected_at` (timestamp): Injection time
  - `wazuh_alert_id` (string): Wazuh alert identifier
- **Why**: Tracks action card → Wazuh alert flow

---

#### 21. EXECUTED_AS
- **From**: ActionCard → ExecutionRecord
- **Direction**: ActionCard → ExecutionRecord
- **Cardinality**: 1-to-N (one action card can have multiple execution attempts)
- **Purpose**: Links action cards to execution attempts
- **Properties**:
  - `executed_at` (timestamp): Execution time
  - `execution_sequence` (integer): Order of attempts
- **Why**: Tracks execution history per action card

---

#### 22. EXECUTED_ON_AGENT
- **From**: ExecutionRecord → Agent
- **Direction**: ExecutionRecord → Agent
- **Cardinality**: N-to-1 (many executions on one agent)
- **Purpose**: Links executions to target agents
- **Properties**:
  - `executed_at` (timestamp): Execution time
- **Why**: Identifies which agents had actions executed

---

#### 23. PRODUCED_FEEDBACK
- **From**: ExecutionRecord → OutcomeFeedback
- **Direction**: ExecutionRecord → OutcomeFeedback
- **Cardinality**: N-to-1 (many executions contribute to one feedback)
- **Purpose**: Links executions to aggregated feedback
- **Properties**:
  - `aggregated_at` (timestamp): Aggregation time
- **Why**: Enables anonymized outcome reporting to Core

---

#### 24. DETECTED_CHANGE
- **From**: DataSnapshot → ProfileDelta
- **Direction**: DataSnapshot → ProfileDelta
- **Cardinality**: 1-to-N (one snapshot detects multiple deltas)
- **Purpose**: Links snapshots to detected changes
- **Properties**:
  - `detected_at` (timestamp): Detection time
- **Why**: Tracks what changed between snapshots

---

#### 25. AFFECTS_ENTITY
- **From**: ProfileDelta → Agent/Package/Process
- **Direction**: ProfileDelta → Changed Entity
- **Cardinality**: 1-to-1 (one delta affects one entity)
- **Purpose**: Links deltas to specific changed entities
- **Properties**:
  - `change_type` (string): Type of change
- **Why**: Identifies what specifically changed

---

## Constraints & Identity Rules

### Uniqueness Constraints

1. **WazuhManager**: `manager_id` must be unique
2. **Agent**: `agent_id` must be unique
3. **AgentGroup**: `name` must be unique
4. **DataSnapshot**: `snapshot_id` must be unique
5. **HardwareProfile**: `profile_id` (agent_id + scan_id) must be unique
6. **NetworkInterface**: `interface_id` (agent_id + interface_name) must be unique
7. **NetworkAddress**: `address_id` (agent_id + interface + address) must be unique
8. **OpenPort**: `port_id` (agent_id + protocol + local_ip + local_port) must be unique
9. **Process**: `process_id` (agent_id + pid + scan_time) must be unique
10. **Package**: `package_id` (agent_id + name + version + architecture) must be unique
11. **Hotfix**: `hotfix_id` (agent_id + hotfix_code) must be unique
12. **FileIntegrityEvent**: `event_id` (agent_id + file_path + date + hash) must be unique
13. **SCAPolicy**: `policy_id` must be unique
14. **SCAScan**: `scan_id` (agent_id + policy_id + start_scan) must be unique
15. **RootcheckFinding**: `finding_id` (agent_id + event + date_first) must be unique
16. **SensitivityClassification**: `classification_id` must be unique
17. **TechnologySubscription**: `subscription_id` must be unique
18. **ActionCard**: `action_card_id` must be unique
19. **ActionCorrelation**: `correlation_id` must be unique
20. **PendingAction**: `pending_action_id` must be unique
21. **ExecutionRecord**: `execution_id` must be unique
22. **ProfileDelta**: `delta_id` must be unique
23. **OutcomeFeedback**: `feedback_id` must be unique

---

### Data Integrity Rules

1. **No Orphan Agents**: Every Agent must have MANAGES relationship to WazuhManager
2. **Snapshot Consistency**: All entities in a snapshot must have CAPTURED_IN_SNAPSHOT relationship
3. **Temporal Ordering**: Timestamps must be logically ordered (start < end, created < updated)
4. **Status Consistency**: Agent status must match status_code (0=active, 3=disconnected)
5. **Current Flags**: Only one entity per agent should have `is_current=true` for time-series data
6. **Process-Port Binding**: If Process has LISTENS_ON_PORT, the OpenPort must exist
7. **Group Membership**: BELONGS_TO_GROUP relationships must be bidirectional with group agent_count
8. **Action Card State**: ActionCard state transitions must follow valid workflow
9. **Execution Outcomes**: ExecutionRecord outcome must be one of valid values
10. **Sensitivity Scores**: SensitivityClassification scores must be between 0.0 and 1.0

---

### Indexing Recommendations

For query performance, create indexes on:

1. **Agent**: `agent_id`, `status`, `os_platform`, `name`, `ip`
2. **AgentGroup**: `name`
3. **DataSnapshot**: `snapshot_id`, `collection_time`
4. **Package**: `name`, `version`
5. **OpenPort**: `local_port`, `protocol`, `state`
6. **Process**: `name`, `pid`
7. **FileIntegrityEvent**: `file_path`, `type`, `date`
8. **SCAScan**: `start_scan`, `score`
9. **SensitivityClassification**: `is_crown_jewel`, `sensitivity_score`
10. **ActionCard**: `action_card_id`, `state`, `received_at`
11. **ExecutionRecord**: `outcome`, `initiated_at`
12. **ProfileDelta**: `delta_type`, `detected_at`

---
## Graph Structure & Traversal Patterns

### Overall Graph Structure

The Node graph follows a **hub-and-spoke** architecture with Agent as the central hub:

```
WazuhManager (Root)
│
├── MANAGES → Agent (Hub)
│   │
│   ├── BELONGS_TO_GROUP → AgentGroup
│   │
│   ├── HAS_HARDWARE → HardwareProfile
│   │
│   ├── HAS_INTERFACE → NetworkInterface
│   │   └── HAS_ADDRESS → NetworkAddress
│   │
│   ├── HAS_OPEN_PORT → OpenPort
│   │   └── LISTENS_ON_PORT ← Process
│   │
│   ├── RUNS_PROCESS → Process
│   │
│   ├── HAS_PACKAGE → Package
│   │   └── GENERATES_SUBSCRIPTION → TechnologySubscription
│   │
│   ├── HAS_HOTFIX → Hotfix
│   │
│   ├── HAS_FIM_EVENT → FileIntegrityEvent
│   │
│   ├── SCANNED_WITH_POLICY → SCAScan
│   │   └── USES_POLICY → SCAPolicy
│   │
│   ├── HAS_ROOTCHECK_FINDING → RootcheckFinding
│   │
│   ├── HAS_SENSITIVITY_CLASSIFICATION → SensitivityClassification
│   │
│   ├── TARGETS_AGENT ← ActionCard
│   │   ├── CORRELATED_AS → ActionCorrelation
│   │   │   └── INJECTED_AS_ALERT → PendingAction
│   │   └── EXECUTED_AS → ExecutionRecord
│   │       ├── EXECUTED_ON_AGENT → Agent
│   │       └── PRODUCED_FEEDBACK → OutcomeFeedback
│   │
│   └── CAPTURED_IN_SNAPSHOT → DataSnapshot
│       └── DETECTED_CHANGE → ProfileDelta
│           └── AFFECTS_ENTITY → [Agent/Package/Process]
```

---

### Typical Query Paths

#### 1. Complete Agent Profile
**Path**: Agent → [All Related Entities]
**Purpose**: "Show me everything about agent 001"
**Traversal**:
```
START: Agent(001)
FOLLOW: HAS_HARDWARE, HAS_INTERFACE, HAS_PACKAGE, HAS_OPEN_PORT, 
        RUNS_PROCESS, HAS_FIM_EVENT, SCANNED_WITH_POLICY, 
        HAS_SENSITIVITY_CLASSIFICATION
RETURN: Complete agent profile
```
**Use Case**: N1 - Local Site Profile, Incident Investigation

---

#### 2. Crown Jewel Identification
**Path**: Agent → SensitivityClassification
**Purpose**: "Which agents are Crown Jewels?"
**Traversal**:
```
START: All Agents
FOLLOW: HAS_SENSITIVITY_CLASSIFICATION
FILTER: is_crown_jewel = true
RETURN: Crown Jewel agents with sensitivity scores
```
**Use Case**: N2 - Privacy Scanning & Crown Jewel Classification

---

#### 3. Technology Subscription Generation
**Path**: Agent → Package → TechnologySubscription
**Purpose**: "What technologies should we subscribe to?"
**Traversal**:
```
START: All Agents
FOLLOW: HAS_PACKAGE
GROUP BY: Package.name, Package.vendor
AGGREGATE: Count, Versions
GENERATE: TechnologySubscription entities
RETURN: Subscription list
```
**Use Case**: N3 - Subscription Generation & Optimization

---

#### 4. Action Card Correlation
**Path**: ActionCard → Agent → SensitivityClassification
**Purpose**: "Which agents are affected by this action card?"
**Traversal**:
```
START: ActionCard(card_id)
MATCH: affected_technology
FOLLOW: Agent → HAS_PACKAGE
FILTER: Package.name matches affected_technology
FOLLOW: Agent → HAS_SENSITIVITY_CLASSIFICATION
COMPUTE: Urgency based on Crown Jewel status
RETURN: Affected agents with urgency levels
```
**Use Case**: N4 - Action Proposal Injection into Wazuh

---

#### 5. Attack Surface Analysis
**Path**: Agent → OpenPort → Process
**Purpose**: "What network services are exposed?"
**Traversal**:
```
START: All Agents
FOLLOW: HAS_OPEN_PORT
FILTER: state = "listening" AND local_ip != "127.0.0.1"
FOLLOW: LISTENS_ON_PORT → Process
GROUP BY: local_port, process_name
RETURN: Exposed services with process details
```
**Use Case**: N1 - Local Site Profile, Security Assessment

---

#### 6. Compliance Status Overview
**Path**: Agent → SCAScan → SCAPolicy
**Purpose**: "What's our compliance posture?"
**Traversal**:
```
START: All Agents
FOLLOW: SCANNED_WITH_POLICY
FILTER: is_latest = true
FOLLOW: USES_POLICY → SCAPolicy
AGGREGATE: Average score, Pass/Fail counts by policy
RETURN: Compliance dashboard data
```
**Use Case**: N1 - Local Site Profile, Compliance Reporting

---

#### 7. Drift Detection
**Path**: DataSnapshot → ProfileDelta → Agent/Package
**Purpose**: "What changed between yesterday and today?"
**Traversal**:
```
START: DataSnapshot(today)
FOLLOW: DETECTED_CHANGE → ProfileDelta
FILTER: delta_type IN ["agent_added", "software_installed", "version_changed"]
FOLLOW: AFFECTS_ENTITY → [Agent/Package]
RETURN: Change timeline with details
```
**Use Case**: N7 - Differential Profile Updates

---

#### 8. Execution Outcome Analysis
**Path**: ActionCard → ExecutionRecord → Agent
**Purpose**: "How effective was this action card?"
**Traversal**:
```
START: ActionCard(card_id)
FOLLOW: EXECUTED_AS → ExecutionRecord
FOLLOW: EXECUTED_ON_AGENT → Agent
AGGREGATE: Success rate, Average time_to_mitigation
FOLLOW: PRODUCED_FEEDBACK → OutcomeFeedback
RETURN: Execution effectiveness metrics
```
**Use Case**: N8 - Completion Feedback & Status Synchronization

---

#### 9. Pending Investigation Queue
**Path**: PendingAction → ActionCard → Agent
**Purpose**: "What investigations are pending?"
**Traversal**:
```
START: All PendingActions
FILTER: status IN ["open", "acknowledged", "in_progress"]
FOLLOW: Related ActionCard
FOLLOW: TARGETS_AGENT → Agent
FOLLOW: Agent → HAS_SENSITIVITY_CLASSIFICATION
ORDER BY: crown_jewel_involved DESC, created_at ASC
RETURN: Prioritized investigation queue
```
**Use Case**: N6 - Pending Action Assignment

---

#### 10. Software Vulnerability Surface
**Path**: Agent → Package → [External CVE Matching]
**Purpose**: "Which agents have vulnerable software?"
**Traversal**:
```
START: All Agents
FOLLOW: HAS_PACKAGE
MATCH: Package against CVE database (external)
FILTER: Has known vulnerabilities
FOLLOW: Agent → HAS_SENSITIVITY_CLASSIFICATION
PRIORITIZE: Crown Jewels first
RETURN: Vulnerable agents with risk scores
```
**Use Case**: N1 - Local Site Profile, Vulnerability Management

---

## Example Use Cases (Conceptual)

### Use Case 1: Crown Jewel Impact Assessment
**Scenario**: Action card received for Apache vulnerability - which Crown Jewels are affected?

**Graph Traversal**:
1. Start at ActionCard where affected_technology = "Apache HTTP Server"
2. Follow TARGETS_AGENT to find agents with Apache
3. Follow HAS_SENSITIVITY_CLASSIFICATION on those agents
4. Filter where is_crown_jewel = true
5. Return Crown Jewel agents with sensitivity scores

**Value**: Immediate identification of high-value assets at risk

---

### Use Case 2: Incident Response Context
**Scenario**: Suspicious FIM event detected - need complete context

**Graph Traversal**:
1. Start at FileIntegrityEvent(event_id)
2. Follow HAS_FIM_EVENT (reverse) to Agent
3. Follow Agent → RUNS_PROCESS to see running processes
4. Follow Agent → HAS_OPEN_PORT to see network connections
5. Follow Agent → HAS_ROOTCHECK_FINDING to see other anomalies
6. Follow Agent → HAS_SENSITIVITY_CLASSIFICATION to assess impact

**Value**: Complete incident context for rapid response

---

### Use Case 3: Patch Compliance Verification
**Scenario**: Verify which Windows agents have critical hotfix installed

**Graph Traversal**:
1. Start at all Agents where os_platform = "windows"
2. Follow HAS_HOTFIX
3. Filter where hotfix = "KB5012345"
4. Identify agents WITHOUT this hotfix
5. Follow HAS_SENSITIVITY_CLASSIFICATION to prioritize

**Value**: Targeted patch deployment prioritization

---

### Use Case 4: Configuration Drift Alert
**Scenario**: Detect agents whose compliance scores decreased

**Graph Traversal**:
1. Start at DataSnapshot(today) and DataSnapshot(yesterday)
2. Follow CAPTURED_IN_SNAPSHOT to Agents
3. Follow Agent → SCANNED_WITH_POLICY for both snapshots
4. Compare SCAScan.score between snapshots
5. Filter where score decreased
6. Return agents with degraded compliance

**Value**: Proactive detection of security posture degradation

---

### Use Case 5: Technology Inventory Report
**Scenario**: Generate report of all database technologies in use

**Graph Traversal**:
1. Start at all Agents
2. Follow HAS_PACKAGE
3. Filter where Package.category = "database"
4. Group by Package.name, Package.version
5. Count agents per technology
6. Follow GENERATES_SUBSCRIPTION to see subscription status

**Value**: Technology portfolio visibility for subscription management

---

## Design Principles

### 1. Wazuh-First Integration
- **All data sourced from Wazuh**: No external dependencies for core inventory
- **Wazuh as UI**: All ORBIT operations visible through Wazuh Manager
- **Event-driven updates**: Graph updates triggered by Wazuh events
- **Native alert injection**: Action cards appear as Wazuh alerts

### 2. Privacy Preservation
- **Local containment**: All sensitive data stays within Node
- **Anonymized exports**: Only aggregated metrics sent to Core
- **No PII in subscriptions**: Technology descriptors only, no asset identifiers
- **Crown Jewel protection**: Sensitivity classifications never leave Node

### 3. Temporal Awareness
- **Snapshot-based**: All entities linked to temporal snapshots
- **Drift detection**: Compare snapshots to identify changes
- **Audit trail**: Complete history of state changes
- **Time-versioned**: Multiple versions of entities over time

### 4. Operational Simplicity
- **Plug-and-play**: Pre-packaged schema, no runtime configuration
- **Incremental updates**: Delta-based, no full rebuilds
- **Self-healing**: Automatic reconciliation with Wazuh state
- **Minimal maintenance**: Schema frozen to Wazuh version

### 5. Action-Oriented
- **Execution tracking**: Complete record of defensive actions
- **Outcome capture**: Success/failure metrics for learning
- **Human accountability**: All actions require analyst authorization
- **Feedback loop**: Outcomes fed back to Core for improvement

---

## Future Extensions

### 1. Vulnerability Management
**New Entities**:
- `Vulnerability`: CVE details with CVSS scores
- `VulnerabilityMatch`: Links packages to known CVEs

**New Relationships**:
- `Package → HAS_VULNERABILITY → Vulnerability`
- `Agent → AFFECTED_BY → Vulnerability` (derived)

**Value**: Automated vulnerability detection and prioritization

---

### 2. User Activity Tracking
**New Entities**:
- `User`: System user account
- `LoginSession`: User login session
- `UserActivity`: User actions and commands

**New Relationships**:
- `Agent → HAS_USER → User`
- `User → HAS_SESSION → LoginSession`
- `FileIntegrityEvent → CAUSED_BY → User`

**Value**: User behavior analysis and insider threat detection

---

### 3. Container & Cloud Integration
**New Entities**:
- `Container`: Docker/Kubernetes container
- `ContainerImage`: Container image
- `CloudResource`: Cloud VM/instance metadata

**New Relationships**:
- `Agent → RUNS_CONTAINER → Container`
- `Container → BASED_ON → ContainerImage`
- `Agent → DEPLOYED_ON → CloudResource`

**Value**: Cloud-native security monitoring

---

### 4. Network Flow Analysis
**New Entities**:
- `NetworkFlow`: Network connection flow record
- `NetworkSegment`: VLAN or subnet

**New Relationships**:
- `Agent → GENERATED_FLOW → NetworkFlow`
- `Agent → LOCATED_IN → NetworkSegment`
- `NetworkFlow → CROSSES_SEGMENT → NetworkSegment`

**Value**: Lateral movement detection and network segmentation analysis

---

### 5. Threat Intelligence Enrichment
**New Entities**:
- `ThreatIndicator`: IOC (IP, domain, hash)
- `ThreatCampaign`: Known attack campaign

**New Relationships**:
- `FileIntegrityEvent → MATCHES_IOC → ThreatIndicator`
- `OpenPort → CONNECTED_TO_IOC → ThreatIndicator`
- `ActionCard → ADDRESSES_CAMPAIGN → ThreatCampaign`

**Value**: Threat context enrichment for local findings

---

## Implementation Considerations

### Graph Database Selection

**Recommended Options**:
1. **Neo4j** (Embedded or Server)
   - Mature, excellent Cypher query language
   - Strong community and tooling
   - Good performance for deep traversals
   
2. **ArangoDB** (Embedded)
   - Multi-model (graph + document)
   - Lightweight, embeddable
   - Good for mixed workloads

3. **SQLite with Graph Extension**
   - Ultra-lightweight
   - No separate process
   - Good for smaller deployments

**Selection Criteria**:
- Embeddable within ORBIT.Node process
- Low resource footprint
- Query performance for 6+ hop traversals
- Support for temporal queries
- Backup and recovery capabilities

---

### Data Ingestion Strategy

**Phase 1: Initial Baseline (Day-0)**
1. Query all Wazuh APIs for complete inventory
2. Build full graph from scratch
3. Mark as baseline snapshot
4. Send aggregated subscription to Core

**Phase 2: Incremental Updates**
1. Subscribe to Wazuh event streams
2. Detect material changes (agent add/remove, software install/uninstall)
3. Apply delta mutations to graph
4. Create new snapshot if significant changes
5. Send deltas to Core

**Phase 3: Periodic Reconciliation**
1. Daily full reconciliation with Wazuh
2. Detect and correct any drift
3. Prune old snapshots per retention policy

---

### Performance Optimization

1. **Batch Writes**: Insert entities and relationships in batches
2. **Lazy Loading**: Load related entities only when queried
3. **Caching**: Cache frequently accessed nodes (Manager, Groups)
4. **Indexes**: Create indexes on all frequently queried properties
5. **Partitioning**: Partition by snapshot_id for temporal queries
6. **Materialized Views**: Pre-compute common aggregations

---

### Data Retention Policy

1. **Snapshots**: Retain last 90 days (configurable)
2. **Execution Records**: Retain 1 year for audit
3. **Action Cards**: Retain until resolved + 90 days
4. **Security Findings**: Retain until resolved + 30 days
5. **Audit Logs**: Retain per regulatory requirements (typically 7 years)

---

### Security & Access Control

1. **Graph-Level Security**: Read-only access for analysts, write access for ORBIT.Node only
2. **Encryption at Rest**: Encrypt graph database files
3. **Audit Logging**: Log all graph queries and mutations
4. **Wazuh Integration**: All access mediated through Wazuh RBAC

---

## Summary

This ORBIT.Node Knowledge Graph schema provides a **comprehensive, Wazuh-integrated foundation** for local security operations:

**Key Strengths**:
- **Complete Coverage**: All Wazuh telemetry modeled (agents, inventory, findings)
- **ORBIT-Ready**: Native support for Crown Jewels, Action Cards, and execution tracking
- **Privacy-Preserving**: Sensitive data never leaves Node
- **Temporal**: Snapshot-based design enables drift detection
- **Operational**: Wazuh-first design for analyst workflows
- **Extensible**: Modular design supports future enhancements

**Supported Workflows**:
- Asset inventory and discovery (N1)
- Crown Jewel classification (N2)
- Technology subscription generation (N3)
- Action card correlation and injection (N4)
- Human-authorized execution (N5)
- Investigation assignment (N6)
- Differential updates (N7)
- Outcome feedback (N8)

**Next Steps**:
1. Validate schema with ORBIT.Node engineering team
2. Select embedded graph database
3. Implement Wazuh API ingestion pipeline
4. Build graph query API layer
5. Integrate with Wazuh UI for visualization
6. Deploy Presidio integration for Crown Jewel classification

This schema is **production-ready** and aligned with the ORBIT High-Level Design document.
