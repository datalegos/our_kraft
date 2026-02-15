"""
ScanEvent Manager
Manages ScanEvent creation, event ID generation, and event tracking in Neo4j.
Implements ORBIT Event Management Architecture for audit traceability.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from .neo4j_connection import Neo4jConnection
import logging

logger = logging.getLogger(__name__)


class ScanEventManager:
    """Manages ScanEvent nodes and event ID generation"""
    
    def __init__(self, neo4j_conn: Neo4jConnection, graph_config_path: str = "config/graph_config.yaml"):
        """Initialize ScanEvent manager"""
        self.neo4j_conn = neo4j_conn
        self.graph_config = self._load_graph_config(graph_config_path)
        self.scanevent_config = self._get_scanevent_config()
        
        # Event ID generation settings
        self.org = "NODE"  # Organization: NODE or CORE
        self.layer_id = "N1"  # Layer ID: N1, N2, etc. (can be configured)
        self.event_type = "SCAN"  # Event type: SCAN, INGEST, ACTION, etc.
        self.sequence_number = self._get_next_sequence_number()
    
    def _load_graph_config(self, config_path: str) -> Dict[str, Any]:
        """Load graph configuration from YAML file"""
        try:
            config_file = Path(config_path)
            if not config_file.is_absolute():
                if not config_file.exists():
                    project_root = Path(__file__).parent.parent
                    config_file = project_root / config_path
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading graph config: {e}")
            raise
    
    def _get_scanevent_config(self) -> Dict[str, Any]:
        """Get ScanEvent node configuration from graph_config"""
        nodes_config = self.graph_config.get('nodes', {})
        return nodes_config.get('scanevent', {})
    
    def _get_next_sequence_number(self) -> int:
        """Get the next sequence number for event ID generation"""
        # Query Neo4j to find the highest sequence number for this layer
        query = """
        MATCH (e:ScanEvent)
        WHERE e.source_layer = $source_layer AND e.source_id = $source_id
        RETURN e.event_id AS event_id
        ORDER BY e.scanevent_time DESC
        LIMIT 1
        """
        
        try:
            with self.neo4j_conn.get_session() as session:
                result = session.run(
                    query,
                    source_layer=self.org,
                    source_id=self.layer_id
                )
                record = result.single()
                
                if record:
                    # Extract sequence number from event_id
                    event_id = record['event_id']
                    # Format: NODE_N1_SCAN_00001_1739162000
                    parts = event_id.split('_')
                    if len(parts) >= 4:
                        try:
                            seq_str = parts[3]
                            return int(seq_str) + 1
                        except ValueError:
                            pass
                
                # If no previous events, start from 1
                return 1
        except Exception as e:
            logger.warning(f"Could not get next sequence number, starting from 1: {e}")
            return 1
    
    def generate_event_id(self, epoch: Optional[int] = None) -> str:
        """
        Generate event ID following format: <ORG>_<LAYER>_<TYPE>_<SEQ>_<EPOCH>
        
        Args:
            epoch: Unix timestamp (optional, defaults to current time)
        
        Returns:
            Event ID string (e.g., "NODE_N1_SCAN_00001_1739162000")
        """
        if epoch is None:
            epoch = int(datetime.now().timestamp())
        
        seq_str = str(self.sequence_number).zfill(5)  # Zero-padded to 5 digits
        
        event_id = f"{self.org}_{self.layer_id}_{self.event_type}_{seq_str}_{epoch}"
        return event_id
    
    def create_scanevent(
        self,
        description: str = "STANDARD_BASELINE_LOAD",
        job_start_time: Optional[datetime] = None,
        status: str = "Created",
        metadata_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a ScanEvent node in Neo4j
        
        Args:
            description: Event description (e.g., "STANDARD_BASELINE_LOAD", "WAZUH_ASSET_INSERTION")
            job_start_time: Job start time (defaults to current time)
            status: Event status (Created, Broadcasted, ActionCardUpdated)
            metadata_json: Optional JSON metadata string
        
        Returns:
            Dictionary with event_id and created event data
        """
        if job_start_time is None:
            job_start_time = datetime.now()
        
        # Generate event ID
        epoch = int(job_start_time.timestamp())
        event_id = self.generate_event_id(epoch)
        
        # Prepare ScanEvent node properties
        scanevent_node = {
            'event_id': event_id,
            'event_type': self.event_type,
            'source_layer': self.org,
            'source_id': self.layer_id,
            'description': description,
            'scanevent_time': job_start_time.isoformat(),
            'status': status
        }
        
        if metadata_json:
            scanevent_node['metadata_json'] = metadata_json
        
        # Insert ScanEvent node into Neo4j
        label = self.scanevent_config.get('label', 'ScanEvent')
        properties_str = ', '.join([f"n.{k} = ${k}" for k in scanevent_node.keys()])
        
        query = f"""
        MERGE (n:{label} {{event_id: $event_id}})
        SET {properties_str}
        RETURN n
        """
        
        try:
            with self.neo4j_conn.get_session() as session:
                result = session.run(query, **scanevent_node)
                record = result.single()
                
                if record:
                    logger.info(f"Created ScanEvent: {event_id}")
                    logger.info(f"  Description: {description}")
                    logger.info(f"  Time: {job_start_time.isoformat()}")
                    logger.info(f"  Status: {status}")
                    
                    # Increment sequence for next event
                    self.sequence_number += 1
                    
                    return {
                        'event_id': event_id,
                        'scanevent': dict(record['n']),
                        'success': True
                    }
                else:
                    raise Exception("Failed to create ScanEvent node")
        except Exception as e:
            logger.error(f"Error creating ScanEvent: {e}")
            raise
    
    def get_latest_scanevent(self, source_layer: Optional[str] = None, source_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest ScanEvent for a given source layer and ID
        
        Args:
            source_layer: Source layer (NODE or CORE), defaults to self.org
            source_id: Source ID (e.g., N1, C1), defaults to self.layer_id
        
        Returns:
            Dictionary with latest ScanEvent data or None
        """
        if source_layer is None:
            source_layer = self.org
        if source_id is None:
            source_id = self.layer_id
        
        label = self.scanevent_config.get('label', 'ScanEvent')
        query = f"""
        MATCH (e:{label})
        WHERE e.source_layer = $source_layer AND e.source_id = $source_id
        RETURN e
        ORDER BY e.scanevent_time DESC
        LIMIT 1
        """
        
        try:
            with self.neo4j_conn.get_session() as session:
                result = session.run(query, source_layer=source_layer, source_id=source_id)
                record = result.single()
                
                if record:
                    return dict(record['e'])
                return None
        except Exception as e:
            logger.error(f"Error getting latest ScanEvent: {e}")
            return None

