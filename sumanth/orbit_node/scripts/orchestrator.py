#!/usr/bin/env python3
"""
DataLegos Pipeline Orchestrator
Sequential execution with validation gates and .done files

Follows DataLegos maintainability standards:
- Parameter-driven configuration
- Structured logging
- Error codes
- Validation at each step
"""

import sys
import json
import logging
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os

# Configure structured logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger('orchestrator')


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code}] {message}")


class PipelineOrchestrator:
    """
    Orchestrates sequential execution of DataLegos pipeline.
    
    Pipeline Steps:
    1. collect_data      - Collect from Wazuh API
    2. extract_data      - Normalize and extract
    3. build_node_graph  - Create Node KG in Neo4j
    4. aggregate_data    - Create privacy-preserving aggregations
    5. detect_pii        - Scan for PII/PCI
    6. build_core_graph  - Create Core Graph in Neo4j
    """
    
    def __init__(self, shared_data_path: Optional[Path] = None):
        """
        Initialize orchestrator.
        
        Args:
            shared_data_path: Path to shared data directory
        """
        self.shared_data_path = shared_data_path or Path(os.getenv('SHARED_DATA_PATH', '/shared_data'))
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / 'scripts'
        
        # Ensure shared data path exists
        self.shared_data_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file
        log_file = self.shared_data_path / 'logs' / 'pipeline.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%SZ'
        ))
        logger.addHandler(file_handler)
        
        logger.info("=" * 80)
        logger.info("DataLegos Pipeline Orchestrator Initialized")
        logger.info(f"Shared Data Path: {self.shared_data_path}")
        logger.info(f"Project Root: {self.project_root}")
        logger.info("=" * 80)
    
    def run_pipeline(self):
        """Run complete pipeline sequentially"""
        logger.info("Starting complete pipeline execution")
        
        steps = [
            ("collect_data", "Collect data from Wazuh API"),
            ("extract_data", "Extract and normalize data"),
            ("build_node_graph", "Build Node Knowledge Graph"),
            ("aggregate_data", "Create privacy-preserving aggregations"),
            ("detect_pii", "Scan for PII/PCI data"),
            ("build_core_graph", "Build Core Graph"),
        ]
        
        start_time = datetime.now()
        
        for step_name, description in steps:
            logger.info("-" * 80)
            logger.info(f"Step: {step_name}")
            logger.info(f"Description: {description}")
            logger.info("-" * 80)
            
            try:
                # Validate prerequisites
                self._validate_prerequisites(step_name)
                
                # Run step
                self._run_step(step_name)
                
                # Validate output
                self._validate_output(step_name)
                
                # Create .done file
                self._create_done_file(step_name)
                
                logger.info(f"✅ Step '{step_name}' completed successfully")
                
            except PipelineError as e:
                logger.error(f"❌ Step '{step_name}' failed: {e}")
                logger.error(f"Error Code: {e.error_code}")
                logger.error("Pipeline execution stopped")
                sys.exit(1)
            
            except Exception as e:
                logger.error(f"❌ Unexpected error in step '{step_name}': {e}", exc_info=True)
                logger.error("Pipeline execution stopped")
                sys.exit(1)
        
        # Create pipeline completion marker
        self._create_pipeline_done_file()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 80)
    
    def run_single_step(self, step_name: str):
        """Run a single pipeline step"""
        logger.info(f"Running single step: {step_name}")
        
        try:
            # Validate prerequisites
            self._validate_prerequisites(step_name)
            
            # Run step
            self._run_step(step_name)
            
            # Validate output
            self._validate_output(step_name)
            
            # Create .done file
            self._create_done_file(step_name)
            
            logger.info(f"✅ Step '{step_name}' completed successfully")
            
        except PipelineError as e:
            logger.error(f"❌ Step '{step_name}' failed: {e}")
            sys.exit(1)
    
    def _validate_prerequisites(self, step_name: str):
        """
        Validate prerequisites for a step.
        
        Args:
            step_name: Name of the step
            
        Raises:
            PipelineError: If prerequisites not met
        """
        logger.info(f"Validating prerequisites for '{step_name}'...")
        
        # Define prerequisites for each step
        prerequisites = {
            "collect_data": [],  # No prerequisites
            "extract_data": ["collect_data"],
            "build_node_graph": ["extract_data"],
            "aggregate_data": ["build_node_graph"],
            "detect_pii": ["aggregate_data"],
            "build_core_graph": ["detect_pii"],
        }
        
        required_steps = prerequisites.get(step_name, [])
        
        for required_step in required_steps:
            done_file = self._get_done_file_path(required_step)
            
            if not done_file.exists():
                raise PipelineError(
                    f"PIPELINE-PREREQ-001",
                    f"Prerequisite step '{required_step}' not completed. Missing: {done_file}"
                )
            
            logger.info(f"  ✅ Prerequisite '{required_step}' completed")
        
        logger.info("✅ All prerequisites validated")
    
    def _run_step(self, step_name: str):
        """
        Run a pipeline step.
        
        Args:
            step_name: Name of the step
            
        Raises:
            PipelineError: If step execution fails
        """
        logger.info(f"Executing step '{step_name}'...")
        
        # Map step names to script files
        step_scripts = {
            "collect_data": "main.py",
            "extract_data": "extract_data.py",
            "build_node_graph": "build_node_graph.py",
            "aggregate_data": "aggregate_data_v2.py",
            "detect_pii": "detect_pii.py",
            "build_core_graph": "build_core_graph.py",
        }
        
        script_name = step_scripts.get(step_name)
        if not script_name:
            raise PipelineError(
                "PIPELINE-STEP-001",
                f"Unknown step: {step_name}"
            )
        
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            raise PipelineError(
                "PIPELINE-STEP-002",
                f"Script not found: {script_path}"
            )
        
        # Setup log file for this step
        log_file = self.shared_data_path / 'logs' / f'{step_name}.log'
        
        # Run the script
        try:
            with open(log_file, 'w') as log_f:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.project_root),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
            
            if result.returncode != 0:
                # Read last 50 lines of log for error context
                with open(log_file, 'r') as log_f:
                    lines = log_f.readlines()
                    error_context = ''.join(lines[-50:])
                
                raise PipelineError(
                    f"PIPELINE-STEP-003",
                    f"Step '{step_name}' failed with exit code {result.returncode}\n"
                    f"Log file: {log_file}\n"
                    f"Last 50 lines:\n{error_context}"
                )
            
            logger.info(f"✅ Step '{step_name}' executed successfully")
            logger.info(f"Log file: {log_file}")
            
        except subprocess.TimeoutExpired:
            raise PipelineError(
                "PIPELINE-STEP-004",
                f"Step '{step_name}' timed out after 1 hour"
            )
        
        except Exception as e:
            raise PipelineError(
                "PIPELINE-STEP-005",
                f"Failed to execute step '{step_name}': {e}"
            )
    
    def _validate_output(self, step_name: str):
        """
        Validate output of a step.
        
        Args:
            step_name: Name of the step
            
        Raises:
            PipelineError: If output validation fails
        """
        logger.info(f"Validating output for '{step_name}'...")
        
        # Define expected outputs for each step
        validations = {
            "collect_data": self._validate_collect_data_output,
            "extract_data": self._validate_extract_data_output,
            "build_node_graph": self._validate_node_graph_output,
            "aggregate_data": self._validate_aggregate_data_output,
            "detect_pii": self._validate_pii_scan_output,
            "build_core_graph": self._validate_core_graph_output,
        }
        
        validation_func = validations.get(step_name)
        if validation_func:
            validation_func()
        
        logger.info(f"✅ Output validated for '{step_name}'")
    
    def _validate_collect_data_output(self):
        """Validate collected data output"""
        # Find latest collected_data folder
        collected_dir = self.shared_data_path / 'collected_data'
        if not collected_dir.exists():
            raise PipelineError(
                "PIPELINE-VALIDATE-001",
                f"Collected data directory not found: {collected_dir}"
            )
        
        # Find latest session
        sessions = [d for d in collected_dir.iterdir() if d.is_dir()]
        if not sessions:
            raise PipelineError(
                "PIPELINE-VALIDATE-002",
                "No collected data sessions found"
            )
        
        latest_session = max(sessions, key=lambda d: d.stat().st_mtime)
        
        # Check for required files
        required_files = ['agents_manager/All_Agents.json']
        for file_path in required_files:
            full_path = latest_session / file_path
            if not full_path.exists():
                raise PipelineError(
                    "PIPELINE-VALIDATE-003",
                    f"Required file not found: {full_path}"
                )
    
    def _validate_extract_data_output(self):
        """Validate extracted data output"""
        extracted_dir = self.shared_data_path / 'extracted_data'
        if not extracted_dir.exists():
            raise PipelineError(
                "PIPELINE-VALIDATE-004",
                f"Extracted data directory not found: {extracted_dir}"
            )
        
        sessions = [d for d in extracted_dir.iterdir() if d.is_dir()]
        if not sessions:
            raise PipelineError(
                "PIPELINE-VALIDATE-005",
                "No extracted data sessions found"
            )
        
        latest_session = max(sessions, key=lambda d: d.stat().st_mtime)
        
        # Check for required files
        required_files = ['agents.json', 'hosts.json', 'relationships.json']
        for file_name in required_files:
            full_path = latest_session / file_name
            if not full_path.exists():
                raise PipelineError(
                    "PIPELINE-VALIDATE-006",
                    f"Required file not found: {full_path}"
                )
    
    def _validate_node_graph_output(self):
        """Validate Node graph was created"""
        # TODO: Query Neo4j to verify nodes exist
        logger.info("Node graph validation: Assuming success (TODO: implement Neo4j query)")
    
    def _validate_aggregate_data_output(self):
        """Validate aggregated data output"""
        aggregated_dir = self.shared_data_path / 'aggregated_data_core'
        if not aggregated_dir.exists():
            raise PipelineError(
                "PIPELINE-VALIDATE-007",
                f"Aggregated data directory not found: {aggregated_dir}"
            )
        
        sessions = [d for d in aggregated_dir.iterdir() if d.is_dir()]
        if not sessions:
            raise PipelineError(
                "PIPELINE-VALIDATE-008",
                "No aggregated data sessions found"
            )
        
        latest_session = max(sessions, key=lambda d: d.stat().st_mtime)
        
        # Check for core_aggregation.json
        core_file = latest_session / 'core_aggregation.json'
        if not core_file.exists():
            raise PipelineError(
                "PIPELINE-VALIDATE-009",
                f"core_aggregation.json not found: {core_file}"
            )
    
    def _validate_pii_scan_output(self):
        """Validate PII scan output"""
        pii_dir = self.shared_data_path / 'pii_scan_results'
        if not pii_dir.exists():
            raise PipelineError(
                "PIPELINE-VALIDATE-010",
                f"PII scan results directory not found: {pii_dir}"
            )
        
        sessions = [d for d in pii_dir.iterdir() if d.is_dir()]
        if not sessions:
            raise PipelineError(
                "PIPELINE-VALIDATE-011",
                "No PII scan sessions found"
            )
        
        latest_session = max(sessions, key=lambda d: d.stat().st_mtime)
        
        # Check for results file
        results_file = latest_session / 'pii_scan_results.json'
        if not results_file.exists():
            raise PipelineError(
                "PIPELINE-VALIDATE-012",
                f"PII scan results not found: {results_file}"
            )
        
        # Verify no PII detected
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if not results.get('safe_for_core_graph', False):
            raise PipelineError(
                "PIPELINE-VALIDATE-013",
                f"PII/PCI detected in data. Not safe for Core Graph. "
                f"Findings: {results.get('total_pii_findings', 0)}"
            )
    
    def _validate_core_graph_output(self):
        """Validate Core graph was created"""
        # TODO: Query Neo4j to verify NJS_Bank node exists
        logger.info("Core graph validation: Assuming success (TODO: implement Neo4j query)")
    
    def _get_done_file_path(self, step_name: str) -> Path:
        """Get path to .done file for a step"""
        # Map steps to their output directories
        done_paths = {
            "collect_data": self.shared_data_path / 'collected_data' / '.done',
            "extract_data": self.shared_data_path / 'extracted_data' / '.done',
            "build_node_graph": self.shared_data_path / 'pipeline' / 'node_graph.done',
            "aggregate_data": self.shared_data_path / 'aggregated_data_core' / '.done',
            "detect_pii": self.shared_data_path / 'pii_scan_results' / '.done',
            "build_core_graph": self.shared_data_path / 'pipeline' / 'core_graph.done',
        }
        
        return done_paths.get(step_name, self.shared_data_path / 'pipeline' / f'{step_name}.done')
    
    def _create_done_file(self, step_name: str):
        """Create .done marker file for a step"""
        done_file = self._get_done_file_path(step_name)
        done_file.parent.mkdir(parents=True, exist_ok=True)
        
        done_data = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        with open(done_file, 'w') as f:
            json.dump(done_data, f, indent=2)
        
        logger.info(f"Created .done file: {done_file}")
    
    def _create_pipeline_done_file(self):
        """Create pipeline completion marker"""
        done_file = self.shared_data_path / 'pipeline' / '.done'
        done_file.parent.mkdir(parents=True, exist_ok=True)
        
        done_data = {
            "pipeline": "complete",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        with open(done_file, 'w') as f:
            json.dump(done_data, f, indent=2)
        
        logger.info(f"Created pipeline completion marker: {done_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='DataLegos Pipeline Orchestrator'
    )
    parser.add_argument(
        '--step',
        type=str,
        help='Run a single step (collect_data, extract_data, build_node_graph, aggregate_data, detect_pii, build_core_graph)'
    )
    parser.add_argument(
        '--shared-data-path',
        type=str,
        help='Path to shared data directory (default: from SHARED_DATA_PATH env var)'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    shared_data_path = Path(args.shared_data_path) if args.shared_data_path else None
    orchestrator = PipelineOrchestrator(shared_data_path)
    
    try:
        if args.step:
            # Run single step
            orchestrator.run_single_step(args.step)
        else:
            # Run complete pipeline
            orchestrator.run_pipeline()
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
