#!/usr/bin/env python3
"""
PII/PCI Detection Script using Microsoft Presidio
Scans aggregated data for sensitive information before sending to Core Graph.

Configuration-driven implementation following NJS standards:
- All patterns loaded from config/aggregation_config.yaml
- No hard-coded values
- Parameterized false positive filtering
"""

import json
import logging
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
from collections import defaultdict

try:
    from presidio_analyzer import (
        AnalyzerEngine, 
        RecognizerRegistry, 
        Pattern, 
        EntityRecognizer, 
        RecognizerResult
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider
except ImportError:
    print("ERROR: Presidio not installed. Install with:")
    print("  pip install presidio-analyzer presidio-anonymizer")
    print("  python -m spacy download en_core_web_lg")
    exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FalsePositiveFilter:
    """
    Configuration-driven false positive filter.
    Loads deny list patterns from config file.
    """
    
    def __init__(self, patterns: List[str] = None):
        """
        Initialize filter with patterns from config.
        
        Args:
            patterns: List of regex patterns to match false positives
        """
        self.patterns = patterns or []
        logger.info(f"Loaded {len(self.patterns)} false positive filter patterns")
    
    def is_false_positive(self, text: str) -> bool:
        """
        Check if detected text matches known false positive patterns.
        
        Args:
            text: The detected text to check
            
        Returns:
            True if it's a false positive, False otherwise
        """
        if not text:
            return False
        
        for pattern in self.patterns:
            try:
                # Use re.search() to match anywhere in the string
                # Use re.match() only for patterns starting with ^
                if pattern.startswith('^'):
                    if re.match(pattern, text, re.IGNORECASE):
                        logger.debug(f"Matched pattern '{pattern}' for text '{text}'")
                        return True
                else:
                    if re.search(pattern, text, re.IGNORECASE):
                        logger.debug(f"Matched pattern '{pattern}' for text '{text}'")
                        return True
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
        
        return False


class SoftwarePackageDenyListRecognizer(EntityRecognizer):
    """
    Custom recognizer that identifies software packages and CVE identifiers.
    Configuration-driven from aggregation_config.yaml.
    """
    
    def __init__(self, patterns: List[Dict[str, Any]] = None, 
                 deny_list_checker: Optional[Callable] = None):
        """
        Initialize recognizer with configurable patterns.
        
        Args:
            patterns: List of pattern dicts with 'name', 'regex', 'score'
            deny_list_checker: Function to check if text matches deny list
        """
        super().__init__(
            supported_entities=["SOFTWARE_PACKAGE"],
            supported_language="en",
            name="software_package_deny_list"
        )
        
        # Load patterns from config
        if patterns:
            self.patterns = []
            for p in patterns:
                try:
                    pattern = Pattern(
                        name=p.get('name', 'unknown'),
                        regex=p.get('regex', ''),
                        score=p.get('score', 0.85)
                    )
                    self.patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to load pattern {p.get('name')}: {e}")
        else:
            # Minimal fallback if config not provided
            self.patterns = []
        
        self.deny_list_checker = deny_list_checker
        logger.info(f"Loaded {len(self.patterns)} custom recognizer patterns")
    
    def load(self) -> None:
        """Load the recognizer - no external resources needed."""
        pass
    
    def analyze(self, text: str, entities: List[str], 
                nlp_artifacts=None) -> List[RecognizerResult]:
        """
        Analyze text and identify software packages/CVE identifiers.
        
        Args:
            text: Text to analyze
            entities: List of entities to look for
            nlp_artifacts: NLP artifacts (not used)
            
        Returns:
            List of RecognizerResult for detected software patterns
        """
        results = []
        
        for pattern in self.patterns:
            try:
                matches = re.finditer(pattern.regex, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group()
                    
                    # Validate using deny list if provided
                    if self.deny_list_checker and not self.deny_list_checker(matched_text):
                        continue
                    
                    result = RecognizerResult(
                        entity_type="SOFTWARE_PACKAGE",
                        start=match.start(),
                        end=match.end(),
                        score=pattern.score
                    )
                    results.append(result)
            except re.error as e:
                logger.warning(f"Invalid regex in pattern {pattern.name}: {e}")
                continue
        
        return results


class PIIDetector:
    """
    Detects PII/PCI and sensitive data using Microsoft Presidio.
    Configuration-driven from aggregation_config.yaml.
    """
    
    def __init__(self, confidence_threshold: float = 0.5, config_path: str = None):
        """
        Initialize Presidio analyzer with configuration.
        
        Args:
            confidence_threshold: Minimum confidence score (0.0 to 1.0)
            config_path: Path to aggregation_config.yaml (optional)
        """
        self.confidence_threshold = confidence_threshold
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.entity_types = self.config.get('entity_types', [])
        
        # Initialize false positive filter
        filter_config = self.config.get('false_positive_filters', {})
        self.filter_enabled = filter_config.get('enabled', False)
        
        if self.filter_enabled:
            deny_patterns = filter_config.get('deny_list_patterns', [])
            self.false_positive_filter = FalsePositiveFilter(deny_patterns)
        else:
            self.false_positive_filter = None
        
        logger.info("Initializing Presidio Analyzer...")
        
        # Create NLP engine
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        }
        
        try:
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()
            
            # Create custom registry
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            
            # Add custom software package recognizer if configured
            if self.filter_enabled:
                custom_patterns = filter_config.get('custom_recognizer_patterns', [])
                if custom_patterns:
                    software_recognizer = SoftwarePackageDenyListRecognizer(
                        patterns=custom_patterns,
                        deny_list_checker=self.false_positive_filter.is_false_positive
                    )
                    registry.add_recognizer(software_recognizer)
                    logger.info("Added custom Software Package Deny List Recognizer")
            
            # Create analyzer
            self.analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                registry=registry
            )
            logger.info("Presidio Analyzer initialized successfully")
            logger.info(f"Enabled entity types: {', '.join(self.entity_types)}")
            logger.info(f"False positive filtering: {'enabled' if self.filter_enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {e}")
            logger.info("Trying with default configuration...")
            self.analyzer = AnalyzerEngine()
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        # Load paths config to get aggregation config path
        project_root = get_project_root()
        paths_config = load_paths_config()
        
        if config_path is None:
            config_path = project_root / paths_config['aggregated_config']
        else:
            config_path = Path(config_path)
        
        # Default configuration
        default_config = {
            'entity_types': [
                'EMAIL_ADDRESS',
                'PHONE_NUMBER',
                'CREDIT_CARD',
                'US_SSN',
                'IP_ADDRESS'
            ],
            'false_positive_filters': {
                'enabled': False,
                'deny_list_patterns': [],
                'custom_recognizer_patterns': []
            }
        }
        
        try:
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return default_config
            
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            pii_config = full_config.get('pii_detection', {})
            
            if not pii_config.get('enabled', False):
                logger.warning("PII detection disabled in config, using defaults")
                return default_config
            
            # Extract configuration
            config = {
                'entity_types': pii_config.get('entities_to_detect', default_config['entity_types']),
                'false_positive_filters': pii_config.get('false_positive_filters', default_config['false_positive_filters'])
            }
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return default_config
    
    def analyze_text(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        Analyze text for PII/PCI with false positive filtering.
        
        Args:
            text: Text to analyze
            context: Context information (e.g., field name)
            
        Returns:
            List of detected entities
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            results = self.analyzer.analyze(
                text=text,
                entities=self.entity_types,
                language='en'
            )
            
            # Filter results
            filtered_results = []
            false_positive_count = 0
            
            for result in results:
                if result.score < self.confidence_threshold:
                    continue
                
                detected_text = text[result.start:result.end]
                
                # Apply false positive filter for PERSON, LOCATION, and DATE_TIME entities
                if self.filter_enabled and self.false_positive_filter:
                    # Apply filter to entities that commonly have false positives
                    if result.entity_type in ["PERSON", "LOCATION", "DATE_TIME"]:
                        if self.false_positive_filter.is_false_positive(detected_text):
                            false_positive_count += 1
                            logger.debug(f"Filtered false positive ({result.entity_type}): {detected_text}")
                            continue
                
                # Skip SOFTWARE_PACKAGE entities (markers, not PII)
                if result.entity_type == "SOFTWARE_PACKAGE":
                    continue
                
                filtered_results.append({
                    'entity_type': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'score': result.score,
                    'text': detected_text,
                    'context': context
                })
            
            if false_positive_count > 0:
                logger.info(f"Filtered {false_positive_count} false positives from '{context}'")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return []
    
    def scan_json_recursive(self, data: Any, path: str = "root") -> List[Dict[str, Any]]:
        """
        Recursively scan JSON data for PII/PCI.
        
        Args:
            data: JSON data (dict, list, or primitive)
            path: Current path in JSON structure
            
        Returns:
            List of detected PII/PCI entities
        """
        findings = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}"
                
                # Analyze key name
                key_findings = self.analyze_text(str(key), f"{current_path} (key)")
                findings.extend(key_findings)
                
                # Recursively analyze value
                value_findings = self.scan_json_recursive(value, current_path)
                findings.extend(value_findings)
                
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                current_path = f"{path}[{idx}]"
                item_findings = self.scan_json_recursive(item, current_path)
                findings.extend(item_findings)
                
        elif isinstance(data, str):
            text_findings = self.analyze_text(data, path)
            findings.extend(text_findings)
            
        elif isinstance(data, (int, float, bool)) or data is None:
            text_findings = self.analyze_text(str(data), path)
            findings.extend(text_findings)
        
        return findings
    
    def scan_json_file(self, file_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Scan a JSON file for PII/PCI.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Tuple of (findings, statistics)
        """
        logger.info(f"Scanning file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            findings = self.scan_json_recursive(data, file_path.name)
            
            # Calculate statistics
            entity_counts = defaultdict(int)
            for finding in findings:
                entity_counts[finding['entity_type']] += 1
            
            stats = {
                'file': str(file_path),
                'total_findings': len(findings),
                'entity_counts': dict(entity_counts),
                'has_pii': len(findings) > 0
            }
            
            return findings, stats
            
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            return [], {'file': str(file_path), 'error': str(e)}


class PIIScanner:
    """
    Scans aggregated data directory for PII/PCI.
    """
    
    def __init__(self, aggregated_data_path: Path, output_path: Path, 
                 confidence_threshold: float = 0.5):
        self.aggregated_data_path = aggregated_data_path
        self.output_path = output_path
        self.detector = PIIDetector(confidence_threshold)
        self.scan_results = {}
    
    def scan_directory(self) -> Dict[str, Any]:
        """Scan JSON files in aggregated data directory."""
        logger.info(f"Scanning directory: {self.aggregated_data_path}")
        
        if not self.aggregated_data_path.exists():
            logger.error(f"Directory not found: {self.aggregated_data_path}")
            return {}
        
        # Only scan core_aggregation.json (the file used for core graph)
        core_file = self.aggregated_data_path / 'core_aggregation.json'
        
        if not core_file.exists():
            logger.error(f"core_aggregation.json not found in {self.aggregated_data_path}")
            logger.info("This file is required for core graph submission")
            return {}
        
        logger.info(f"Scanning core_aggregation.json (used for core graph)")
        
        findings, stats = self.detector.scan_json_file(core_file)
        
        # Aggregate results
        total_findings = len(findings)
        entity_type_counts = defaultdict(int)
        
        for finding in findings:
            entity_type_counts[finding['entity_type']] += 1
        
        results = {
            'scan_timestamp': datetime.now().isoformat(),
            'directory_scanned': str(self.aggregated_data_path),
            'file_scanned': str(core_file),
            'total_files_scanned': 1,
            'files_with_pii': 1 if total_findings > 0 else 0,
            'total_pii_findings': total_findings,
            'entity_type_distribution': dict(entity_type_counts),
            'file_statistics': [stats],
            'detailed_findings': findings,
            'privacy_compliant': total_findings == 0,
            'safe_for_core_graph': total_findings == 0
        }
        
        self.scan_results = results
        return results

    
    def save_results(self, session_name: str = None):
        """Save scan results to output directory."""
        if not self.scan_results:
            logger.error("No scan results to save. Run scan_directory() first.")
            return
        
        # Load paths config for filenames
        paths_config = load_paths_config()
        
        # Create session folder
        if session_name is None:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        session_path = self.output_path / session_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        results_filename = paths_config['pii_scan_results_filename']
        results_file = session_path / results_filename
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.scan_results, f, indent=2)
        logger.info(f"Saved scan results to {results_file}")
        
        # Save summary report
        self._create_summary_report(session_path, paths_config)
        
        # Save detailed findings if any
        if self.scan_results['detailed_findings']:
            findings_filename = paths_config['pii_detailed_findings_filename']
            findings_file = session_path / findings_filename
            with open(findings_file, 'w', encoding='utf-8') as f:
                json.dump(self.scan_results['detailed_findings'], f, indent=2)
            logger.info(f"Saved detailed findings to {findings_file}")
        
        logger.info(f"All results saved to {session_path}")
        return session_path
    
    def _create_summary_report(self, session_path: Path, paths_config: Dict[str, Any]):
        """Create human-readable summary report."""
        results = self.scan_results
        
        summary_lines = [
            "=" * 80,
            "PII/PCI DETECTION SCAN REPORT",
            "=" * 80,
            f"Scan Timestamp: {results['scan_timestamp']}",
            f"Directory Scanned: {results['directory_scanned']}",
            "=" * 80,
            "",
            "SCAN SUMMARY",
            "-" * 80,
            f"Total Files Scanned: {results['total_files_scanned']}",
            f"Files with PII/PCI: {results['files_with_pii']}",
            f"Total PII/PCI Findings: {results['total_pii_findings']}",
            "",
        ]
        
        # Privacy status
        if results['privacy_compliant']:
            summary_lines.extend([
                "✅ PRIVACY STATUS: COMPLIANT",
                "✅ SAFE FOR CORE GRAPH: YES",
                "✅ NO PII/PCI DETECTED",
                ""
            ])
        else:
            summary_lines.extend([
                "❌ PRIVACY STATUS: NON-COMPLIANT",
                "❌ SAFE FOR CORE GRAPH: NO",
                "❌ PII/PCI DETECTED - REVIEW REQUIRED",
                ""
            ])
        
        # Entity type distribution
        if results['entity_type_distribution']:
            summary_lines.extend([
                "DETECTED ENTITY TYPES",
                "-" * 80
            ])
            for entity_type, count in sorted(
                results['entity_type_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                summary_lines.append(f"  {entity_type}: {count}")
            summary_lines.append("")
        
        # File-by-file summary
        summary_lines.extend([
            "FILE-BY-FILE SUMMARY",
            "-" * 80
        ])
        
        for file_stat in results['file_statistics']:
            file_name = Path(file_stat['file']).name
            has_pii = file_stat.get('has_pii', False)
            total = file_stat.get('total_findings', 0)
            
            status = "❌ PII FOUND" if has_pii else "✅ CLEAN"
            summary_lines.append(f"  {file_name}: {status} ({total} findings)")
            
            if has_pii and file_stat.get('entity_counts'):
                for entity_type, count in file_stat['entity_counts'].items():
                    summary_lines.append(f"    - {entity_type}: {count}")
        
        summary_lines.append("")
        
        # Detailed findings (if any)
        if results['detailed_findings']:
            summary_lines.extend([
                "DETAILED FINDINGS",
                "-" * 80,
                ""
            ])
            
            for idx, finding in enumerate(results['detailed_findings'][:20], 1):
                summary_lines.extend([
                    f"Finding #{idx}:",
                    f"  Entity Type: {finding['entity_type']}",
                    f"  Confidence: {finding['score']:.2f}",
                    f"  Location: {finding['context']}",
                    f"  Text: {finding['text']}",
                    ""
                ])
            
            if len(results['detailed_findings']) > 20:
                summary_lines.append(f"... and {len(results['detailed_findings']) - 20} more findings")
                summary_lines.append("See detailed_findings.json for complete list")
                summary_lines.append("")
        
        # Recommendations
        summary_lines.extend([
            "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            ""
        ])
        
        if results['privacy_compliant']:
            summary_lines.extend([
                "✅ Data is safe to send to Core Graph",
                "✅ No PII/PCI detected",
                "✅ Privacy compliance verified",
                "",
                "Next Steps:",
                "  1. Proceed with Core Graph submission",
                "  2. Run: python scripts/build_core_graph.py",
                ""
            ])
        else:
            summary_lines.extend([
                "❌ DO NOT send this data to Core Graph",
                "❌ PII/PCI detected - must be removed or anonymized",
                "",
                "Required Actions:",
                "  1. Review detailed findings in detailed_findings.json",
                "  2. Remove or anonymize detected PII/PCI",
                "  3. Re-run aggregation with privacy fixes",
                "  4. Re-scan with: python scripts/detect_pii.py",
                ""
            ])
        
        summary_lines.append("=" * 80)
        
        # Save summary report
        summary_filename = paths_config['pii_scan_summary_filename']
        summary_file = session_path / summary_filename
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        logger.info(f"Saved summary report to {summary_file}")


def get_project_root() -> Path:
    """
    Get the project root directory.
    Assumes script is in scripts/ subdirectory.
    
    Returns:
        Path to project root
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Project root is one level up from scripts/
    project_root = script_dir.parent
    
    return project_root


def load_paths_config() -> Dict[str, Any]:
    """
    Load paths configuration from paths_config.yaml.
    
    Returns:
        Dictionary with path configurations
    """
    project_root = get_project_root()
    config_file = project_root / 'config' / 'paths_config.yaml'
    
    # Default paths (relative to project root)
    default_paths = {
        'aggregated_config': 'config/aggregation_config.yaml',
        'paths_config': 'config/paths_config.yaml',
        'aggregated_core_directory': 'aggregated_data_core',
        'pii_scan_results_directory': 'pii_scan_results',
        'pii_scan_results_filename': 'pii_scan_results.json',
        'pii_scan_summary_filename': 'pii_scan_summary.txt',
        'pii_detailed_findings_filename': 'detailed_findings.json',
        'current_session_marker': '.current_session'
    }
    
    try:
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return default_paths
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        paths = config.get('paths', {})
        
        # Merge with defaults
        result = default_paths.copy()
        result.update({
            'aggregated_core_directory': paths.get('aggregated_core_directory', default_paths['aggregated_core_directory']),
            'pii_scan_results_directory': paths.get('pii_scan_results_directory', default_paths['pii_scan_results_directory'])
        })
        
        logger.info(f"Loaded paths configuration from {config_file}")
        return result
        
    except Exception as e:
        logger.error(f"Error loading paths config: {e}, using defaults")
        return default_paths


def load_output_path_from_config() -> Path:
    """
    Load PII scan results output path from paths_config.yaml.
    
    Returns:
        Path object for PII scan results directory (absolute path)
    """
    project_root = get_project_root()
    paths_config = load_paths_config()
    pii_path = paths_config.get('pii_scan_results_directory', 'pii_scan_results')
    return project_root / pii_path


def find_latest_aggregated_core_folder() -> Path:
    """Find the latest aggregated_data_core folder using config."""
    project_root = get_project_root()
    paths_config = load_paths_config()
    aggregated_core_path = project_root / paths_config['aggregated_core_directory']
    
    if not aggregated_core_path.exists():
        raise FileNotFoundError(f"{aggregated_core_path} folder not found")
    
    # Look for .current_session file
    current_session_marker = paths_config['current_session_marker']
    current_session_file = aggregated_core_path / current_session_marker
    
    if current_session_file.exists():
        with open(current_session_file, 'r') as f:
            session_name = f.read().strip()
            session_path = aggregated_core_path / session_name
            if session_path.exists():
                logger.info(f"Using current session: {session_name}")
                return session_path
    
    # Fallback: find latest folder by timestamp
    session_folders = [f for f in aggregated_core_path.iterdir() if f.is_dir()]
    if not session_folders:
        raise FileNotFoundError(f"No {aggregated_core_path} sessions found")
    
    latest_folder = max(session_folders, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using latest {aggregated_core_path} folder: {latest_folder.name}")
    return latest_folder


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scan aggregated data for PII/PCI using Microsoft Presidio'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to aggregated data directory (default: latest in aggregated_data_core/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for scan results (default: from paths_config.yaml)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold (0.0-1.0, default: 0.5)'
    )
    
    args = parser.parse_args()
    
    try:
        # Determine input path
        if args.input:
            aggregated_data_path = Path(args.input)
        else:
            aggregated_data_path = find_latest_aggregated_core_folder()
        
        if not aggregated_data_path.exists():
            logger.error(f"Input path not found: {aggregated_data_path}")
            return 1
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = load_output_path_from_config()
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create scanner
        scanner = PIIScanner(
            aggregated_data_path,
            output_path,
            confidence_threshold=args.confidence
        )
        
        # Run scan
        logger.info("=" * 80)
        logger.info("Starting PII/PCI Detection Scan...")
        logger.info(f"Confidence Threshold: {args.confidence}")
        logger.info("=" * 80)
        
        results = scanner.scan_directory()
        
        if not results:
            logger.error("Scan failed - no results generated")
            return 1
        
        # Save results
        session_path = scanner.save_results()
        
        # Print summary
        logger.info("=" * 80)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total Files Scanned: {results['total_files_scanned']}")
        logger.info(f"Files with PII/PCI: {results['files_with_pii']}")
        logger.info(f"Total PII/PCI Findings: {results['total_pii_findings']}")
        logger.info("")
        
        if results['privacy_compliant']:
            logger.info("✅ PRIVACY STATUS: COMPLIANT")
            logger.info("✅ SAFE FOR CORE GRAPH: YES")
            logger.info("✅ NO PII/PCI DETECTED")
        else:
            logger.error("❌ PRIVACY STATUS: NON-COMPLIANT")
            logger.error("❌ SAFE FOR CORE GRAPH: NO")
            logger.error("❌ PII/PCI DETECTED - REVIEW REQUIRED")
        
        logger.info("")
        logger.info(f"Results saved to: {session_path}")
        logger.info("=" * 80)
        
        # Return exit code based on compliance
        return 0 if results['privacy_compliant'] else 1
        
    except Exception as e:
        logger.error(f"Error during PII/PCI scan: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
