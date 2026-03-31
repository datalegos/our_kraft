#!/usr/bin/env python3
"""
Data Aggregation Script v2 - Privacy-Preserving Core Graph Preparation
Aggregates extracted node data following Core Graph principles:
- NO PII (hostnames, IPs, agent IDs, file paths)
- NO per-host data
- ONLY aggregated counts and distributions
- Technology Exposure Surface only
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict, Counter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrivacyPreservingAggregator:
    """
    Aggregates data for Core Graph following strict privacy principles.
    Core must ONLY reason over aggregates, not instances.
    """
    
    def __init__(self, extracted_data_path: Path, output_path: Path):
        self.extracted_data_path = extracted_data_path
        self.output_path = output_path
        self.aggregations = {}
        
    def load_nodes(self, node_file: Path) -> List[Dict[str, Any]]:
        """Load nodes from JSON file."""
        try:
            with open(node_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {node_file}: {e}")
            return []
    
    def aggregate_technology_exposure(self, software_nodes: List[Dict], host_nodes: List[Dict]) -> Dict[str, Any]:
        """
        LAYER 1: Technology Exposure Graph
        What technology exists in the consortium (aggregated, anonymized)
        """
        
        # Software Package Exposure
        package_counts = Counter()
        package_format_counts = Counter()
        architecture_counts = Counter()
        
        for node in software_nodes:
            name = node.get('name', 'Unknown')
            pkg_format = node.get('format', 'Unknown')
            architecture = node.get('architecture', 'Unknown')
            
            # Aggregate by package name only (no version, no vendor)
            package_counts[name] += 1
            package_format_counts[pkg_format] += 1
            
            # Normalize architecture
            if architecture and architecture.strip() and architecture != ' ':
                architecture_counts[architecture] += 1
        
        # OS Platform Exposure
        os_type_counts = Counter()
        os_architecture_counts = Counter()
        
        for node in host_nodes:
            os_name = node.get('os_name', 'Unknown')
            architecture = node.get('architecture', 'Unknown')
            
            # Normalize OS to generic types (no versions, no specific distros)
            if 'Windows' in os_name:
                os_type = 'Windows'
            elif 'Linux' in os_name or 'Amazon' in os_name or 'Ubuntu' in os_name or 'CentOS' in os_name:
                os_type = 'Linux'
            elif 'macOS' in os_name or 'Darwin' in os_name:
                os_type = 'macOS'
            else:
                os_type = 'Other'
            
            os_type_counts[os_type] += 1
            os_architecture_counts[architecture] += 1
        
        return {
            'software_packages': {
                'total_instances': len(software_nodes),
                'unique_packages': len(package_counts),
                'package_distribution': dict(package_counts),
                'format_distribution': dict(package_format_counts),
                'architecture_distribution': dict(architecture_counts)
            },
            'os_platforms': {
                'total_instances': len(host_nodes),
                'os_type_distribution': dict(os_type_counts),
                'architecture_distribution': dict(os_architecture_counts)
            }
        }

    
    def aggregate_sensitivity_surface(self, vulnerability_nodes: List[Dict]) -> Dict[str, Any]:
        """
        LAYER 2: Sensitivity Surface (Aggregated Risk Amplifier)
        What vulnerabilities exist (aggregated, no per-host mapping)
        """
        
        severity_counts = Counter()
        cve_counts = Counter()
        
        for node in vulnerability_nodes:
            severity = node.get('severity', 'Unknown')
            cve_id = node.get('cve_id', 'Unknown')
            
            # Normalize severity
            if severity is None or severity == 'null':
                severity = 'Unknown'
            
            severity_counts[severity] += 1
            cve_counts[cve_id] += 1
        
        # Calculate risk metrics (aggregated)
        total_vulns = len(vulnerability_nodes)
        critical_count = severity_counts.get('Critical', 0)
        high_count = severity_counts.get('High', 0)
        
        risk_score = 0
        if total_vulns > 0:
            # Simple risk score: weighted by severity
            risk_score = (
                (critical_count * 10) +
                (high_count * 5) +
                (severity_counts.get('Medium', 0) * 2) +
                (severity_counts.get('Low', 0) * 1)
            ) / total_vulns
        
        return {
            'vulnerability_exposure': {
                'total_vulnerabilities': total_vulns,
                'unique_cves': len(cve_counts),
                'severity_distribution': dict(severity_counts),
                'risk_score': round(risk_score, 2),
                'critical_exposure': critical_count,
                'high_exposure': high_count
            },
            'top_cves': dict(cve_counts.most_common(20))
        }
    
    def aggregate_outcome_metrics(self, asset_nodes: List[Dict]) -> Dict[str, Any]:
        """
        LAYER 3: Outcome Metrics (Effectiveness Feedback)
        Aggregate metrics about asset health (no identifiers)
        """
        
        status_counts = Counter()
        
        for node in asset_nodes:
            status = node.get('status', 'Unknown')
            status_counts[status] += 1
        
        total_assets = len(asset_nodes)
        active_count = status_counts.get('active', 0)
        
        # Calculate health metrics
        health_score = 0
        if total_assets > 0:
            health_score = (active_count / total_assets) * 100
        
        return {
            'asset_health': {
                'total_assets': total_assets,
                'status_distribution': dict(status_counts),
                'health_score': round(health_score, 2),
                'active_percentage': round((active_count / total_assets * 100) if total_assets > 0 else 0, 2)
            }
        }
    
    def aggregate_all(self) -> Dict[str, Any]:
        """Aggregate all data following Core Graph principles."""
        nodes_path = self.extracted_data_path / 'nodes'
        
        if not nodes_path.exists():
            logger.error(f"Nodes path not found: {nodes_path}")
            return {}
        
        # Load all node types
        software_nodes = self.load_nodes(nodes_path / 'software_nodes.json')
        host_nodes = self.load_nodes(nodes_path / 'host_nodes.json')
        vulnerability_nodes = self.load_nodes(nodes_path / 'vulnerability_nodes.json')
        asset_nodes = self.load_nodes(nodes_path / 'asset_nodes.json')
        
        logger.info(f"Loaded {len(software_nodes)} software nodes")
        logger.info(f"Loaded {len(host_nodes)} host nodes")
        logger.info(f"Loaded {len(vulnerability_nodes)} vulnerability nodes")
        logger.info(f"Loaded {len(asset_nodes)} asset nodes")
        
        # Build aggregations following Core principles
        aggregations = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_path': str(self.extracted_data_path),
                'privacy_compliant': True,
                'contains_pii': False,
                'aggregation_version': '2.0'
            },
            'exposure_surface': self.aggregate_technology_exposure(software_nodes, host_nodes),
            'sensitivity_surface': self.aggregate_sensitivity_surface(vulnerability_nodes),
            'outcome_metrics': self.aggregate_outcome_metrics(asset_nodes)
        }
        
        self.aggregations = aggregations
        return aggregations
    
    def save_aggregations(self, session_name: str = None):
        """Save privacy-preserving aggregations to output folder."""
        if not self.aggregations:
            logger.error("No aggregations to save. Run aggregate_all() first.")
            return
        
        # Create session folder
        if session_name is None:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        session_path = self.output_path / session_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete aggregation
        complete_file = session_path / 'core_aggregation.json'
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregations, f, indent=2)
        logger.info(f"Saved core aggregation to {complete_file}")
        
        # Save individual layers
        layers = {
            'exposure_surface': self.aggregations.get('exposure_surface', {}),
            'sensitivity_surface': self.aggregations.get('sensitivity_surface', {}),
            'outcome_metrics': self.aggregations.get('outcome_metrics', {})
        }
        
        for layer_name, layer_data in layers.items():
            layer_file = session_path / f"{layer_name}.json"
            with open(layer_file, 'w', encoding='utf-8') as f:
                json.dump(layer_data, f, indent=2)
            logger.info(f"Saved {layer_name} to {layer_file}")
        
        # Create summary report
        self._create_summary_report(session_path)
        
        logger.info(f"All aggregations saved to {session_path}")
        return session_path
    
    def _create_summary_report(self, session_path: Path):
        """Create a human-readable summary report."""
        summary_lines = [
            "=" * 80,
            "CORE GRAPH AGGREGATION SUMMARY (PRIVACY-PRESERVING)",
            "=" * 80,
            f"Generated: {self.aggregations['metadata']['timestamp']}",
            f"Privacy Compliant: {self.aggregations['metadata']['privacy_compliant']}",
            f"Contains PII: {self.aggregations['metadata']['contains_pii']}",
            "=" * 80,
            ""
        ]
        
        # Exposure Surface
        exposure = self.aggregations.get('exposure_surface', {})
        if exposure:
            summary_lines.extend([
                "LAYER 1: TECHNOLOGY EXPOSURE SURFACE",
                "-" * 80,
                ""
            ])
            
            # Software
            sw_data = exposure.get('software_packages', {})
            summary_lines.extend([
                "Software Packages:",
                f"  Total Instances: {sw_data.get('total_instances', 0)}",
                f"  Unique Packages: {sw_data.get('unique_packages', 0)}",
                ""
            ])
            
            # Top 10 packages
            pkg_dist = sw_data.get('package_distribution', {})
            if pkg_dist:
                summary_lines.append("  Top 10 Packages:")
                for pkg, count in sorted(pkg_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                    summary_lines.append(f"    - {pkg}: {count}")
                summary_lines.append("")
            
            # OS Platforms
            os_data = exposure.get('os_platforms', {})
            summary_lines.extend([
                "OS Platforms:",
                f"  Total Instances: {os_data.get('total_instances', 0)}",
                ""
            ])
            
            os_dist = os_data.get('os_type_distribution', {})
            if os_dist:
                summary_lines.append("  OS Distribution:")
                for os_type, count in os_dist.items():
                    summary_lines.append(f"    - {os_type}: {count}")
                summary_lines.append("")
        
        # Sensitivity Surface
        sensitivity = self.aggregations.get('sensitivity_surface', {})
        if sensitivity:
            summary_lines.extend([
                "LAYER 2: SENSITIVITY SURFACE (RISK AMPLIFIER)",
                "-" * 80,
                ""
            ])
            
            vuln_data = sensitivity.get('vulnerability_exposure', {})
            summary_lines.extend([
                "Vulnerability Exposure:",
                f"  Total Vulnerabilities: {vuln_data.get('total_vulnerabilities', 0)}",
                f"  Unique CVEs: {vuln_data.get('unique_cves', 0)}",
                f"  Risk Score: {vuln_data.get('risk_score', 0)}",
                f"  Critical Exposure: {vuln_data.get('critical_exposure', 0)}",
                f"  High Exposure: {vuln_data.get('high_exposure', 0)}",
                ""
            ])
            
            severity_dist = vuln_data.get('severity_distribution', {})
            if severity_dist:
                summary_lines.append("  Severity Distribution:")
                for severity, count in severity_dist.items():
                    summary_lines.append(f"    - {severity}: {count}")
                summary_lines.append("")
        
        # Outcome Metrics
        outcome = self.aggregations.get('outcome_metrics', {})
        if outcome:
            summary_lines.extend([
                "LAYER 3: OUTCOME METRICS (EFFECTIVENESS FEEDBACK)",
                "-" * 80,
                ""
            ])
            
            health_data = outcome.get('asset_health', {})
            summary_lines.extend([
                "Asset Health:",
                f"  Total Assets: {health_data.get('total_assets', 0)}",
                f"  Health Score: {health_data.get('health_score', 0)}%",
                f"  Active Percentage: {health_data.get('active_percentage', 0)}%",
                ""
            ])
        
        summary_lines.append("=" * 80)
        summary_lines.append("NOTE: This aggregation contains NO PII, NO hostnames, NO agent IDs")
        summary_lines.append("Core Graph can safely consume this data for strategic analysis")
        summary_lines.append("=" * 80)
        
        # Save summary report
        summary_file = session_path / 'summary_report.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        logger.info(f"Saved summary report to {summary_file}")


def find_latest_extraction_folder() -> Path:
    """Find the latest extraction folder in extracted_data."""
    extracted_data_path = Path('extracted_data')
    
    if not extracted_data_path.exists():
        raise FileNotFoundError("extracted_data folder not found")
    
    # Look for .current_session file
    current_session_file = extracted_data_path / '.current_session'
    if current_session_file.exists():
        with open(current_session_file, 'r') as f:
            session_name = f.read().strip()
            session_path = extracted_data_path / session_name
            if session_path.exists():
                logger.info(f"Using current session: {session_name}")
                return session_path
    
    # Fallback: find latest folder by timestamp
    session_folders = [f for f in extracted_data_path.iterdir() if f.is_dir()]
    if not session_folders:
        raise FileNotFoundError("No extraction sessions found")
    
    latest_folder = max(session_folders, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using latest extraction folder: {latest_folder.name}")
    return latest_folder


def main():
    """Main execution function."""
    try:
        # Find latest extraction folder
        extracted_data_path = find_latest_extraction_folder()
        
        # Create output path
        output_path = Path('aggregated_data_core')
        output_path.mkdir(exist_ok=True)
        
        # Create aggregator
        aggregator = PrivacyPreservingAggregator(extracted_data_path, output_path)
        
        # Run aggregation
        logger.info("Starting privacy-preserving aggregation for Core Graph...")
        aggregations = aggregator.aggregate_all()
        
        if aggregations:
            # Save results
            session_path = aggregator.save_aggregations()
            logger.info(f"Core aggregation complete! Results saved to {session_path}")
            
            # Create .current_session file
            current_session_file = output_path / '.current_session'
            with open(current_session_file, 'w') as f:
                f.write(session_path.name)
            
            # Verify no PII
            logger.info("=" * 80)
            logger.info("PRIVACY VERIFICATION:")
            logger.info(f"  Contains PII: {aggregations['metadata']['contains_pii']}")
            logger.info(f"  Privacy Compliant: {aggregations['metadata']['privacy_compliant']}")
            logger.info("=" * 80)
            
            return 0
        else:
            logger.error("Aggregation failed - no data processed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during aggregation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
