#!/usr/bin/env python3
"""
Data Aggregation Script
Aggregates extracted node data to generate counts and statistics.
Stores aggregated data in aggregated_data folder for downstream processing.
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


class DataAggregator:
    """Aggregates extracted node data into counts and statistics."""
    
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
    
    def aggregate_hosts(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate host/OS information."""
        os_counts = Counter()
        platform_counts = Counter()
        architecture_counts = Counter()
        os_version_counts = Counter()
        
        for node in nodes:
            os_name = node.get('os_name', 'Unknown')
            platform = node.get('platform', 'Unknown')
            architecture = node.get('architecture', 'Unknown')
            os_version = node.get('os_version', 'Unknown')
            
            # Normalize OS names
            if 'Windows' in os_name:
                os_type = 'Windows'
            elif 'Linux' in os_name or 'Amazon' in os_name:
                os_type = 'Linux'
            elif 'macOS' in os_name or 'darwin' in platform:
                os_type = 'macOS'
            else:
                os_type = os_name
            
            os_counts[os_type] += 1
            platform_counts[platform] += 1
            architecture_counts[architecture] += 1
            os_version_counts[f"{os_type}-{os_version}"] += 1
        
        return {
            'total_hosts': len(nodes),
            'os_distribution': dict(os_counts),
            'platform_distribution': dict(platform_counts),
            'architecture_distribution': dict(architecture_counts),
            'os_version_distribution': dict(os_version_counts)
        }

    def aggregate_software(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate software package information."""
        software_counts = Counter()
        vendor_counts = Counter()
        format_counts = Counter()
        architecture_counts = Counter()
        version_counts = defaultdict(Counter)
        
        for node in nodes:
            name = node.get('name', 'Unknown')
            version = node.get('version', 'Unknown')
            vendor = node.get('vendor', 'Unknown')
            pkg_format = node.get('format', 'Unknown')
            architecture = node.get('architecture', 'Unknown')
            
            software_counts[name] += 1
            vendor_counts[vendor] += 1
            format_counts[pkg_format] += 1
            architecture_counts[architecture] += 1
            version_counts[name][version] += 1
        
        # Get top software packages
        top_software = dict(software_counts.most_common(50))
        
        # Get software with multiple versions
        multi_version_software = {
            name: dict(versions) 
            for name, versions in version_counts.items() 
            if len(versions) > 1
        }
        
        return {
            'total_packages': len(nodes),
            'unique_packages': len(software_counts),
            'top_50_packages': top_software,
            'vendor_distribution': dict(vendor_counts),
            'format_distribution': dict(format_counts),
            'architecture_distribution': dict(architecture_counts),
            'multi_version_packages': multi_version_software,
            'multi_version_count': len(multi_version_software)
        }
    
    def aggregate_vulnerabilities(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate vulnerability information."""
        severity_counts = Counter()
        cve_counts = Counter()
        status_counts = Counter()
        
        for node in nodes:
            severity = node.get('severity', 'Unknown')
            cve_id = node.get('cve_id', 'Unknown')
            status = node.get('status', 'Unknown')
            
            severity_counts[severity] += 1
            cve_counts[cve_id] += 1
            status_counts[status] += 1
        
        # Get top CVEs
        top_cves = dict(cve_counts.most_common(20))
        
        return {
            'total_vulnerabilities': len(nodes),
            'unique_cves': len(cve_counts),
            'severity_distribution': dict(severity_counts),
            'status_distribution': dict(status_counts),
            'top_20_cves': top_cves
        }
    
    def aggregate_hardware(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate hardware information."""
        cpu_counts = Counter()
        ram_distribution = []
        
        for node in nodes:
            cpu_name = node.get('cpu_name', 'Unknown')
            cpu_cores = node.get('cpu_cores', 0)
            ram_total = node.get('ram_total', 0)
            
            cpu_counts[cpu_name] += 1
            if ram_total:
                ram_distribution.append(ram_total)
        
        return {
            'total_hardware_records': len(nodes),
            'cpu_distribution': dict(cpu_counts),
            'ram_stats': {
                'total_ram_gb': sum(ram_distribution) / (1024**3) if ram_distribution else 0,
                'avg_ram_gb': (sum(ram_distribution) / len(ram_distribution) / (1024**3)) if ram_distribution else 0,
                'min_ram_gb': min(ram_distribution) / (1024**3) if ram_distribution else 0,
                'max_ram_gb': max(ram_distribution) / (1024**3) if ram_distribution else 0
            }
        }
    
    def aggregate_assets(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate asset information."""
        status_counts = Counter()
        
        for node in nodes:
            status = node.get('status', 'Unknown')
            status_counts[status] += 1
        
        return {
            'total_assets': len(nodes),
            'status_distribution': dict(status_counts)
        }
    
    def aggregate_assetgroups(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate asset group information."""
        group_names = [node.get('name', 'Unknown') for node in nodes]
        
        return {
            'total_groups': len(nodes),
            'group_names': group_names
        }

    def aggregate_all(self) -> Dict[str, Any]:
        """Aggregate all node types."""
        nodes_path = self.extracted_data_path / 'nodes'
        
        if not nodes_path.exists():
            logger.error(f"Nodes path not found: {nodes_path}")
            return {}
        
        aggregations = {
            'timestamp': datetime.now().isoformat(),
            'source_path': str(self.extracted_data_path)
        }
        
        # Define node types and their aggregation functions
        node_types = {
            'host': self.aggregate_hosts,
            'software': self.aggregate_software,
            'vulnerability': self.aggregate_vulnerabilities,
            'hardware': self.aggregate_hardware,
            'asset': self.aggregate_assets,
            'assetgroup': self.aggregate_assetgroups
        }
        
        for node_type, aggregator_func in node_types.items():
            node_file = nodes_path / f"{node_type}_nodes.json"
            
            if node_file.exists():
                logger.info(f"Aggregating {node_type} nodes from {node_file}")
                nodes = self.load_nodes(node_file)
                
                if nodes:
                    aggregations[node_type] = aggregator_func(nodes)
                    logger.info(f"Aggregated {len(nodes)} {node_type} nodes")
                else:
                    logger.warning(f"No nodes found in {node_file}")
            else:
                logger.warning(f"Node file not found: {node_file}")
        
        self.aggregations = aggregations
        return aggregations
    
    def save_aggregations(self, session_name: str = None):
        """Save aggregations to output folder."""
        if not self.aggregations:
            logger.error("No aggregations to save. Run aggregate_all() first.")
            return
        
        # Create session folder
        if session_name is None:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        session_path = self.output_path / session_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete aggregation
        complete_file = session_path / 'complete_aggregation.json'
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregations, f, indent=2)
        logger.info(f"Saved complete aggregation to {complete_file}")
        
        # Save individual aggregations
        for node_type, data in self.aggregations.items():
            if node_type not in ['timestamp', 'source_path']:
                individual_file = session_path / f"{node_type}_aggregation.json"
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved {node_type} aggregation to {individual_file}")
        
        # Create summary report
        self._create_summary_report(session_path)
        
        logger.info(f"All aggregations saved to {session_path}")
        return session_path
    
    def _create_summary_report(self, session_path: Path):
        """Create a human-readable summary report."""
        summary_lines = [
            "=" * 80,
            "DATA AGGREGATION SUMMARY REPORT",
            "=" * 80,
            f"Generated: {self.aggregations.get('timestamp', 'Unknown')}",
            f"Source: {self.aggregations.get('source_path', 'Unknown')}",
            "=" * 80,
            ""
        ]
        
        # Host summary
        if 'host' in self.aggregations:
            host_data = self.aggregations['host']
            summary_lines.extend([
                "HOSTS / OPERATING SYSTEMS",
                "-" * 80,
                f"Total Hosts: {host_data.get('total_hosts', 0)}",
                "",
                "OS Distribution:"
            ])
            for os_name, count in host_data.get('os_distribution', {}).items():
                summary_lines.append(f"  - {os_name}: {count}")
            summary_lines.append("")
        
        # Software summary
        if 'software' in self.aggregations:
            sw_data = self.aggregations['software']
            summary_lines.extend([
                "SOFTWARE PACKAGES",
                "-" * 80,
                f"Total Packages: {sw_data.get('total_packages', 0)}",
                f"Unique Packages: {sw_data.get('unique_packages', 0)}",
                f"Multi-Version Packages: {sw_data.get('multi_version_count', 0)}",
                "",
                "Top 10 Packages:"
            ])
            top_packages = list(sw_data.get('top_50_packages', {}).items())[:10]
            for pkg_name, count in top_packages:
                summary_lines.append(f"  - {pkg_name}: {count}")
            summary_lines.append("")
        
        # Vulnerability summary
        if 'vulnerability' in self.aggregations:
            vuln_data = self.aggregations['vulnerability']
            summary_lines.extend([
                "VULNERABILITIES",
                "-" * 80,
                f"Total Vulnerabilities: {vuln_data.get('total_vulnerabilities', 0)}",
                f"Unique CVEs: {vuln_data.get('unique_cves', 0)}",
                "",
                "Severity Distribution:"
            ])
            for severity, count in vuln_data.get('severity_distribution', {}).items():
                summary_lines.append(f"  - {severity}: {count}")
            summary_lines.append("")
        
        # Asset summary
        if 'asset' in self.aggregations:
            asset_data = self.aggregations['asset']
            summary_lines.extend([
                "ASSETS",
                "-" * 80,
                f"Total Assets: {asset_data.get('total_assets', 0)}",
                ""
            ])
        
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
        output_path = Path('aggregated_data')
        output_path.mkdir(exist_ok=True)
        
        # Create aggregator
        aggregator = DataAggregator(extracted_data_path, output_path)
        
        # Run aggregation
        logger.info("Starting data aggregation...")
        aggregations = aggregator.aggregate_all()
        
        if aggregations:
            # Save results
            session_path = aggregator.save_aggregations()
            logger.info(f"Aggregation complete! Results saved to {session_path}")
            
            # Create .current_session file
            current_session_file = output_path / '.current_session'
            with open(current_session_file, 'w') as f:
                f.write(session_path.name)
            
            return 0
        else:
            logger.error("Aggregation failed - no data processed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during aggregation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
