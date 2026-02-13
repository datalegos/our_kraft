#!/usr/bin/env python3
"""
Presidio PCI Data Analysis Tool
Analyzes extracted database data for PII/PCI compliance using Microsoft Presidio
"""

import yaml
import json
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass, asdict
import concurrent.futures

# Presidio imports
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# Data processing imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

@dataclass
class PIIEntity:
    """Detected PII entity information"""
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    context: str = ""

@dataclass
class AnalysisResult:
    """Analysis result for a single record"""
    source_file: str
    database: str
    table: str
    record_id: str
    field_name: str
    original_text: str
    entities: List[PIIEntity]
    risk_score: float
    risk_level: str

@dataclass
class ComplianceReport:
    """Compliance summary report"""
    timestamp: str
    total_files_processed: int
    total_records_analyzed: int
    total_entities_found: int
    entity_counts: Dict[str, int]
    risk_distribution: Dict[str, int]
    database_summary: Dict[str, Dict]
    compliance_status: str

class PresidioAnalyzer:
    """Main Presidio analysis class"""
    
    def __init__(self, config_file: str = "config/presidio_config.yml"):
        """Initialize the analyzer with configuration"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_output_directory()
        self.analyzer = None
        self.results = []
        
        if not PRESIDIO_AVAILABLE:
            self.logger.error("Presidio is not installed. Please install: pip install presidio-analyzer presidio-anonymizer")
            sys.exit(1)
        
        self.setup_presidio()
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_file}' not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        log_level = getattr(logging, log_config['level'].upper())
        
        # Create logs directory
        log_file = log_config['file'].format(timestamp=self.timestamp)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        handlers = [logging.FileHandler(log_file)]
        if log_config['console']:
            handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Presidio PCI Analyzer initialized")
    
    def setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        output_dir = self.config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {output_dir}")
    
    def setup_presidio(self):
        """Initialize Presidio analyzer with custom recognizers"""
        try:
            # Create NLP engine
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
            
            # Create registry and add custom recognizers
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            
            # Add custom recognizers
            self.add_custom_recognizers(registry)
            
            # Create analyzer
            self.analyzer = AnalyzerEngine(
                registry=registry,
                nlp_engine=nlp_engine,
                supported_languages=[self.config['detection']['language']]
            )
            
            self.logger.info("Presidio analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Presidio: {e}")
            sys.exit(1)
    
    def add_custom_recognizers(self, registry: RecognizerRegistry):
        """Add custom PCI-specific recognizers"""
        custom_recognizers = self.config['detection'].get('custom_recognizers', {})
        
        for name, config in custom_recognizers.items():
            try:
                recognizer = PatternRecognizer(
                    supported_entity=name.upper(),
                    patterns=[{
                        "name": name,
                        "regex": config['pattern'],
                        "score": config['score']
                    }],
                    context=config.get('context', [])
                )
                
                registry.add_recognizer(recognizer)
                self.logger.info(f"Added custom recognizer: {name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to add custom recognizer {name}: {e}")
    
    def analyze_text(self, text: str, context: str = "") -> List[PIIEntity]:
        """Analyze text for PII entities"""
        try:
            # Run Presidio analysis
            results = self.analyzer.analyze(
                text=text,
                entities=self.config['detection']['entities'],
                language=self.config['detection']['language'],
                score_threshold=self.config['detection']['min_score']
            )
            
            # Convert to PIIEntity objects
            entities = []
            for result in results:
                entity = PIIEntity(
                    entity_type=result.entity_type,
                    text=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    context=self.extract_context(text, result.start, result.end)
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Analysis failed for text: {e}")
            return []
    
    def extract_context(self, text: str, start: int, end: int) -> str:
        """Extract context around detected entity"""
        if not self.config['reporting']['include_context']:
            return ""
        
        window = self.config['reporting']['context_window']
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        return text[context_start:context_end]
    
    def calculate_risk_score(self, entities: List[PIIEntity]) -> Tuple[float, str]:
        """Calculate risk score based on detected entities"""
        weights = self.config['risk_assessment']['entity_weights']
        thresholds = self.config['risk_assessment']['thresholds']
        
        total_score = 0
        max_possible = 0
        
        for entity in entities:
            weight = weights.get(entity.entity_type, 1)
            total_score += entity.score * weight
            max_possible += weight
        
        # Normalize score
        if max_possible > 0:
            normalized_score = total_score / max_possible
        else:
            normalized_score = 0
        
        # Determine risk level
        if normalized_score >= thresholds['critical']:
            risk_level = "CRITICAL"
        elif normalized_score >= thresholds['high']:
            risk_level = "HIGH"
        elif normalized_score >= thresholds['medium']:
            risk_level = "MEDIUM"
        elif normalized_score >= thresholds['low']:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return normalized_score, risk_level
    
    def process_json_file(self, file_path: str) -> List[AnalysisResult]:
        """Process a single JSON file from database extraction"""
        results = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            db_info = data.get('database_info', {})
            table_info = data.get('table_info', {})
            records = data.get('data', [])
            
            database = db_info.get('database', 'unknown')
            table = table_info.get('name', 'unknown')
            
            self.logger.info(f"Processing {len(records)} records from {database}.{table}")
            
            # Process each record
            for i, record in enumerate(records):
                record_id = str(record.get('id', i))
                
                # Analyze each field in the record
                for field_name, field_value in record.items():
                    if self.should_analyze_field(field_name, field_value):
                        text = str(field_value)
                        entities = self.analyze_text(text)
                        
                        if entities:  # Only create result if entities found
                            risk_score, risk_level = self.calculate_risk_score(entities)
                            
                            result = AnalysisResult(
                                source_file=os.path.basename(file_path),
                                database=database,
                                table=table,
                                record_id=record_id,
                                field_name=field_name,
                                original_text=text,
                                entities=entities,
                                risk_score=risk_score,
                                risk_level=risk_level
                            )
                            
                            results.append(result)
                            
                            if self.config['logging']['log_detections']:
                                self.logger.info(f"Found {len(entities)} entities in {database}.{table}.{field_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
        
        return results
    
    def should_analyze_field(self, field_name: str, field_value: Any) -> bool:
        """Determine if a field should be analyzed"""
        # Skip excluded fields
        if field_name in self.config['input']['exclude_fields']:
            return False
        
        # Skip non-string values that can't contain PII
        if not isinstance(field_value, (str, int, float)):
            return False
        
        # Skip empty or very short values
        text = str(field_value).strip()
        if len(text) < 3:
            return False
        
        # If specific fields are configured, only analyze those
        analyze_fields = self.config['input']['analyze_fields']
        if analyze_fields and field_name not in analyze_fields:
            return False
        
        return True
    
    def process_all_files(self) -> List[AnalysisResult]:
        """Process all files in the source directory"""
        source_dir = self.config['input']['source_directory']
        file_patterns = self.config['input']['file_patterns']
        
        all_results = []
        files_to_process = []
        
        # Find files to process
        for pattern in file_patterns:
            files_to_process.extend(Path(source_dir).glob(pattern))
        
        if not files_to_process:
            self.logger.warning(f"No files found in {source_dir} matching patterns {file_patterns}")
            return []
        
        self.logger.info(f"Processing {len(files_to_process)} files")
        
        # Process files
        for file_path in files_to_process:
            self.logger.info(f"Processing file: {file_path}")
            results = self.process_json_file(str(file_path))
            all_results.extend(results)
        
        return all_results
    
    def generate_compliance_report(self, results: List[AnalysisResult]) -> ComplianceReport:
        """Generate compliance summary report"""
        entity_counts = {}
        risk_distribution = {"MINIMAL": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        database_summary = {}
        
        for result in results:
            # Count entities
            for entity in result.entities:
                entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1
            
            # Count risk levels
            risk_distribution[result.risk_level] += 1
            
            # Database summary
            db_key = f"{result.database}.{result.table}"
            if db_key not in database_summary:
                database_summary[db_key] = {
                    "total_records": 0,
                    "entities_found": 0,
                    "risk_levels": {"MINIMAL": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
                }
            
            database_summary[db_key]["total_records"] += 1
            database_summary[db_key]["entities_found"] += len(result.entities)
            database_summary[db_key]["risk_levels"][result.risk_level] += 1
        
        # Determine overall compliance status
        total_high_risk = risk_distribution["HIGH"] + risk_distribution["CRITICAL"]
        if total_high_risk > 0:
            compliance_status = "HIGH_RISK"
        elif risk_distribution["MEDIUM"] > 0:
            compliance_status = "MEDIUM_RISK"
        else:
            compliance_status = "LOW_RISK"
        
        return ComplianceReport(
            timestamp=self.timestamp,
            total_files_processed=len(set(r.source_file for r in results)),
            total_records_analyzed=len(results),
            total_entities_found=sum(len(r.entities) for r in results),
            entity_counts=entity_counts,
            risk_distribution=risk_distribution,
            database_summary=database_summary,
            compliance_status=compliance_status
        )
    
    def save_results(self, results: List[AnalysisResult], compliance_report: ComplianceReport):
        """Save analysis results and reports"""
        output_dir = self.config['output']['directory']
        
        # Save detailed results
        if self.config['output']['detailed_reports']:
            detailed_file = os.path.join(output_dir, f"detailed_pci_analysis_{self.timestamp}.json")
            with open(detailed_file, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2, default=str)
            self.logger.info(f"Saved detailed results to {detailed_file}")
        
        # Save compliance report
        compliance_file = os.path.join(output_dir, self.config['output']['compliance_report'].format(timestamp=self.timestamp))
        with open(compliance_file, 'w') as f:
            json.dump(asdict(compliance_report), f, indent=2, default=str)
        self.logger.info(f"Saved compliance report to {compliance_file}")
        
        # Save CSV export if requested
        if 'csv' in self.config['reporting']['export_formats']:
            self.save_csv_report(results, output_dir)
        
        # Generate HTML dashboard if requested
        if self.config['output']['summary_report'] and 'html' in self.config['reporting']['export_formats']:
            self.generate_html_dashboard(results, compliance_report, output_dir)
    
    def save_csv_report(self, results: List[AnalysisResult], output_dir: str):
        """Save results as CSV file"""
        if not PANDAS_AVAILABLE:
            self.logger.warning("Pandas not available, skipping CSV export")
            return
        
        # Flatten results for CSV
        csv_data = []
        for result in results:
            for entity in result.entities:
                csv_data.append({
                    'source_file': result.source_file,
                    'database': result.database,
                    'table': result.table,
                    'record_id': result.record_id,
                    'field_name': result.field_name,
                    'entity_type': entity.entity_type,
                    'entity_text': entity.text,
                    'confidence_score': entity.score,
                    'risk_score': result.risk_score,
                    'risk_level': result.risk_level,
                    'context': entity.context
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(output_dir, f"pci_analysis_{self.timestamp}.csv")
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved CSV report to {csv_file}")
    
    def generate_html_dashboard(self, results: List[AnalysisResult], compliance_report: ComplianceReport, output_dir: str):
        """Generate HTML dashboard report"""
        try:
            from jinja2 import Template
            
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>PCI Compliance Analysis Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .summary { display: flex; gap: 20px; margin: 20px 0; }
                    .card { background-color: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }
                    .risk-critical { background-color: #ffebee; border-color: #f44336; }
                    .risk-high { background-color: #fff3e0; border-color: #ff9800; }
                    .risk-medium { background-color: #f3e5f5; border-color: #9c27b0; }
                    .risk-low { background-color: #e8f5e8; border-color: #4caf50; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>PCI Compliance Analysis Dashboard</h1>
                    <p>Generated: {{ compliance_report.timestamp }}</p>
                    <p>Status: <strong>{{ compliance_report.compliance_status }}</strong></p>
                </div>
                
                <div class="summary">
                    <div class="card">
                        <h3>Files Processed</h3>
                        <h2>{{ compliance_report.total_files_processed }}</h2>
                    </div>
                    <div class="card">
                        <h3>Records Analyzed</h3>
                        <h2>{{ compliance_report.total_records_analyzed }}</h2>
                    </div>
                    <div class="card">
                        <h3>PII Entities Found</h3>
                        <h2>{{ compliance_report.total_entities_found }}</h2>
                    </div>
                </div>
                
                <h2>Entity Types Detected</h2>
                <table>
                    <tr><th>Entity Type</th><th>Count</th></tr>
                    {% for entity_type, count in compliance_report.entity_counts.items() %}
                    <tr><td>{{ entity_type }}</td><td>{{ count }}</td></tr>
                    {% endfor %}
                </table>
                
                <h2>Risk Distribution</h2>
                <table>
                    <tr><th>Risk Level</th><th>Count</th></tr>
                    {% for risk_level, count in compliance_report.risk_distribution.items() %}
                    <tr><td>{{ risk_level }}</td><td>{{ count }}</td></tr>
                    {% endfor %}
                </table>
                
                <h2>Database Summary</h2>
                <table>
                    <tr><th>Database.Table</th><th>Records</th><th>Entities Found</th><th>High Risk</th></tr>
                    {% for db_table, summary in compliance_report.database_summary.items() %}
                    <tr>
                        <td>{{ db_table }}</td>
                        <td>{{ summary.total_records }}</td>
                        <td>{{ summary.entities_found }}</td>
                        <td>{{ summary.risk_levels.HIGH + summary.risk_levels.CRITICAL }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </body>
            </html>
            """
            
            template = Template(html_template)
            html_content = template.render(compliance_report=compliance_report)
            
            html_file = os.path.join(output_dir, self.config['output']['dashboard_report'].format(timestamp=self.timestamp))
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Generated HTML dashboard: {html_file}")
            
        except ImportError:
            self.logger.warning("Jinja2 not available, skipping HTML dashboard generation")
        except Exception as e:
            self.logger.error(f"Failed to generate HTML dashboard: {e}")
    
    def run(self):
        """Main execution method"""
        self.logger.info("Starting PCI data analysis with Presidio")
        
        # Process all files
        results = self.process_all_files()
        
        if not results:
            self.logger.warning("No PII entities found in any files")
            return
        
        self.logger.info(f"Analysis completed. Found {len(results)} records with PII entities")
        
        # Generate compliance report
        compliance_report = self.generate_compliance_report(results)
        
        # Save results
        self.save_results(results, compliance_report)
        
        # Print summary
        self.print_summary(compliance_report)
    
    def print_summary(self, compliance_report: ComplianceReport):
        """Print analysis summary to console"""
        print("\n" + "="*60)
        print("           PCI COMPLIANCE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Timestamp: {compliance_report.timestamp}")
        print(f"Files Processed: {compliance_report.total_files_processed}")
        print(f"Records Analyzed: {compliance_report.total_records_analyzed}")
        print(f"PII Entities Found: {compliance_report.total_entities_found}")
        print(f"Compliance Status: {compliance_report.compliance_status}")
        print("\nEntity Types Detected:")
        for entity_type, count in compliance_report.entity_counts.items():
            print(f"  {entity_type}: {count}")
        print("\nRisk Distribution:")
        for risk_level, count in compliance_report.risk_distribution.items():
            print(f"  {risk_level}: {count}")
        print("="*60)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Presidio PCI Data Analysis Tool")
    parser.add_argument("-c", "--config", default="config/presidio_config.yml",
                       help="Configuration file path (default: config/presidio_config.yml)")
    
    args = parser.parse_args()
    
    analyzer = PresidioAnalyzer(args.config)
    analyzer.run()

if __name__ == "__main__":
    main()