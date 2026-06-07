#!/usr/bin/env python3
"""
Script to analyze Neo4j query data using Presidio
Processes neo4j_query_table_data_2026-2-16.json and saves results to a separate folder
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("WARNING: Presidio not installed. Install with: pip install presidio-analyzer presidio-anonymizer")

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f'neo4j_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_text_with_presidio(analyzer, text, entities_to_detect):
    """Analyze text using Presidio and return detected entities"""
    if not text or not isinstance(text, str):
        return []
    
    try:
        results = analyzer.analyze(
            text=text,
            language='en',
            entities=entities_to_detect
        )
        return results
    except Exception as e:
        logging.error(f"Error analyzing text: {e}")
        return []

def process_neo4j_data(input_file, output_dir):
    """Process the Neo4j JSON data file with Presidio"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting analysis of {input_file}")
    
    # Check if Presidio is available
    if not PRESIDIO_AVAILABLE:
        logger.error("Presidio is not installed. Please install it first.")
        return False
    
    # Initialize Presidio analyzer
    logger.info("Initializing Presidio analyzer...")
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
    }
    
    try:
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    except Exception as e:
        logger.warning(f"Could not initialize spaCy model: {e}")
        logger.info("Falling back to simple analyzer without NLP")
        analyzer = AnalyzerEngine()
    
    # Entities to detect (PCI/PII focused)
    entities_to_detect = [
        "CREDIT_CARD",
        "US_SSN",
        "PERSON",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "US_BANK_NUMBER",
        "IBAN_CODE",
        "US_DRIVER_LICENSE",
        "US_PASSPORT",
        "DATE_TIME",
        "LOCATION",
        "ORGANIZATION",
        "IP_ADDRESS",
        "URL"
    ]
    
    # Load input JSON file
    logger.info(f"Loading data from {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return False
    
    logger.info(f"Loaded {len(data)} records from JSON file")
    
    # Process each record
    all_results = []
    entity_summary = {}
    
    for idx, record in enumerate(data):
        logger.info(f"Processing record {idx + 1}/{len(data)}: {record.get('c.id', 'Unknown')}")
        
        record_results = {
            'record_id': record.get('c.id', f'record_{idx}'),
            'record_name': record.get('c.name', ''),
            'fields_analyzed': {},
            'total_entities_found': 0,
            'entity_types_found': []
        }
        
        # Analyze each field in the record
        for field_name, field_value in record.items():
            if isinstance(field_value, str):
                # Analyze string fields
                entities = analyze_text_with_presidio(analyzer, field_value, entities_to_detect)
                
                if entities:
                    record_results['fields_analyzed'][field_name] = {
                        'original_text': field_value[:200] + '...' if len(field_value) > 200 else field_value,
                        'entities': []
                    }
                    
                    for entity in entities:
                        entity_info = {
                            'type': entity.entity_type,
                            'text': field_value[entity.start:entity.end],
                            'start': entity.start,
                            'end': entity.end,
                            'score': round(entity.score, 3)
                        }
                        record_results['fields_analyzed'][field_name]['entities'].append(entity_info)
                        record_results['total_entities_found'] += 1
                        
                        # Update entity summary
                        entity_type = entity.entity_type
                        if entity_type not in entity_summary:
                            entity_summary[entity_type] = 0
                        entity_summary[entity_type] += 1
                        
                        if entity_type not in record_results['entity_types_found']:
                            record_results['entity_types_found'].append(entity_type)
                        
                        logger.info(f"  Found {entity_type} in field '{field_name}': {field_value[entity.start:entity.end]} (score: {entity.score:.3f})")
            
            elif isinstance(field_value, list):
                # Analyze list fields (like CPEs)
                for list_idx, item in enumerate(field_value):
                    if isinstance(item, str):
                        entities = analyze_text_with_presidio(analyzer, item, entities_to_detect)
                        
                        if entities:
                            list_field_name = f"{field_name}[{list_idx}]"
                            record_results['fields_analyzed'][list_field_name] = {
                                'original_text': item,
                                'entities': []
                            }
                            
                            for entity in entities:
                                entity_info = {
                                    'type': entity.entity_type,
                                    'text': item[entity.start:entity.end],
                                    'start': entity.start,
                                    'end': entity.end,
                                    'score': round(entity.score, 3)
                                }
                                record_results['fields_analyzed'][list_field_name]['entities'].append(entity_info)
                                record_results['total_entities_found'] += 1
                                
                                # Update entity summary
                                entity_type = entity.entity_type
                                if entity_type not in entity_summary:
                                    entity_summary[entity_type] = 0
                                entity_summary[entity_type] += 1
                                
                                if entity_type not in record_results['entity_types_found']:
                                    record_results['entity_types_found'].append(entity_type)
        
        all_results.append(record_results)
    
    # Generate summary report
    summary_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'input_file': os.path.basename(input_file),
        'total_records_processed': len(data),
        'records_with_entities': sum(1 for r in all_results if r['total_entities_found'] > 0),
        'total_entities_found': sum(r['total_entities_found'] for r in all_results),
        'entity_type_summary': entity_summary,
        'entity_types_detected': list(entity_summary.keys())
    }
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_results_file = os.path.join(output_dir, f'presidio_detailed_results_{timestamp}.json')
    
    logger.info(f"Saving detailed results to {detailed_results_file}")
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary_report,
            'detailed_results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    # Save summary report
    summary_file = os.path.join(output_dir, f'presidio_summary_{timestamp}.json')
    logger.info(f"Saving summary report to {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    # Generate simple HTML report
    html_file = os.path.join(output_dir, f'presidio_report_{timestamp}.html')
    logger.info(f"Generating HTML report to {html_file}")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Presidio Analysis Report - Neo4j Data</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .stat {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .stat-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .stat-value {{
            font-size: 24px;
            color: #2c3e50;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .entity-badge {{
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .CREDIT_CARD {{ background-color: #e74c3c; color: white; }}
        .US_SSN {{ background-color: #e67e22; color: white; }}
        .PERSON {{ background-color: #3498db; color: white; }}
        .PHONE_NUMBER {{ background-color: #9b59b6; color: white; }}
        .EMAIL_ADDRESS {{ background-color: #1abc9c; color: white; }}
        .LOCATION {{ background-color: #16a085; color: white; }}
        .ORGANIZATION {{ background-color: #27ae60; color: white; }}
        .DATE_TIME {{ background-color: #95a5a6; color: white; }}
        .IP_ADDRESS {{ background-color: #c0392b; color: white; }}
        .URL {{ background-color: #8e44ad; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Presidio PII/PCI Analysis Report</h1>
        <p><strong>Analysis Date:</strong> {summary_report['analysis_timestamp']}</p>
        <p><strong>Input File:</strong> {summary_report['input_file']}</p>
        
        <div class="summary-box">
            <h2>Summary Statistics</h2>
            <div class="stat">
                <div class="stat-label">Total Records</div>
                <div class="stat-value">{summary_report['total_records_processed']}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Records with Entities</div>
                <div class="stat-value">{summary_report['records_with_entities']}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total Entities Found</div>
                <div class="stat-value">{summary_report['total_entities_found']}</div>
            </div>
        </div>
        
        <h2>Entity Type Distribution</h2>
        <table>
            <thead>
                <tr>
                    <th>Entity Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
    
    total_entities = summary_report['total_entities_found']
    for entity_type, count in sorted(entity_summary.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_entities * 100) if total_entities > 0 else 0
        html_content += f"""
                <tr>
                    <td><span class="entity-badge {entity_type}">{entity_type}</span></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
        
        <h2>Detailed Findings by Record</h2>
        <table>
            <thead>
                <tr>
                    <th>Record ID</th>
                    <th>Entities Found</th>
                    <th>Entity Types</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for result in all_results:
        if result['total_entities_found'] > 0:
            entity_badges = ''.join([f'<span class="entity-badge {et}">{et}</span>' for et in result['entity_types_found']])
            html_content += f"""
                <tr>
                    <td>{result['record_id']}</td>
                    <td>{result['total_entities_found']}</td>
                    <td>{entity_badges}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
        
        <p style="margin-top: 40px; color: #7f8c8d; font-size: 12px;">
            Generated by Presidio PII/PCI Analysis Tool
        </p>
    </div>
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Total records processed: {summary_report['total_records_processed']}")
    logger.info(f"Records with entities: {summary_report['records_with_entities']}")
    logger.info(f"Total entities found: {summary_report['total_entities_found']}")
    logger.info(f"\nEntity types detected:")
    for entity_type, count in sorted(entity_summary.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {entity_type}: {count}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - Detailed results: {os.path.basename(detailed_results_file)}")
    logger.info(f"  - Summary report: {os.path.basename(summary_file)}")
    logger.info(f"  - HTML report: {os.path.basename(html_file)}")
    logger.info("="*60)
    
    return True

def main():
    """Main entry point"""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_dir, 'neo4j_query_table_data_2026-2-16.json')
    output_dir = os.path.join(project_dir, 'presidio_results', f'neo4j_analysis_{datetime.now().strftime("%Y%m%d")}')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return 1
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process the data
    success = process_neo4j_data(input_file, output_dir)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
