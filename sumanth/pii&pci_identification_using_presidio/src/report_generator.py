#!/usr/bin/env python3
"""
HTML Report Generator for PCI Analysis Results
Creates comprehensive HTML reports from Presidio analysis results
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

class ReportGenerator:
    """Generate HTML reports from PCI analysis results"""
    
    def __init__(self, results_dir: str = "presidio_results"):
        self.results_dir = results_dir
        self.logger = logging.getLogger(__name__)
    
    def generate_html_report(self, summary_file: str) -> str:
        """Generate comprehensive HTML report"""
        
        # Load summary data
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Generate HTML
        html_content = self.create_html_template(summary)
        
        # Save HTML report
        report_file = os.path.join(
            self.results_dir, 
            "reports", 
            f"{Path(summary_file).stem}_report.html"
        )
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated HTML report: {report_file}")
        return report_file
    
    def create_html_template(self, summary: Dict[str, Any]) -> str:
        """Create HTML report template"""
        
        stats = summary['summary_statistics']
        entities = summary['entity_distribution']
        risks = summary['risk_distribution']
        tables = summary['table_analyses']
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCI Data Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-s