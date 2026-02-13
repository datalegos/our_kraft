"""
CSV Converter Utility
Converts JSON files to CSV format while maintaining the same folder structure
"""
import json
import csv
from pathlib import Path
from typing import Any, Dict, List


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Handle lists - if list of dicts, create separate rows
            if len(v) > 0 and isinstance(v[0], dict):
                # This will be handled separately
                items.append((new_key, v))
            else:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def json_to_csv_simple(json_data: Any, output_path: Path):
    """Convert simple JSON structure to CSV"""
    rows = []
    
    if isinstance(json_data, dict):
        # If it's a single dict, convert to single-row CSV
        if 'data' in json_data and 'affected_items' in json_data['data']:
            # Wazuh API format - extract affected_items
            items = json_data['data']['affected_items']
            if items and isinstance(items[0], dict):
                # List of dictionaries - each becomes a row
                rows = items
            else:
                # Single dict - flatten it
                rows = [flatten_dict(json_data)]
        else:
            # Try to flatten and convert
            rows = [flatten_dict(json_data)]
    elif isinstance(json_data, list):
        # List of items
        if len(json_data) > 0 and isinstance(json_data[0], dict):
            rows = json_data
        else:
            # Simple list - convert to rows with 'value' column
            rows = [{'value': item} for item in json_data]
    else:
        # Simple value
        rows = [{'value': json_data}]
    
    # Write to CSV
    if rows:
        # Get all unique keys from all rows
        all_keys = set()
        for row in rows:
            if isinstance(row, dict):
                all_keys.update(row.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for row in rows:
                    if isinstance(row, dict):
                        # Convert non-string values to strings
                        clean_row = {k: str(v) if not isinstance(v, (str, int, float, bool)) or v is None else v 
                                    for k, v in row.items()}
                        writer.writerow(clean_row)
            else:
                # Empty data
                writer = csv.writer(f)
                writer.writerow(['value'])
                writer.writerow([str(json_data)])


def convert_json_to_csv(json_path: Path, csv_output_dir: Path, relative_path: Path = None):
    """Convert a single JSON file to CSV, maintaining folder structure"""
    try:
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Determine output CSV path
        if relative_path:
            csv_path = csv_output_dir / relative_path.with_suffix('.csv')
        else:
            csv_path = csv_output_dir / json_path.relative_to(csv_output_dir.parent).with_suffix('.csv')
        
        # Create parent directories
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to CSV
        json_to_csv_simple(json_data, csv_path)
        return True
        
    except Exception as e:
        print(f"Error converting {json_path} to CSV: {e}")
        return False


def convert_directory_to_csv(json_dir: Path, csv_output_dir: Path, base_path: Path = None):
    """Recursively convert all JSON files in a directory to CSV"""
    if base_path is None:
        base_path = json_dir
    
    json_files = list(json_dir.rglob('*.json'))
    converted = 0
    failed = 0
    
    print(f"Found {len(json_files)} JSON files to convert...")
    
    for json_file in json_files:
        # Calculate relative path from base
        relative_path = json_file.relative_to(base_path)
        
        if convert_json_to_csv(json_file, csv_output_dir, relative_path):
            converted += 1
        else:
            failed += 1
    
    print(f"Conversion complete: {converted} converted, {failed} failed")
    return converted, failed


def convert_collected_data_to_csv(collected_data_dir: Path, csv_output_dir: Path = None):
    """Convert collected data directory to CSV format"""
    if csv_output_dir is None:
        csv_output_dir = collected_data_dir.parent / f"{collected_data_dir.name}_csv"
    
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting JSON files from {collected_data_dir} to CSV in {csv_output_dir}...")
    
    converted, failed = convert_directory_to_csv(collected_data_dir, csv_output_dir, collected_data_dir)
    
    print(f"\nCSV conversion completed!")
    print(f"  Output directory: {csv_output_dir}")
    print(f"  Files converted: {converted}")
    print(f"  Files failed: {failed}")
    
    return csv_output_dir

