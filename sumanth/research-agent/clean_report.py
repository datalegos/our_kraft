"""
Utility script to clean existing report files by removing JSON objects.
"""

import os
import re
import json
from pathlib import Path

def clean_report_content(content: str) -> str:
    """Remove JSON objects from report content."""
    if not content:
        return ""
    
    # Try to find and extract JSON object
    json_start = content.rfind('{"title"')
    if json_start == -1:
        json_start = content.rfind('{\n  "title"')
    
    if json_start >= 0:
        try:
            # Extract JSON string
            json_str = content[json_start:]
            # Find the end of JSON
            brace_count = 0
            json_end = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > 0:
                json_str = json_str[:json_end]
                json_data = json.loads(json_str)
                # If JSON has well-formatted content, use it
                if 'content' in json_data and len(json_data['content']) > 100:
                    # Keep the header and replace content
                    header_end = content[:json_start].rfind('---')
                    if header_end > 0:
                        header = content[:header_end + 3].strip()
                        return f"{header}\n\n{json_data['content'].strip()}\n"
                    else:
                        return json_data['content'].strip()
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Remove JSON from content
    lines = content.split('\n')
    cleaned_lines = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('{') and '"title"' in stripped and '"content"' in stripped:
            in_json = True
            brace_count = stripped.count('{') - stripped.count('}')
            continue
        
        if in_json:
            brace_count += stripped.count('{') - stripped.count('}')
            if brace_count <= 0:
                in_json = False
            continue
        
        cleaned_lines.append(line)
    
    cleaned_content = '\n'.join(cleaned_lines).strip()
    
    # Remove trailing JSON patterns
    patterns = [
        r'\s*\{[^{}]*"title"[^{}]*"content"[^{}]*\}.*$',
        r'\s*\{[^}]*"title"[^}]*"content"[^}]*\}.*$',
        r'\s*\{.*?"title".*?"content".*?\}.*$',
    ]
    
    for pattern in patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL)
    
    return cleaned_content.strip()


def clean_all_reports():
    """Clean all report files in the reports directory."""
    reports_dir = Path("reports")
    
    if not reports_dir.exists():
        print("Reports directory not found!")
        return
    
    report_files = list(reports_dir.glob("*.md"))
    
    if not report_files:
        print("No report files found!")
        return
    
    print(f"Found {len(report_files)} report file(s).")
    
    for report_file in report_files:
        print(f"\nCleaning: {report_file.name}")
        
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned = clean_report_content(content)
            
            if cleaned != content:
                # Backup original
                backup_file = report_file.with_suffix('.md.bak')
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Write cleaned version
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
                print(f"  ✓ Cleaned and backed up to {backup_file.name}")
            else:
                print(f"  - No changes needed")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    clean_all_reports()

