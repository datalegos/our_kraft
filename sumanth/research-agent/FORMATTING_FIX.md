# Report Formatting Fix

## Problem Identified

The report files were including both:
1. Properly formatted markdown content
2. A raw JSON object at the end (like `{"title": "...", "content": "..."}`)

This made the reports look unprofessional and hard to read.

## Solution Implemented

### 1. **Content Cleaning Function** (`_clean_content`)
   - Detects JSON objects in the agent's output
   - Extracts well-formatted content from JSON if present
   - Removes raw JSON objects from the final output
   - Handles both single-line and multi-line JSON

### 2. **Improved Auto-Save**
   - The `_auto_save_report` function now uses `_clean_content` to clean the output
   - Extracts titles from content when possible
   - Ensures only clean markdown is saved

### 3. **Better Agent Prompt**
   - Updated instructions to encourage proper use of `save_report` tool
   - Clearer guidance on markdown formatting
   - Instructions to avoid including JSON in text output

## Result

✅ Reports are now clean and properly formatted
✅ No JSON objects in the final output
✅ Professional markdown formatting
✅ Proper headings and structure

## Testing

Run the test script to verify:
```bash
conda activate agents
python test_agent.py
```

Check the generated report in the `reports/` folder - it should be clean markdown without any JSON objects.

