# Dependency Check Results

## How to Check Dependencies

Run the dependency checker:
```bash
python check_dependencies.py
```

## Required Packages

All these packages should be installed:

### Core Dependencies
- gradio
- langchain
- langchain-community
- langchain-text-splitters
- faiss-cpu
- sentence-transformers
- openai

### Web Scraping
- beautifulsoup4
- requests
- urllib3
- pyyaml

### Document Processing
- PyPDF2
- python-docx
- reportlab

## Installation

If any packages are missing, install them:

```bash
pip install -r requirements.txt
```

## Common Issues

### "No module named 'config'"
- This is fixed - config is now in `src/chatbot/core/config.py`
- Make sure you're running scripts from project root
- Scripts automatically add `src/` to Python path

### "No module named 'gradio'"
- Install: `pip install gradio`
- Or install all: `pip install -r requirements.txt`

### Import Errors After Restructuring
- All imports have been updated to use `chatbot.` prefix
- Make sure you're using the scripts in `scripts/` folder
- They automatically set up the Python path correctly

## Verification

After installation, verify:
```bash
python check_dependencies.py
```

All packages should show `[OK]`.

