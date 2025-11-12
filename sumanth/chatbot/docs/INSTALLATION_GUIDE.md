# Installation Guide

## Installing Packages

### Method 1: Using requirements.txt (Recommended)

This is the simplest method using pip:

```bash
pip install -r requirements.txt
```

### Method 2: Using environment.yml (Conda)

If you prefer using conda:

```bash
conda env create -f environment.yml
conda activate practice
```

## Step-by-Step Instructions

### Using requirements.txt

1. **Open Terminal/Command Prompt**

2. **Navigate to project directory**:
   ```bash
   cd C:\Users\HP\OneDrive\Desktop\chatbot
   ```

3. **Create virtual environment (recommended)**:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate it
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

4. **Install packages**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   pip list
   ```

### Using environment.yml (Conda)

1. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate environment**:
   ```bash
   conda activate practice
   ```

3. **Verify installation**:
   ```bash
   conda list
   ```

## Quick Installation

### For pip users:
```bash
pip install -r requirements.txt
```

### For conda users:
```bash
conda env create -f environment.yml
conda activate practice
```

## Package List

### Core Dependencies
- gradio - Web interface
- langchain - LLM framework
- langchain-community - Community integrations
- langchain-text-splitters - Text splitting utilities
- faiss-cpu - Vector database
- sentence-transformers - Embeddings
- openai - OpenAI API client

### Web Scraping
- beautifulsoup4 - HTML parsing
- requests - HTTP library
- pyyaml - YAML parser

### Document Processing
- PyPDF2 - PDF reading
- python-docx - Word document processing
- reportlab - PDF generation

### Development Tools (Optional)
- pytest - Testing framework
- black - Code formatter
- flake8 - Linter

## Troubleshooting

### "pip: command not found"
- Install Python from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation

### "Package installation failed"
- Update pip first: `python -m pip install --upgrade pip`
- Try installing packages individually to identify the issue

### "Permission denied"
- Use `--user` flag: `pip install -r requirements.txt --user`
- Or use virtual environment (recommended)

### "Python version too old"
- Requires Python 3.8 or higher
- Check version: `python --version`
- Update Python if needed

## Verify Installation

After installation, verify key packages:

```bash
python -c "import gradio; print('Gradio: OK')"
python -c "import langchain; print('LangChain: OK')"
python -c "import openai; print('OpenAI: OK')"
python -c "import faiss; print('FAISS: OK')"
python -c "import reportlab; print('ReportLab: OK')"
```

## Next Steps

After installation:
1. Configure API key in `config.yaml`
2. Set website URL in `config.yaml` (already set to https://www.data-legos.com)
3. Run: `python scripts/scrape_to_llm.py`
