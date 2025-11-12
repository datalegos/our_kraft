# Quick Start Guide

## Complete Pipeline: Website → PDF → Embeddings → Chatbot

### One Command to Rule Them All

```bash
python scripts/scrape_to_llm.py https://your-website.com
```

That's it! This single command will:
1. ✅ Scrape the website
2. ✅ Create a PDF
3. ✅ Generate embeddings
4. ✅ Launch the chatbot

## Prerequisites

1. **Install dependencies**:
   ```bash
   conda env create -f environment.yml
   conda activate practice
   ```

2. **Configure API key** in `config.yaml`:
   ```yaml
   openai:
     api_key: "your-api-key-here"
   ```

## Usage Examples

### Basic Usage
```bash
# Scrape website and launch chatbot
python scripts/scrape_to_llm.py https://data-legos.com
```

### Scrape More Pages
```bash
python scripts/scrape_to_llm.py https://example.com --max-pages 20
```

### Just Create PDF (Skip Chatbot)
```bash
python scripts/scrape_to_llm.py https://example.com --skip-embeddings --skip-chatbot
```

### Use Existing PDF
```bash
python scripts/scrape_to_llm.py https://example.com --skip-scrape --skip-pdf
```

## What Happens

1. **Scraping**: Extracts text from website pages
2. **PDF Creation**: Converts scraped content to formatted PDF
3. **Embeddings**: Creates vector embeddings from PDF
4. **Chatbot**: Launches interactive chatbot interface

## Output

- **PDF**: Saved in `documents/` folder
- **Embeddings**: Saved in `faiss_index_bge_small/`
- **Chatbot**: Opens in browser at `http://127.0.0.1:7860`

## Troubleshooting

**"No content scraped"**
- Check URL is accessible
- Try increasing `--max-pages`

**"PDF creation failed"**
- Install reportlab: `pip install reportlab`

**"Chatbot won't start"**
- Check OpenAI API key in `config.yaml`
- Ensure embeddings were generated

## Full Documentation

- See `SCRAPE_TO_LLM_GUIDE.md` for detailed guide
- See `README.md` for general information
- See `PROJECT_STRUCTURE.md` for code organization

