# Scrape to LLM Pipeline Guide

Complete guide for the automated pipeline: Website Scraping → PDF Generation → Embeddings → Chatbot

## Overview

The `scrape_to_llm.py` script automates the entire process:
1. **Scrape** content from a website
2. **Create PDF** from scraped content
3. **Generate embeddings** from PDF
4. **Launch chatbot** ready to answer questions

## Quick Start

### Basic Usage

```bash
python scripts/scrape_to_llm.py https://example.com
```

This will:
- Scrape up to 10 pages from the website
- Create a PDF in `documents/` folder
- Generate embeddings
- Launch the chatbot

### Advanced Usage

```bash
# Scrape more pages
python scripts/scrape_to_llm.py https://example.com --max-pages 20

# Custom output directory
python scripts/scrape_to_llm.py https://example.com --output-dir my_docs

# Skip steps (useful for re-running)
python scripts/scrape_to_llm.py https://example.com --skip-scrape --skip-pdf
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `url` | Website URL to scrape (required) | - |
| `--max-pages` | Maximum pages to scrape | 10 |
| `--output-dir` | PDF output directory | `documents` |
| `--skip-scrape` | Skip scraping step | False |
| `--skip-pdf` | Skip PDF creation | False |
| `--skip-embeddings` | Skip embedding generation | False |
| `--skip-chatbot` | Skip chatbot launch | False |

## Examples

### Example 1: Full Pipeline
```bash
python scripts/scrape_to_llm.py https://data-legos.com
```

### Example 2: Scrape and Create PDF Only
```bash
python scripts/scrape_to_llm.py https://example.com --skip-embeddings --skip-chatbot
```

### Example 3: Generate Embeddings from Existing PDF
```bash
python scripts/scrape_to_llm.py https://example.com --skip-scrape --skip-pdf --skip-chatbot
```

### Example 4: Just Launch Chatbot (after setup)
```bash
python scripts/scrape_to_llm.py https://example.com --skip-scrape --skip-pdf --skip-embeddings
```

## Step-by-Step Process

### Step 1: Website Scraping
- Uses the `Scraper` class
- Follows links within the same domain
- Respects rate limits and timeouts
- Extracts clean text content

### Step 2: PDF Generation
- Converts scraped text to PDF
- Formats with proper paragraphs
- Includes metadata (URL, timestamp)
- Saves to `documents/` folder

### Step 3: Embedding Generation
- Processes PDF using `DocumentProcessor`
- Creates semantic chunks
- Generates embeddings with HuggingFace
- Builds FAISS index

### Step 4: Chatbot Launch
- Loads FAISS index
- Initializes all optimization modules
- Launches Gradio interface
- Ready to answer questions

## Output Files

### PDF Files
- Location: `documents/`
- Naming: `{domain}_{timestamp}.pdf`
- Example: `data_legos_com_20241211_143022.pdf`

### Embeddings
- Location: `faiss_index_bge_small/`
- Files: `index.faiss`, `index.pkl`

### Logs
- Location: `chatbot.log`
- Contains all pipeline steps

## Troubleshooting

### "No content scraped"
- Check if website is accessible
- Verify URL is correct
- Check if website blocks scrapers
- Try increasing `--max-pages`

### "PDF creation failed"
- Ensure `reportlab` is installed: `pip install reportlab`
- Check disk space
- Verify write permissions

### "Embedding generation failed"
- Ensure PDF file exists
- Check if FAISS index directory is writable
- Verify embedding model is available

### "Chatbot launch failed"
- Check if embeddings were generated
- Verify `config.yaml` has OpenAI API key
- Check if port 7860 is available

## Integration with Existing Workflow

### Using with Existing Documents
If you already have PDFs in `documents/` folder:

```bash
# Just generate embeddings and launch
python scripts/scrape_to_llm.py https://example.com --skip-scrape --skip-pdf
```

### Updating Existing Index
To update embeddings with new content:

```bash
# Scrape new content and regenerate
python scripts/scrape_to_llm.py https://example.com --skip-chatbot
python scripts/run_chatbot.py
```

## Best Practices

1. **Start Small**: Begin with `--max-pages 5` to test
2. **Check Content**: Review PDF before generating embeddings
3. **Monitor Logs**: Check `chatbot.log` for issues
4. **Clean Up**: Remove old PDFs periodically
5. **Backup Index**: Keep backup of FAISS index

## Workflow Diagram

```
URL Input
    ↓
[Scrape Website]
    ↓
Scraped Content
    ↓
[Create PDF]
    ↓
PDF File
    ↓
[Generate Embeddings]
    ↓
FAISS Index
    ↓
[Launch Chatbot]
    ↓
Ready to Answer Questions!
```

## Next Steps

After running the pipeline:
1. Test the chatbot with sample questions
2. Review analytics in `analytics/` folder
3. Adjust configuration in `config.yaml` if needed
4. Add FAQs to `faq_database.json` for common questions

## Support

For issues or questions:
- Check logs in `chatbot.log`
- Review configuration in `config.yaml`
- See `README.md` for general setup
- See `PROJECT_STRUCTURE.md` for code organization

