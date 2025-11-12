# How to Use the Scrape-to-LLM Pipeline

## Where to Paste the Website URL

You have **two options** for providing the website URL:

### Option 1: In config.yaml (Recommended)
Edit `config.yaml` and add your URL:
```yaml
scraper:
  url: "https://your-website.com"
```

Then run:
```bash
python scripts/scrape_to_llm.py
```

### Option 2: As Command Line Argument
Pass the URL directly when running the script:
```bash
python scripts/scrape_to_llm.py https://your-website.com
```

## Step-by-Step Instructions

### Method 1: Using config.yaml (Easiest)

1. **Edit `config.yaml`**:
   ```yaml
   scraper:
     url: "https://your-website.com"
     max_pages: 10
   ```

2. **Run the script** (no URL needed):
   ```bash
   python scripts/scrape_to_llm.py
   ```

### Method 2: Using Command Line

1. **Open Terminal/Command Prompt**
2. **Navigate to Project Directory**:
   ```bash
   cd C:\Users\HP\OneDrive\Desktop\chatbot
   ```
3. **Run with URL**:
   ```bash
   python scripts/scrape_to_llm.py https://your-website.com
   ```

## Examples

### Example 1: Basic Usage
```bash
python scripts/scrape_to_llm.py https://data-legos.com
```

### Example 2: With More Pages
```bash
python scripts/scrape_to_llm.py https://example.com --max-pages 20
```

### Example 3: Custom Output Directory
```bash
python scripts/scrape_to_llm.py https://example.com --output-dir my_documents
```

## Complete Example

```bash
# 1. Open terminal in the chatbot folder
cd C:\Users\HP\OneDrive\Desktop\chatbot

# 2. Run with your URL (replace with actual URL)
python scripts/scrape_to_llm.py https://data-legos.com

# The script will:
# - Scrape the website
# - Create PDF
# - Generate embeddings
# - Launch chatbot
```

## Alternative: Using scraper_config.yaml

If you prefer to configure the URL in a file instead:

1. **Edit `scraper_config.yaml`**:
   ```yaml
   home_page: "https://your-website.com"
   message: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
   max_pages: 10
   output_file: "content.txt"
   ```

2. **Then run the scraper separately**:
   ```bash
   python scripts/run_scraper.py
   ```

   Then create embeddings:
   ```bash
   python scripts/create_embeddings.py
   ```

   Then launch chatbot:
   ```bash
   python scripts/run_chatbot.py
   ```

## Quick Reference

| Method | Command | URL Location |
|--------|---------|--------------|
| **Recommended** | `python scripts/scrape_to_llm.py` | `config.yaml` → `scraper.url` |
| Alternative | `python scripts/scrape_to_llm.py <URL>` | Command line argument |

**Note**: If URL is provided in both places, command line argument takes priority.

## Need Help?

- See `SCRAPE_TO_LLM_GUIDE.md` for detailed documentation
- See `QUICK_START.md` for quick examples
- Check `chatbot.log` for error messages

