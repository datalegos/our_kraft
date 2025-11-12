# Chatbot Application

A RAG (Retrieval-Augmented Generation) chatbot that scrapes website content, creates embeddings, and answers questions using OpenAI's GPT models.

## Features

- **Web Scraping**: Automated website content collection with rate limiting and error handling
- **Vector Search**: FAISS-based semantic search using HuggingFace embeddings
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate, context-aware responses
- **Configurable**: All settings configurable via YAML configuration file
- **Error Handling**: Comprehensive error handling with fallbacks and logging
- **Production Ready**: Organized code structure with proper separation of concerns

## Project Structure

```
chatbot/
├── src/                    # Source code
│   └── chatbot/            # Main package
│       ├── core/           # Core functionality
│       ├── optimizers/     # Token optimization modules
│       ├── processors/     # Document/content processors
│       └── utils/          # Utilities
├── scripts/                # Executable scripts
│   ├── documents/          # PDF documents for processing
│   └── faiss_index_bge_small/  # Vector embeddings index
├── docs/                   # All documentation
├── config.yaml             # Main configuration
└── scraper_config.yaml     # Scraper settings
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed structure.

## Documentation

All documentation is organized in the [`docs/`](docs/) folder:
- [Installation Guide](docs/INSTALLATION_GUIDE.md)
- [Quick Start Guide](docs/QUICK_START.md)
- [How to Use](docs/HOW_TO_USE.md)
- [Test Questions](docs/CHATBOT_TEST_QUESTIONS.md)
- [Scrape to LLM Guide](docs/SCRAPE_TO_LLM_GUIDE.md)
- And more...

## Setup

### 1. Environment Setup

#### Option A: Using requirements.txt (Recommended)

```bash
# Install all packages
pip install -r requirements.txt
```

#### Option B: Using conda (environment.yml)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate the environment
conda activate practice
```

**Note**: For conda, if the environment already exists, update it with:
```bash
conda env update -f environment.yml --prune
```

See [docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md) for detailed installation instructions.

### 2. Configuration

1. Edit `config.yaml` and add your OpenAI API key:
   ```yaml
   openai:
     api_key: "your_openai_api_key_here"
   ```

2. Configure scraper settings in `scraper_config.yaml`:
   ```yaml
   home_page: "https://your-website.com"
   message: "Your User-Agent string"
   max_pages: 10
   output_file: "content.txt"
   ```

### 3. Run the Pipeline

#### Step 1: Scrape Website Content (Optional)
```bash
python scripts/run_scraper.py
```

This will:
- Scrape the website specified in `scraper_config.yaml`
- Save content to `content.txt` (or configured output file)
- Handle errors gracefully with retries and timeouts

#### Step 2: Create Embeddings
```bash
python scripts/create_embeddings.py
```

This will:
- Process documents from `documents/` folder (or content.txt)
- Split into chunks (configurable size and overlap)
- Create embeddings using HuggingFace model
- Save FAISS index to `faiss_index_bge_small/`

#### Step 3: Launch Chatbot
```bash
python scripts/run_chatbot.py
```

This will:
- Load the FAISS index and embedding model
- Start a Gradio web interface
- Allow users to ask questions about the content

## Configuration

All configuration is centralized in `config.yaml`. Edit this file to customize settings:

### Key Settings

- **rag.k_documents**: Number of documents to retrieve (default: 3)
- **rag.chunk_size**: Text chunk size in words (default: 500)
- **rag.chunk_overlap**: Overlap between chunks in words (default: 50)
- **openai.model**: OpenAI model to use (default: gpt-4o-mini)
- **openai.temperature**: Model temperature (default: 0.7)
- **scraper.timeout**: Request timeout in seconds (default: 10)
- **scraper.delay**: Delay between requests in seconds (default: 1.0)

See `config.yaml` for all available settings and their descriptions.

## Improvements Made

### ✅ Code Organization
- Separated concerns into modules (scraper, embeddings, app, config)
- Created utility modules for logging and exceptions
- Removed hardcoded values

### ✅ Error Handling
- Comprehensive try-except blocks with proper error messages
- Custom exception classes for different error types
- Fallback responses for API failures
- Validation of configuration and dependencies

### ✅ Configuration Management
- All settings in `config.yaml` (YAML format)
- Default values for all parameters
- Configuration validation on startup
- No dependency on .env files

### ✅ Logging
- Structured logging with file and console output
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Detailed error logging with stack traces

### ✅ Scraper Improvements
- Request timeouts and retries
- Rate limiting between requests
- Better text extraction (removes scripts, styles, navigation)
- Content type validation
- Progress logging

### ✅ Code Quality
- Fixed typos in prompts
- Removed unused imports
- Improved prompt template with better instructions
- Type hints and docstrings
- Better variable names

### ✅ User Experience
- Clear error messages
- Progress indicators
- Validation messages
- Graceful degradation

## Usage Tips

1. **Adjust Chunk Size**: If responses are too generic, try increasing `rag.chunk_size` in `config.yaml` (e.g., 1000)
2. **More Documents**: Increase `rag.k_documents` in `config.yaml` for more context (e.g., 5-7)
3. **Scraper Settings**: Adjust `max_pages` in `scraper_config.yaml` based on website size
4. **Rate Limiting**: Increase `scraper.delay` in `config.yaml` if you encounter rate limiting issues

## Troubleshooting

### "FAISS index not found"
- Run `python embeddings.py` first to create the index

### "Content file not found"
- Run `python scraper.py` first to scrape content

### "openai.api_key is not set"
- Edit `config.yaml` and add your OpenAI API key to the `openai.api_key` field

### Scraper fails on some pages
- Check logs in `chatbot.log` for details
- Increase `scraper.timeout` in `config.yaml` if pages load slowly
- Adjust `scraper.max_retries` in `config.yaml` for unreliable connections

## License

This project is for educational and internal use.

