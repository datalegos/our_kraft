# DataLegos RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for DataLegos company information using LangChain, FAISS, and OpenAI.

## Project Structure

```
chatbot/
├── app.py                    # Main Gradio web application
├── scraper.py               # Website scraper (recursive, deduplication)
├── create_embeddings.py     # Process content and create vector embeddings
├── test_bot.py             # Command-line testing script
├── scraper_config.yaml     # Scraper configuration
├── environment.yml         # Conda environment dependencies
├── .env                    # Environment variables (OpenAI API key)
├── scraped_content.txt     # Raw scraped website content
├── processed_chunks.txt    # Processed text chunks for inspection
└── vector_index/           # FAISS vector database
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate practice

# Or install manually
pip install gradio langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu sentence-transformers python-dotenv openai beautifulsoup4 requests pyyaml
```

### 2. Configure API Key
Create/edit `.env` file:
```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Scrape Website Content
```bash
python scraper.py
```
This will:
- Discover all pages on the DataLegos website
- Scrape unique content from each page
- Save clean content to `scraped_content.txt`

### 4. Create Vector Embeddings
```bash
python create_embeddings.py
```
This will:
- Process scraped content into optimized chunks
- Create FAISS vector embeddings
- Save chunks to `processed_chunks.txt`
- Save vector index to `vector_index/`

### 5. Run the Chatbot
```bash
python app.py
```
Access at: http://127.0.0.1:7860

## Testing

### Command Line Test
```bash
python test_bot.py
```

### Web Interface
Run `python app.py` and test queries like:
- "What services does DataLegos offer?"
- "Who are the team members?"
- "What is the Career Catalyst program?"
- "How can I contact DataLegos?"

## Features

- **Recursive Web Scraping**: Discovers and scrapes all website pages
- **Duplicate Prevention**: Avoids scraping the same content twice
- **Smart Chunking**: Optimized text chunking for better retrieval
- **Token Optimization**: Efficient prompt engineering to reduce costs
- **Quality Content**: Clean, structured content from multiple pages

## Configuration

### Scraper Settings (`scraper_config.yaml`)
```yaml
home_page: "https://data-legos.com"
message: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
max_pages: 10
```

### Embedding Settings
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 100 characters
- **Embedding Model**: BAAI/bge-small-en-v1.5
- **Vector Store**: FAISS

## Content Coverage

The chatbot has comprehensive information about:
- ✅ **Services**: Neo4j, Data Engineering, AI Analytics
- ✅ **Team**: Detailed profiles of key team members
- ✅ **Industries**: Finance, Healthcare, Retail, Logistics, Startups
- ✅ **Contact**: Phone, email, business hours
- ✅ **Programs**: Career Catalyst internship program

## Troubleshooting

### Common Issues
1. **"I don't have that information"**: Re-run scraper and embeddings
2. **OpenAI API Error**: Check your API key in `.env`
3. **Import Errors**: Ensure all dependencies are installed
4. **Port 7860 in use**: Change port in `app.py`

### Regenerate Content
If website content changes:
```bash
python scraper.py          # Re-scrape website
python create_embeddings.py # Re-create embeddings
python app.py              # Restart chatbot
```