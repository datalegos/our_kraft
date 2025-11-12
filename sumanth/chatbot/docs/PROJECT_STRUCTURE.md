# Project Structure

This document describes the organized structure of the chatbot application.

## Directory Structure

```
chatbot/
├── src/                          # Source code
│   └── chatbot/                  # Main package
│       ├── __init__.py          # Package initialization
│       ├── core/                 # Core functionality
│       │   ├── __init__.py
│       │   ├── app.py           # Main application
│       │   └── config.py        # Configuration management
│       ├── optimizers/           # Token optimization modules
│       │   ├── __init__.py
│       │   ├── analytics.py     # Analytics tracking
│       │   ├── cost_optimizer.py # Token control & optimization
│       │   ├── memory_manager.py # Conversation memory
│       │   ├── query_router.py   # Query classification
│       │   └── response_cache.py # Response caching
│       ├── processors/           # Document/content processors
│       │   ├── __init__.py
│       │   ├── document_processor.py # Document processing
│       │   ├── embeddings.py     # Embedding creation
│       │   └── scraper.py        # Web scraping
│       └── utils/                # Utilities
│           ├── __init__.py
│           ├── exceptions.py     # Custom exceptions
│           └── logger.py         # Logging utilities
│
├── scripts/                      # Executable scripts
│   ├── run_chatbot.py           # Main chatbot runner
│   ├── create_embeddings.py     # Embedding creation script
│   └── run_scraper.py           # Scraper script
│
├── data/                         # Data directory
├── config.yaml                   # Main configuration file
├── scraper_config.yaml           # Scraper configuration
├── faq_database.json             # FAQ database
├── environment.yml               # Conda environment
├── pyproject.toml                # Python project configuration
├── README.md                     # Main documentation
├── PROJECT_STRUCTURE.md          # This file
├── DOCUMENT_PROCESSING_GUIDE.md  # Document processing guide
└── OPTIMIZATION_IMPLEMENTATION.md # Optimization details
```

## Package Organization

### `chatbot.core`
Core application functionality:
- **app.py**: Main chatbot application class
- **config.py**: Configuration loading and management

### `chatbot.optimizers`
Token optimization modules:
- **analytics.py**: Performance tracking and metrics
- **cost_optimizer.py**: Token counting, compression, re-ranking
- **memory_manager.py**: Conversation memory with sliding window
- **query_router.py**: Query classification and routing
- **response_cache.py**: Response caching with similarity matching

### `chatbot.processors`
Content processing modules:
- **document_processor.py**: Document extraction and processing
- **embeddings.py**: Embedding creation and FAISS index
- **scraper.py**: Web scraping functionality

### `chatbot.utils`
Utility modules:
- **exceptions.py**: Custom exception classes
- **logger.py**: Logging configuration

## Usage

### Running the Application

```bash
# From project root
python scripts/run_chatbot.py

# Or if installed as package
chatbot
```

### Creating Embeddings

```bash
python scripts/create_embeddings.py

# Or if installed as package
create-embeddings
```

### Running Scraper

```bash
python scripts/run_scraper.py

# Or if installed as package
scraper
```

## Installation

### Development Installation

```bash
# Install in development mode
pip install -e .

# Or using conda
conda env create -f environment.yml
conda activate practice
pip install -e .
```

## Import Examples

```python
# Core modules
from chatbot.core.app import OptimizedChatbotApp
from chatbot.core.config import load_config, validate_config

# Optimizers
from chatbot.optimizers import (
    ConversationMemory,
    QueryRouter,
    ResponseCache,
    AnalyticsTracker
)

# Processors
from chatbot.processors import (
    DocumentProcessor,
    create_embeddings_from_documents
)

# Utils
from chatbot.utils import logger, ConfigurationError
```

## Benefits of This Structure

1. **Modularity**: Clear separation of concerns
2. **Maintainability**: Easy to find and modify code
3. **Scalability**: Easy to add new features
4. **Testability**: Each module can be tested independently
5. **Reusability**: Modules can be imported and reused
6. **Professional**: Follows Python packaging best practices

## Configuration

All configuration is in `config.yaml` at the project root. The config module automatically finds it.

## Data Directories

- `data/`: General data storage
- `conversations/`: Stored conversation history
- `cache/`: Response cache
- `analytics/`: Analytics data
- `faiss_index_bge_small/`: FAISS vector index

