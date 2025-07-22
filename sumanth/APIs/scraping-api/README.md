# 🕷️ Web Scraping API

A powerful web scraping microservice for extracting news articles and content from various websites. Built with Flask, BeautifulSoup, and newspaper3k.

## 🚀 Features

### 📰 **Article Scraping**
- Extract full article content from URLs
- Support for multiple scraping methods (newspaper3k & BeautifulSoup)
- Automatic content cleaning and formatting
- Extract metadata (title, author, publish date, images)

### 🔍 **Website Discovery**
- Scrape all article links from news websites
- Intelligent article URL detection
- Filter and deduplicate results

### 📡 **RSS Feed Parsing**
- Parse RSS/Atom feeds
- Extract article metadata and URLs
- Support for various feed formats

### 🔎 **Search Integration**
- Search for articles on specific websites
- Google search with site: operator
- Extract search results with snippets

### ⚡ **Batch Processing**
- Concurrent scraping of multiple URLs
- Configurable worker threads
- Comprehensive error handling

## 📡 API Endpoints

### Single Article Scraping
```http
POST /api/scrape/article
Content-Type: application/json

{
  "url": "https://example.com/article",
  "method": "newspaper"  // "newspaper" or "beautifulsoup"
}
```

### Batch Article Scraping
```http
POST /api/scrape/batch
Content-Type: application/json

{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ],
  "method": "newspaper",
  "maxWorkers": 5
}
```

### Website Link Discovery
```http
POST /api/scrape/website
Content-Type: application/json

{
  "url": "https://news-website.com",
  "maxLinks": 20
}
```

### RSS Feed Parsing
```http
POST /api/scrape/rss
Content-Type: application/json

{
  "url": "https://example.com/rss.xml",
  "maxItems": 20
}
```

### Article Search
```http
POST /api/scrape/search
Content-Type: application/json

{
  "query": "artificial intelligence",
  "site": "techcrunch.com",
  "maxResults": 10
}
```

### Health Check
```http
GET /health
```

## 🛠️ Installation

### Using Docker (Recommended)
```bash
# Build and run
docker build -t scraping-api .
docker run -p 5004:5004 scraping-api

# Or use docker-compose
docker-compose up scraping-api
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python app.py
```

## 🧪 Usage Examples

### Scrape Single Article
```bash
curl -X POST http://localhost:5004/api/scrape/article \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://techcrunch.com/2024/01/15/ai-breakthrough/",
    "method": "newspaper"
  }'
```

### Batch Scraping
```bash
curl -X POST http://localhost:5004/api/scrape/batch \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://techcrunch.com/article1/",
      "https://techcrunch.com/article2/"
    ],
    "method": "newspaper",
    "maxWorkers": 3
  }'
```

### Discover Articles on Website
```bash
curl -X POST http://localhost:5004/api/scrape/website \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://techcrunch.com",
    "maxLinks": 15
  }'
```

### Parse RSS Feed
```bash
curl -X POST http://localhost:5004/api/scrape/rss \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://techcrunch.com/feed/",
    "maxItems": 10
  }'
```

## 📊 Response Format

### Article Scraping Response
```json
{
  "status": "success",
  "url": "https://example.com/article",
  "method": "newspaper",
  "article": {
    "title": "Article Title",
    "content": "Full article content...",
    "summary": "Article summary...",
    "authors": ["Author Name"],
    "publishDate": "2024-01-15T10:30:00",
    "topImage": "https://example.com/image.jpg",
    "images": ["https://example.com/img1.jpg"],
    "keywords": ["keyword1", "keyword2"],
    "wordCount": 850
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Batch Scraping Response
```json
{
  "status": "success",
  "totalUrls": 3,
  "successful": 2,
  "failed": 1,
  "method": "newspaper",
  "results": [
    {
      "url": "https://example.com/article1",
      "status": "success",
      "article": { /* article data */ }
    },
    {
      "url": "https://example.com/article2",
      "status": "failed",
      "error": "Failed to extract content"
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## ⚙️ Configuration

### Environment Variables
```env
PORT=5004                        # API port
MAX_CONCURRENT_REQUESTS=5        # Max concurrent scraping threads
REQUEST_TIMEOUT=10               # Request timeout in seconds
USER_AGENT=Mozilla/5.0...        # Custom user agent
```

### Scraping Methods

#### 1. **Newspaper3k** (Recommended)
- Advanced article extraction
- Automatic content cleaning
- Metadata extraction (authors, dates, keywords)
- Image extraction
- Natural language processing

#### 2. **BeautifulSoup**
- Fallback method for difficult sites
- Custom selector-based extraction
- More control over parsing
- Better for non-standard layouts

## 🛡️ Best Practices

### Rate Limiting
- Built-in concurrent request limiting
- Configurable worker threads
- Automatic retry with backoff

### Respectful Scraping
- Proper User-Agent headers
- Request delays between batches
- Respect robots.txt (manual check recommended)

### Error Handling
- Comprehensive exception handling
- Detailed error messages
- Graceful degradation

## 🔧 Integration Examples

### With Feed API
```python
# 1. Discover articles
response = requests.post('http://localhost:5004/api/scrape/website', 
                        json={'url': 'https://news-site.com'})
links = response.json()['links']

# 2. Scrape full content
articles = []
for link in links[:5]:
    scrape_response = requests.post('http://localhost:5004/api/scrape/article',
                                   json={'url': link['url']})
    if scrape_response.status_code == 200:
        articles.append(scrape_response.json()['article'])
```

### With Summarization API
```python
# 1. Scrape article
scrape_response = requests.post('http://localhost:5004/api/scrape/article',
                               json={'url': 'https://example.com/article'})
article = scrape_response.json()['article']

# 2. Summarize content
summary_response = requests.post('http://localhost:5003/api/summarize/article',
                                json={'content': article['content']})
summary = summary_response.json()['summary']
```

## 🚨 Limitations

- Some websites may block automated requests
- JavaScript-heavy sites may not work (consider Selenium for those)
- Rate limiting may be needed for large-scale scraping
- Respect website terms of service and robots.txt

## 🔍 Troubleshooting

### Common Issues

**403 Forbidden Errors**
- Website blocking automated requests
- Try different User-Agent headers
- Add delays between requests

**Empty Content**
- Site uses JavaScript to load content
- Try BeautifulSoup method as fallback
- Consider using Selenium for JS-heavy sites

**Timeout Errors**
- Increase REQUEST_TIMEOUT
- Check network connectivity
- Some sites may be slow to respond

## 📈 Performance

- **Concurrent Processing**: Up to 10 simultaneous requests
- **Memory Efficient**: Streaming content processing
- **Fast Parsing**: Optimized BeautifulSoup and newspaper3k usage
- **Caching**: Consider adding Redis for frequently scraped URLs

## 🔐 Security

- Input URL validation
- Request timeout protection
- User-Agent rotation support
- No execution of scraped JavaScript
- Safe HTML parsing

---

**🕷️ Happy Scraping!** 

This API provides a robust foundation for web scraping in your news aggregator microservices architecture.