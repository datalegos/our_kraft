# 📰 Civilian News Scraper - Usage Guide

## 🎯 What This Scraper Does

This scraper is designed specifically for **civilian news aggregators** that need:
- **Home page blog discovery** from trusted news sources
- **Full article content extraction** with civilian relevance filtering
- **Clean, structured data** ready for news apps

## 🚀 Quick Start

### 1. Start the Scraper API

```bash
cd scraping-api
python app.py
```

The API will start on `http://localhost:5004`

### 2. Run the Simple Example

```bash
python simple_example.py
```

This will:
1. Discover civilian-relevant articles from BBC and CNN home pages
2. Scrape full content from the top 5 articles
3. Save everything to `civilian_news.json`

## 📡 API Endpoints

### Discover Home Page Blogs
```http
POST /api/discover-blogs
Content-Type: application/json

{
  "source": "bbc",        // Available: bbc, cnn, reuters
  "maxArticles": 10       // Max articles to discover
}
```

**Response:**
```json
{
  "status": "success",
  "source": "BBC News",
  "totalArticles": 8,
  "articles": [
    {
      "url": "https://www.bbc.com/news/article-123",
      "title": "Important Health Update for Citizens",
      "source": "BBC News"
    }
  ]
}
```

### Scrape Full Article
```http
POST /api/scrape-article
Content-Type: application/json

{
  "url": "https://www.bbc.com/news/article-123"
}
```

**Response:**
```json
{
  "status": "success",
  "article": {
    "title": "Important Health Update for Citizens",
    "content": "Full article content here...",
    "summary": "Brief summary of the article...",
    "authors": ["John Smith"],
    "publishDate": "2024-01-15T10:30:00",
    "wordCount": 450,
    "civilian_relevant": true,
    "content_type": "civilian_news"
  }
}
```

### Batch Scrape Articles
```http
POST /api/scrape-batch
Content-Type: application/json

{
  "urls": [
    "https://www.bbc.com/news/article-1",
    "https://www.cnn.com/news/article-2"
  ],
  "maxWorkers": 3
}
```

## 📁 Where Content is Stored

### API Responses (Default)
- Content is returned in **JSON format** via API responses
- **Not stored permanently** by default
- Perfect for real-time news aggregation

### File Storage (Optional)
You can save content to files:

```python
import json

# Save articles to JSON file
with open('civilian_news.json', 'w') as f:
    json.dump(articles, f, indent=2)

# Save to CSV for spreadsheet use
import csv
with open('civilian_news.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Title', 'Content', 'Authors', 'Date', 'Word Count'])
    for article in articles:
        writer.writerow([
            article['title'],
            article['content'][:100] + '...',  # Truncated for CSV
            ', '.join(article.get('authors', [])),
            article.get('publishDate', ''),
            article.get('wordCount', 0)
        ])
```

### Database Storage (Advanced)
For production news aggregators:

```python
import sqlite3

# Create database
conn = sqlite3.connect('civilian_news.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY,
        title TEXT,
        content TEXT,
        summary TEXT,
        authors TEXT,
        publish_date TEXT,
        word_count INTEGER,
        url TEXT UNIQUE,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Insert articles
for article in articles:
    cursor.execute('''
        INSERT OR REPLACE INTO articles 
        (title, content, summary, authors, publish_date, word_count, url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        article['title'],
        article['content'],
        article['summary'],
        ', '.join(article.get('authors', [])),
        article.get('publishDate'),
        article.get('wordCount', 0),
        article['url']
    ))

conn.commit()
conn.close()
```

## 🎯 Civilian Focus Features

### Content Filtering
Articles are automatically filtered for civilian relevance:

**✅ Included Topics:**
- Health & Education
- Politics & Government
- Environment & Science
- Economy & Technology
- Community & Society

**❌ Excluded Topics:**
- Celebrity gossip
- Entertainment news
- Sports & gaming
- Fashion & lifestyle
- Travel & food

### Quality Indicators
Each article includes:
```json
{
  "civilian_relevant": true,
  "content_type": "civilian_news",
  "wordCount": 450,
  "summary": "Auto-generated summary..."
}
```

## 🔄 Complete Workflow Example

```python
import requests
import json

BASE_URL = 'http://localhost:5004'

# 1. Discover articles from multiple sources
all_articles = []
for source in ['bbc', 'cnn', 'reuters']:
    response = requests.post(f"{BASE_URL}/api/discover-blogs", 
                           json={"source": source, "maxArticles": 5})
    if response.status_code == 200:
        articles = response.json()['articles']
        all_articles.extend(articles)

# 2. Scrape top articles
top_urls = [article['url'] for article in all_articles[:10]]
response = requests.post(f"{BASE_URL}/api/scrape-batch", 
                       json={"urls": top_urls, "maxWorkers": 3})

if response.status_code == 200:
    results = response.json()['results']
    scraped_articles = [r['article'] for r in results if r['status'] == 'success']
    
    # 3. Save for your news aggregator
    with open('daily_civilian_news.json', 'w') as f:
        json.dump(scraped_articles, f, indent=2)
    
    print(f"✅ Scraped {len(scraped_articles)} civilian news articles!")
```

## 📊 Data Structure

Each scraped article contains:

```json
{
  "title": "Article headline",
  "content": "Full article text content",
  "summary": "Auto-generated summary (2-3 sentences)",
  "authors": ["Author Name"],
  "publishDate": "2024-01-15T10:30:00",
  "topImage": "https://example.com/image.jpg",
  "keywords": ["keyword1", "keyword2"],
  "url": "https://source.com/article",
  "wordCount": 450,
  "civilian_relevant": true,
  "content_type": "civilian_news",
  "source": "bbc"
}
```

## 🛠️ Integration with Your News App

### For Mobile Apps
```javascript
// React Native / JavaScript
fetch('http://localhost:5004/api/discover-blogs', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({source: 'bbc', maxArticles: 10})
})
.then(response => response.json())
.then(data => {
  // Use data.articles in your app
  console.log(`Found ${data.totalArticles} civilian news articles`);
});
```

### For Web Apps
```python
# Flask/Django backend
import requests

def get_daily_civilian_news():
    articles = []
    for source in ['bbc', 'cnn']:
        response = requests.post('http://localhost:5004/api/discover-blogs',
                               json={'source': source, 'maxArticles': 5})
        if response.status_code == 200:
            articles.extend(response.json()['articles'])
    return articles
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Add more civilian sources
civilian_sources:
  guardian:
    name: "The Guardian"
    url: "https://www.theguardian.com"
    priority: "high"

# Adjust content filtering
content_filters:
  include_keywords:
    - "healthcare"
    - "education"
    - "climate"
  
  exclude_keywords:
    - "celebrity"
    - "sports"

# Performance settings
settings:
  max_articles_per_source: 15
  max_concurrent_requests: 5
```

## 🚨 Important Notes

### Rate Limiting
- Built-in delays between requests
- Respectful scraping practices
- Max 3-5 concurrent requests

### Content Freshness
- Articles are scraped in real-time
- No caching by default
- Perfect for live news feeds

### Error Handling
- Graceful failure handling
- Detailed error messages
- Fallback extraction methods

## 📈 Performance Tips

1. **Batch Processing**: Use `/api/scrape-batch` for multiple articles
2. **Source Selection**: Focus on 2-3 reliable sources
3. **Content Limits**: Set reasonable `maxArticles` limits
4. **Caching**: Implement your own caching for frequently accessed articles

## 🎉 You're Ready!

Your civilian news scraper is now ready to provide clean, relevant content for your news aggregator app. The scraped content goes wherever you direct it - API responses, files, databases, or directly into your application.

**Happy scraping!** 📰✨