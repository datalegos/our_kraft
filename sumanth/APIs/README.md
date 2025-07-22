# 🚀 News Aggregator Microservices

A complete microservices architecture for news aggregation with authentication, feed fetching, tokenization, and AI-powered summarization.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Authentication │    │    Feed API     │    │ Tokenization    │    │ Summarization   │    │   Scraping      │
│      API        │    │                 │    │      API        │    │      API        │    │      API        │
│   Port: 5000    │    │   Port: 5001    │    │   Port: 5002    │    │   Port: 5003    │    │   Port: 5004    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │                       │
                         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                         │     MySQL       │    │   News APIs     │    │   Web Content   │
                         │   Database      │    │  (NewsAPI.org)  │    │   (Any Website) │
                         │   Port: 3306    │    │                 │    │                 │
                         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Services

### 1. **Authentication API** (Port 5000)
- User registration and login
- JWT token management
- User preferences for news categories
- Profile management

### 2. **Feed API** (Port 5001)
- Fetch top headlines from NewsAPI
- Search news articles
- Get news sources
- Filter by category, country, language

### 3. **Tokenization API** (Port 5002)
- Count tokens for different LLM models
- Estimate API costs
- Batch token counting
- Support for GPT-4, GPT-3.5, Claude models

### 4. **Summarization API** (Port 5003)
- AI-powered article summarization
- Multiple summary lengths (short, medium, long)
- Batch summarization
- Custom summarization instructions
- Token usage tracking

### 5. **Scraping API** (Port 5004)
- Extract full article content from any URL
- Batch scraping with concurrent processing
- Website link discovery and RSS parsing
- Multiple scraping methods (newspaper3k, BeautifulSoup)
- Search integration for finding articles

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- NewsAPI key (free at [newsapi.org](https://newsapi.org))
- OpenAI API key (from [platform.openai.com](https://platform.openai.com))

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo>
cd news-aggregator

# Create environment file
cp .env.example .env
```

### 2. Configure API Keys
Edit `.env` file:
```env
NEWS_API_KEY=your_news_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start with Docker
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Manual Setup (Alternative)
```bash
# Install dependencies for each service
cd authentication && pip install -r requirements.txt && cd ..
cd feed-api && pip install -r requirements.txt && cd ..
cd tokenization-api && pip install -r requirements.txt && cd ..
cd summarization-api && pip install -r requirements.txt && cd ..

# Start services in separate terminals
cd authentication && python run.py
cd feed-api && python app.py
cd tokenization-api && python app.py
cd summarization-api && python app.py
```

## 📡 API Endpoints

### Authentication API (5000)
```
POST /api/register          - Register new user
POST /api/login             - User login
GET  /api/user/profile      - Get user profile
PUT  /api/user/preferences  - Update news preferences
GET  /health                - Health check
```

### Feed API (5001)
```
GET  /api/news/top-headlines - Get top news headlines
GET  /api/news/search        - Search news articles
GET  /api/news/sources       - Get news sources
GET  /health                 - Health check
```

### Tokenization API (5002)
```
POST /api/tokens/count        - Count tokens in text
POST /api/tokens/estimate-cost - Estimate API costs
POST /api/tokens/batch-count  - Batch token counting
GET  /api/tokens/models       - Get supported models
GET  /health                  - Health check
```

### Summarization API (5003)
```
POST /api/summarize/article  - Summarize single article
POST /api/summarize/batch    - Batch summarization
POST /api/summarize/custom   - Custom summarization
GET  /health                 - Health check
```

### Scraping API (5004)
```
POST /api/scrape/article     - Scrape single article content
POST /api/scrape/batch       - Batch scrape multiple URLs
POST /api/scrape/website     - Discover article links on website
POST /api/scrape/rss         - Parse RSS feeds
POST /api/scrape/search      - Search articles on specific sites
GET  /health                 - Health check
```

## 🧪 Testing the APIs

### Test Authentication
```bash
# Register user
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "SecurePass123",
    "preferred_categories": "technology,business"
  }'

# Login
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "SecurePass123"
  }'
```

### Test Feed API
```bash
# Get top headlines
curl "http://localhost:5001/api/news/top-headlines?category=technology&pageSize=5"

# Search news
curl "http://localhost:5001/api/news/search?q=artificial%20intelligence"
```

### Test Tokenization API
```bash
# Count tokens
curl -X POST http://localhost:5002/api/tokens/count \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample news article to count tokens.",
    "model": "gpt-4"
  }'

# Estimate cost
curl -X POST http://localhost:5002/api/tokens/estimate-cost \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample news article to estimate cost.",
    "model": "gpt-4",
    "type": "input"
  }'
```

### Test Summarization API
```bash
# Summarize article
curl -X POST http://localhost:5003/api/summarize/article \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your long news article content here...",
    "length": "medium",
    "model": "gpt-3.5-turbo"
  }'
```

## 🔄 Complete Workflow Example

Here's how all services work together:

```bash
# 1. Register and login user
TOKEN=$(curl -s -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "SecurePass123"}' \
  | jq -r '.token')

# 2. Get user preferences
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:5000/api/user/preferences

# 3. Fetch news based on preferences
curl "http://localhost:5001/api/news/top-headlines?category=technology"

# 4. Count tokens before summarization
curl -X POST http://localhost:5002/api/tokens/count \
  -H "Content-Type: application/json" \
  -d '{"text": "Article content...", "model": "gpt-3.5-turbo"}'

# 5. Summarize the article
curl -X POST http://localhost:5003/api/summarize/article \
  -H "Content-Type: application/json" \
  -d '{"content": "Article content...", "length": "medium"}'
```

## 🛠️ Development

### Project Structure
```
news-aggregator/
├── authentication/          # User auth & preferences
│   ├── app.py
│   ├── models.py
│   ├── routes.py
│   └── requirements.txt
├── feed-api/               # News fetching
│   ├── app.py
│   └── requirements.txt
├── tokenization-api/       # Token counting & cost estimation
│   ├── app.py
│   └── requirements.txt
├── summarization-api/      # AI summarization
│   ├── app.py
│   └── requirements.txt
├── docker-compose.yml      # Container orchestration
└── README.md
```

### Adding New Features

1. **Authentication**: Edit `authentication/routes.py`
2. **Feed Sources**: Modify `feed-api/app.py`
3. **Token Models**: Update `tokenization-api/app.py`
4. **Summary Types**: Enhance `summarization-api/app.py`

### Environment Variables

Create `.env` file:
```env
# Required API Keys
NEWS_API_KEY=your_news_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Database (for Docker)
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_DATABASE=news_auth_db

# Optional: Custom ports
AUTH_PORT=5000
FEED_PORT=5001
TOKEN_PORT=5002
SUMMARY_PORT=5003
```

## 🔒 Security Features

- JWT token authentication
- Password hashing
- Input validation
- CORS configuration
- Environment-based secrets
- SQL injection protection

## 📊 Monitoring

Check service health:
```bash
curl http://localhost:5000/health  # Auth API
curl http://localhost:5001/health  # Feed API
curl http://localhost:5002/health  # Tokenization API
curl http://localhost:5003/health  # Summarization API
```

## 🚀 Production Deployment

### Using Docker Compose
```bash
# Production build
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale feed-api=3 --scale summarization-api=2
```

### Manual Deployment
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app('production')"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Update documentation
5. Submit pull request

## 📝 License

MIT License - see LICENSE file for details.

---

**🎉 Your complete news aggregation microservices are ready!**

Start with `docker-compose up -d` and begin building your news application!