from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import logging
from datetime import datetime
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Load news sources
def load_config():
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {'news_sources': {}, 'settings': {}}

CONFIG = load_config()
NEWS_SOURCES = CONFIG.get('news_sources', {})
SETTINGS = CONFIG.get('settings', {})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/trending-news', methods=['GET'])
def get_trending_news():
    """Get trending articles with full content from all sources"""
    try:
        max_articles = int(request.args.get('limit', SETTINGS.get('default_limit', 10)))
        sources = request.args.get('sources', '').split(',') if request.args.get('sources') else list(NEWS_SOURCES.keys())
        
        all_articles = []
        
        # Get trending articles from each source
        max_workers = SETTINGS.get('max_source_workers', 3)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(scrape_source_trending, source, max_articles): source 
                for source in sources if source in NEWS_SOURCES
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    timeout = SETTINGS.get('source_timeout', 60)
                    articles = future.result(timeout=timeout)
                    if articles:
                        all_articles.extend(articles)
                        logger.info(f"Got {len(articles)} articles from {source}")
                except Exception as e:
                    logger.error(f"Failed to scrape {source}: {e}")
        
        # Sort by most recent/trending
        sort_key = SETTINGS.get('sort_by', 'publishDate')
        all_articles.sort(key=lambda x: x.get(sort_key, ''), reverse=True)
        
        max_total = SETTINGS.get('max_total_articles', max_articles * len(sources))
        
        return jsonify({
            'status': 'success',
            'totalArticles': len(all_articles),
            'articles': all_articles[:max_total],
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Trending news error: {e}")
        return jsonify({'error': 'Failed to get trending news'}), 500

def scrape_source_trending(source, max_articles):
    """Scrape trending articles with full content from a single source"""
    try:
        source_config = NEWS_SOURCES[source]
        
        # Step 1: Get homepage and find article links
        timeout = SETTINGS.get('request_timeout', 20)
        response = requests.get(source_config['url'], headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        article_urls = []
        seen_urls = set()
        
        # Extract article URLs from homepage
        min_title_length = SETTINGS.get('min_title_length', 15)
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(source_config['url'], href)
            title = link.get_text(strip=True)
            
            if (is_article_url(full_url) and 
                title and len(title) > min_title_length and 
                full_url not in seen_urls and 
                len(article_urls) < max_articles):
                
                article_urls.append(full_url)
                seen_urls.add(full_url)
        
        # Step 2: Scrape full content for each article
        articles_with_content = []
        
        max_article_workers = SETTINGS.get('max_article_workers', 5)
        with ThreadPoolExecutor(max_workers=max_article_workers) as executor:
            future_to_url = {
                executor.submit(scrape_full_article, url, source_config['name']): url 
                for url in article_urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article_timeout = SETTINGS.get('article_timeout', 30)
                    min_content_length = SETTINGS.get('min_content_length', 100)
                    
                    article = future.result(timeout=article_timeout)
                    if article and len(article.get('content', '')) > min_content_length:
                        articles_with_content.append(article)
                except Exception as e:
                    logger.error(f"Failed to scrape article {url}: {e}")
        
        return articles_with_content
        
    except Exception as e:
        logger.error(f"Source scraping error for {source}: {e}")
        return []

def scrape_full_article(url, source_name):
    """Scrape full article content"""
    min_content_length = SETTINGS.get('min_content_length', 100)
    
    try:
        # Try newspaper3k first
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        
        if len(article.text) > min_content_length:
            return {
                'title': article.title,
                'content': article.text,
                'summary': article.summary,
                'url': url,
                'source': source_name,
                'authors': article.authors,
                'publishDate': article.publish_date.isoformat() if article.publish_date else None,
                'topImage': article.top_image,
                'wordCount': len(article.text.split())
            }
    except:
        pass
    
    # Fallback to BeautifulSoup
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            element.decompose()
        
        # Extract title
        title = ''
        for selector in ['h1', 'title']:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text(strip=True)
                break
        
        # Extract content
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
        
        if len(content) > 100:
            return {
                'title': title,
                'content': content,
                'summary': content[:300] + '...' if len(content) > 300 else content,
                'url': url,
                'source': source_name,
                'authors': [],
                'publishDate': None,
                'topImage': None,
                'wordCount': len(content.split())
            }
    except:
        pass
    
    return None

def is_article_url(url):
    """Check if URL looks like an article"""
    url_lower = url.lower()
    
    # Skip non-article patterns
    skip_patterns = [
        '/video/', '/live/', '/search', '/tag/', '/category/', '/author/', 
        'javascript:', 'mailto:', '#', '?page=', '/rss', '/feed', '/sport/',
        '/weather/', '/market/', '/entertainment/'
    ]
    
    if any(pattern in url_lower for pattern in skip_patterns):
        return False
    
    # Must have reasonable path depth
    return len(url.split('/')) >= 4

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)