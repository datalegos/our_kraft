from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from datetime import datetime
import logging
from bs4 import BeautifulSoup
import hashlib
from urllib.parse import urljoin, urlparse
import time

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Web scraping configuration
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
REQUEST_TIMEOUT = 10

# Available categories
CATEGORIES = [
    'general', 'politics', 'business', 'technology', 'sports', 
    'entertainment', 'health', 'science', 'local', 'breaking'
]

# Local news sources configuration with category support
NEWS_SOURCES = {
    'local_news_1': {
        'name': 'Local News Site 1',
        'base_url': 'https://example-local-news.com',
        'category_urls': {
            'general': '/news',
            'politics': '/politics',
            'business': '/business',
            'technology': '/tech',
            'sports': '/sports',
            'entertainment': '/entertainment',
            'health': '/health',
            'science': '/science',
            'local': '/local',
            'breaking': '/breaking'
        },
        'selectors': {
            'articles': '.news-item',           # CSS selector for article containers
            'title': '.news-title',            # CSS selector for article title
            'link': 'a',                       # CSS selector for article link
            'description': '.news-summary',     # CSS selector for article description
            'image': 'img',                    # CSS selector for article image
            'date': '.news-date'               # CSS selector for publish date
        }
    },
    'local_news_2': {
        'name': 'Local News Site 2',
        'base_url': 'https://another-local-news.com',
        'category_urls': {
            'general': '/',
            'politics': '/category/politics',
            'business': '/category/business',
            'technology': '/category/technology',
            'sports': '/category/sports',
            'entertainment': '/category/entertainment',
            'health': '/category/health',
            'science': '/category/science',
            'local': '/category/local',
            'breaking': '/breaking-news'
        },
        'selectors': {
            'articles': 'article.post',
            'title': 'h2.post-title',
            'link': 'h2.post-title a',
            'description': '.post-excerpt',
            'image': '.post-thumbnail img',
            'date': '.post-date'
        }
    },
    'local_news_3': {
        'name': 'Local News Site 3',
        'base_url': 'https://third-local-news.com',
        'category_urls': {
            'general': '/news/all',
            'politics': '/news/politics',
            'business': '/news/business',
            'technology': '/news/tech',
            'sports': '/news/sports',
            'entertainment': '/news/entertainment',
            'health': '/news/health',
            'science': '/news/science',
            'local': '/news/local',
            'breaking': '/news/urgent'
        },
        'selectors': {
            'articles': 'div.story-card',
            'title': '.story-headline',
            'link': '.story-headline a',
            'description': '.story-teaser',
            'image': '.story-image img',
            'date': '.story-timestamp'
        }
    }
}

def scrape_website(source_config, category='general'):
    """Scrape news articles from a website for a specific category"""
    try:
        # Build the URL based on category
        if category in source_config['category_urls']:
            category_path = source_config['category_urls'][category]
            full_url = source_config['base_url'] + category_path
        else:
            # Fallback to general category
            category_path = source_config['category_urls']['general']
            full_url = source_config['base_url'] + category_path
        
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(full_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        
        # Find article containers
        article_elements = soup.select(source_config['selectors']['articles'])
        
        for element in article_elements[:20]:  # Limit to 20 articles
            try:
                # Extract title
                title_elem = element.select_one(source_config['selectors']['title'])
                title = title_elem.get_text(strip=True) if title_elem else None
                
                # Extract link
                link_elem = element.select_one(source_config['selectors']['link'])
                link = None
                if link_elem:
                    link = link_elem.get('href')
                    if link and not link.startswith('http'):
                        link = urljoin(full_url, link)
                
                # Extract description
                desc_elem = element.select_one(source_config['selectors']['description'])
                description = desc_elem.get_text(strip=True) if desc_elem else None
                
                # Extract image
                img_elem = element.select_one(source_config['selectors']['image'])
                image_url = None
                if img_elem:
                    image_url = img_elem.get('src') or img_elem.get('data-src')
                    if image_url and not image_url.startswith('http'):
                        image_url = urljoin(full_url, image_url)
                
                # Extract date if available
                date_elem = element.select_one(source_config['selectors'].get('date', ''))
                publish_date = None
                if date_elem:
                    publish_date = date_elem.get_text(strip=True)
                
                # Only add if we have at least title and link
                if title and link:
                    article = {
                        'id': hashlib.md5(link.encode()).hexdigest()[:16],
                        'title': title,
                        'description': description,
                        'url': link,
                        'source': source_config['name'],
                        'category': category,
                        'urlToImage': image_url,
                        'publishedAt': publish_date or datetime.utcnow().isoformat(),
                        'scraped_at': datetime.utcnow().isoformat(),
                        'scraped_url': full_url
                    }
                    articles.append(article)
                    
            except Exception as e:
                logger.warning(f"Error processing article element: {str(e)}")
                continue
        
        return articles
        
    except requests.RequestException as e:
        logger.error(f"Failed to scrape {source_config['name']} for category {category}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error scraping {source_config['name']} for category {category}: {str(e)}")
        return []

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'feed-api'}), 200

@app.route('/api/news/categories', methods=['GET'])
def get_available_categories():
    """Get list of available news categories"""
    return jsonify({
        'status': 'success',
        'categories': CATEGORIES,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/api/news/sources', methods=['GET'])
def get_available_sources():
    """Get list of available news sources"""
    sources = []
    for key, config in NEWS_SOURCES.items():
        source_info = {
            'id': key,
            'name': config['name'],
            'base_url': config['base_url'],
            'available_categories': list(config['category_urls'].keys())
        }
        sources.append(source_info)
    
    return jsonify({
        'status': 'success',
        'sources': sources,
        'total_categories': len(CATEGORIES),
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/api/news/headlines', methods=['GET'])
def get_headlines():
    """Get news headlines from specified sources and categories"""
    try:
        # Get parameters
        source_param = request.args.get('source', 'all')
        category_param = request.args.get('category', 'general')
        page_size = min(int(request.args.get('pageSize', 20)), 100)
        
        # Validate category
        if category_param not in CATEGORIES:
            return jsonify({'error': f'Invalid category. Available: {", ".join(CATEGORIES)}'}), 400
        
        all_articles = []
        
        if source_param == 'all':
            # Scrape from all sources for the specified category
            for source_key, source_config in NEWS_SOURCES.items():
                logger.info(f"Scraping {source_config['name']} - {category_param}...")
                articles = scrape_website(source_config, category_param)
                all_articles.extend(articles)
                time.sleep(1)  # Be respectful to websites
        else:
            # Scrape from specific source
            if source_param in NEWS_SOURCES:
                source_config = NEWS_SOURCES[source_param]
                logger.info(f"Scraping {source_config['name']} - {category_param}...")
                all_articles = scrape_website(source_config, category_param)
            else:
                return jsonify({'error': f'Unknown source: {source_param}'}), 400
        
        # Sort by scraped time (most recent first) and limit results
        all_articles.sort(key=lambda x: x['scraped_at'], reverse=True)
        limited_articles = all_articles[:page_size]
        
        return jsonify({
            'status': 'success',
            'totalResults': len(limited_articles),
            'articles': limited_articles,
            'source': source_param,
            'category': category_param,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting headlines: {str(e)}")
        return jsonify({'error': 'Failed to fetch news', 'details': str(e)}), 500

@app.route('/api/news/search', methods=['GET'])
def search_news():
    """Search for news articles containing specific keywords"""
    try:
        query = request.args.get('q')
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        source_param = request.args.get('source', 'all')
        category_param = request.args.get('category', 'general')
        page_size = min(int(request.args.get('pageSize', 20)), 100)
        
        # Validate category
        if category_param not in CATEGORIES:
            return jsonify({'error': f'Invalid category. Available: {", ".join(CATEGORIES)}'}), 400
        
        all_articles = []
        
        # Get articles from sources
        if source_param == 'all':
            for source_key, source_config in NEWS_SOURCES.items():
                articles = scrape_website(source_config, category_param)
                all_articles.extend(articles)
                time.sleep(1)
        else:
            if source_param in NEWS_SOURCES:
                source_config = NEWS_SOURCES[source_param]
                all_articles = scrape_website(source_config, category_param)
            else:
                return jsonify({'error': f'Unknown source: {source_param}'}), 400
        
        # Filter articles by query
        query_lower = query.lower()
        filtered_articles = []
        
        for article in all_articles:
            title_match = query_lower in article['title'].lower() if article['title'] else False
            desc_match = query_lower in article['description'].lower() if article['description'] else False
            
            if title_match or desc_match:
                filtered_articles.append(article)
        
        # Limit results
        limited_articles = filtered_articles[:page_size]
        
        return jsonify({
            'status': 'success',
            'totalResults': len(limited_articles),
            'articles': limited_articles,
            'query': query,
            'source': source_param,
            'category': category_param,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error searching news: {str(e)}")
        return jsonify({'error': 'Failed to search news', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)