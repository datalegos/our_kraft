import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import re
from typing import List, Dict, Tuple
from config import EENADU_BASE_URL, HEADERS, CATEGORY_URLS
from utils import clean_telugu_text, extract_date_from_text, truncate_text, is_valid_article_title

class EenaduScraper:
    def __init__(self):
        self.base_url = EENADU_BASE_URL
        self.headers = HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_category_url(self, category: str = None) -> str:
        """Get URL based on category"""
        if category and category in CATEGORY_URLS:
            return self.base_url + CATEGORY_URLS[category]
        return self.base_url
    
    def extract_article_content(self, article_url: str) -> Tuple[str, str]:
        """Extract full content from article page with proper Telugu encoding"""
        try:
            response = self.session.get(article_url, timeout=15)
            response.encoding = 'utf-8'  # Ensure proper encoding for Telugu
            
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            
            # Multiple selectors to find article content
            content_selectors = [
                '.story-content',
                '.article-content', 
                '.news-content',
                '[class*="story"]',
                '[class*="content"]',
                '[class*="article"]'
            ]
            
            content = ""
            for selector in content_selectors:
                content_divs = soup.select(selector)
                if content_divs:
                    for div in content_divs:
                        text = div.get_text(strip=True)
                        if text and len(text) > 50:
                            content += text + " "
                    if content:
                        break
            
            # Fallback: get all paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 30:
                        content_parts.append(text)
                
                content = " ".join(content_parts)
            
            # Clean the content for proper Telugu display
            cleaned_content = clean_telugu_text(content)
            return cleaned_content, ""
            
        except Exception as e:
            return "", str(e)
    
    def extract_article_links(self, soup: BeautifulSoup, limit: int) -> List[Tuple[any, str]]:
        """Extract article links from the page"""
        article_selectors = [
            'a[href*="/story/"]',
            'a[href*="/news/"]', 
            'a[href*="/article/"]',
            '.story-card a',
            '.news-item a',
            '.headline a',
            'h1 a', 'h2 a', 'h3 a'
        ]
        
        article_links = []
        seen_urls = set()
        
        for selector in article_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href', '')
                if href and href not in seen_urls:
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = self.base_url + href
                    elif not href.startswith('http'):
                        continue
                        
                    seen_urls.add(href)
                    article_links.append((link, href))
                    
                    if len(article_links) >= limit * 3:  # Get extra for filtering
                        break
            
            if len(article_links) >= limit * 2:
                break
        
        return article_links
    
    def extract_image_url(self, link_elem) -> str:
        """Extract image URL from article link element"""
        try:
            # Look for image in parent containers
            for parent in [link_elem.find_parent(), link_elem.find_parent().find_parent() if link_elem.find_parent() else None]:
                if not parent:
                    continue
                    
                img = parent.find('img')
                if img and img.get('src'):
                    img_url = img['src']
                    if img_url.startswith('/'):
                        return self.base_url + img_url
                    elif img_url.startswith('http'):
                        return img_url
                        
            # Look for data-src attribute (lazy loading)
            img = link_elem.find_parent().find('img') if link_elem.find_parent() else None
            if img and img.get('data-src'):
                img_url = img['data-src']
                if img_url.startswith('/'):
                    return self.base_url + img_url
                elif img_url.startswith('http'):
                    return img_url
                    
        except Exception:
            pass
            
        return None
    
    def scrape_articles(self, category: str = None, limit: int = 10, full_content: bool = True) -> Tuple[List[Dict], List[str]]:
        """Scrape articles from Eenadu with proper Telugu text handling"""
        articles = []
        errors = []
        
        try:
            url = self.get_category_url(category)
            response = self.session.get(url, timeout=20)
            response.encoding = 'utf-8'  # Ensure proper encoding
            
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            
            # Extract article links
            article_links = self.extract_article_links(soup, limit)
            
            if not article_links:
                errors.append("No article links found on the page")
                return articles, errors
            
            # Process each article
            for i, (link_elem, article_url) in enumerate(article_links[:limit]):
                if len(articles) >= limit:
                    break
                    
                try:
                    # Extract title with proper Telugu handling
                    title = clean_telugu_text(link_elem.get_text(strip=True))
                    
                    if not is_valid_article_title(title):
                        continue
                    
                    # Get full content if requested
                    content = ""
                    if full_content:
                        content, error = self.extract_article_content(article_url)
                        if error:
                            errors.append(f"Content extraction error for '{title[:50]}...': {error}")
                    
                    # Generate summary
                    if content and len(content) > 200:
                        summary = truncate_text(content, 200)
                    else:
                        summary = truncate_text(title, 150)
                    
                    # Extract image URL
                    image_url = self.extract_image_url(link_elem)
                    
                    # Create article object
                    article = {
                        "title": title,
                        "url": article_url,
                        "published_at": extract_date_from_text(title + " " + content),
                        "summary": summary,
                        "content": content or summary,
                        "image_url": image_url,
                        "author": "Eenadu Desk"
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    errors.append(f"Error processing article {i+1}: {str(e)}")
                    continue
            
            if not articles:
                errors.append("No valid articles could be extracted")
                
        except Exception as e:
            errors.append(f"Main scraping error: {str(e)}")
        
        return articles, errors