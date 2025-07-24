from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import requests
import yaml
import re
import time
from collections import deque

with open('scraper_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def normalize_url(url):
    """Normalize URL to avoid duplicates"""
    parsed = urlparse(url)
    # Remove fragments, query parameters, and trailing slashes
    path = parsed.path.rstrip('/') or '/'
    return urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))

def is_valid_url(url, base_domain):
    """Check if URL is valid for scraping"""
    try:
        parsed = urlparse(url)
        
        # Must be same domain
        if parsed.netloc != base_domain:
            return False
            
        # Skip certain file types
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx', '.xls', '.xlsx']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        # Skip certain URL patterns
        skip_patterns = ['#', 'mailto:', 'tel:', 'javascript:', 'ftp:']
        if any(url.lower().startswith(pattern) for pattern in skip_patterns):
            return False
            
        return True
    except:
        return False

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common navigation/footer text patterns
    noise_patterns = [
        r'Toggle navigation.*?',
        r'Copyright.*?All rights reserved.*?',
        r'Follow Us.*?',
        r'Quick Links.*?',
        r'×.*?',
        r'Cookie.*?',
        r'Privacy Policy.*?',
        r'Terms.*?Service.*?',
        r'Skip to.*?',
        r'Menu.*?',
        r'Navigation.*?',
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove excessive repetition
    text = re.sub(r'(.{30,}?)\1{2,}', r'\1', text)
    
    return text.strip()

def extract_content_and_links(soup, base_url):
    """Extract meaningful content and all valid links from HTML"""
    # Remove script, style, nav, footer elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Get title
    title = soup.title.get_text(strip=True) if soup.title else ''
    
    # Priority content selectors
    content_selectors = [
        'main', 'article', '.content', '#content', 
        '.main-content', '.page-content', '.post-content',
        '.container', '.wrapper'
    ]
    
    content_text = ""
    
    # Try to find main content area
    for selector in content_selectors:
        content_area = soup.select_one(selector)
        if content_area:
            content_text = content_area.get_text(separator='\n', strip=True)
            break
    
    # If no main content found, use body but filter out noise
    if not content_text:
        # Remove common noise elements
        for element in soup(['nav', 'aside', '.sidebar', '.menu', '.navigation', '.footer']):
            element.decompose()
        
        content_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ''
    
    # Extract all links
    links = set()
    base_domain = urlparse(base_url).netloc
    
    for link in soup.find_all('a', href=True):
        href = link['href'].strip()
        if not href:
            continue
            
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        normalized_url = normalize_url(full_url)
        
        # Check if URL is valid for scraping
        if is_valid_url(normalized_url, base_domain):
            links.add(normalized_url)
    
    return title, content_text, links

class RecursiveScraper:
    def __init__(self):
        self.visited_urls = set()
        self.discovered_urls = set()
        self.content_hashes = set()
        self.page_contents = []
        self.failed_urls = set()
        
    def discover_all_links(self, start_url, user_agent, max_discovery=50):
        """First pass: discover all unique links on the website"""
        print(f"🔍 Discovering all links from {start_url}...")
        
        to_visit = deque([normalize_url(start_url)])
        base_domain = urlparse(start_url).netloc
        base_url = start_url.rstrip('/')
        discovered_count = 0
        
        while to_visit and discovered_count < max_discovery:
            url = to_visit.popleft()
            
            if url in self.discovered_urls:
                continue
                
            self.discovered_urls.add(url)
            discovered_count += 1
            
            try:
                print(f"  Discovering links from: {url}")
                headers = {'User-Agent': user_agent}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                _, _, links = extract_content_and_links(soup, url)
                
                # Add new links to discovery queue
                new_links = links - self.discovered_urls
                for link in new_links:
                    if link not in to_visit:
                        to_visit.append(link)
                        print(f"    → Found: {link}")
                
                # Also check for common HTML page patterns that might not be linked
                common_pages = ['about.html', 'services.html', 'contact.html', 'cci-chat.html', 'team.html']
                for page in common_pages:
                    full_page_url = f"{base_url}/{page}"
                    normalized_page_url = normalize_url(full_page_url)
                    if (normalized_page_url not in self.discovered_urls and 
                        normalized_page_url not in to_visit):
                        to_visit.append(normalized_page_url)
                        print(f"    → Added common page: {normalized_page_url}")
                
                # Small delay to be respectful
                time.sleep(0.2)
                
            except Exception as e:
                print(f"    ✗ Failed to discover from {url}: {e}")
                self.failed_urls.add(url)
        
        print(f"✓ Discovery complete! Found {len(self.discovered_urls)} unique URLs")
        return self.discovered_urls
    
    def scrape_discovered_urls(self, user_agent, max_pages=None):
        """Second pass: scrape content from all discovered URLs"""
        if max_pages is None:
            max_pages = len(self.discovered_urls)
        
        print(f"📄 Scraping content from discovered URLs (max {max_pages})...")
        
        urls_to_scrape = list(self.discovered_urls)[:max_pages]
        
        for i, url in enumerate(urls_to_scrape, 1):
            if url in self.visited_urls or url in self.failed_urls:
                continue
                
            try:
                print(f"  [{i}/{len(urls_to_scrape)}] Scraping: {url}")
                headers = {'User-Agent': user_agent}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                title, content, _ = extract_content_and_links(soup, url)
                
                # Clean the content
                clean_title = clean_text(title)
                clean_content = clean_text(content)
                
                # Check for duplicate content using hash
                content_hash = hash(clean_content)
                if content_hash not in self.content_hashes and len(clean_content) > 100:
                    self.content_hashes.add(content_hash)
                    
                    # Store structured content
                    page_data = {
                        'url': url,
                        'title': clean_title,
                        'content': clean_content,
                        'content_length': len(clean_content)
                    }
                    self.page_contents.append(page_data)
                    print(f"    ✓ Added: {clean_title} ({len(clean_content)} chars)")
                else:
                    print(f"    ✗ Skipped: duplicate/short content")
                
                self.visited_urls.add(url)
                
                # Small delay to be respectful
                time.sleep(0.3)
                
            except Exception as e:
                print(f"    ✗ Failed to scrape {url}: {e}")
                self.failed_urls.add(url)
        
        return self.page_contents
    
    def scrape_website(self, start_url, user_agent, max_discovery=50, max_pages=None):
        """Complete scraping process: discover then scrape"""
        print(f"🚀 Starting comprehensive website scraping...")
        print(f"   Start URL: {start_url}")
        print(f"   Max discovery: {max_discovery}")
        print(f"   Max pages to scrape: {max_pages or 'all discovered'}")
        print("-" * 60)
        
        # Step 1: Discover all links
        self.discover_all_links(start_url, user_agent, max_discovery)
        
        print("-" * 60)
        
        # Step 2: Scrape content from discovered URLs
        self.scrape_discovered_urls(user_agent, max_pages)
        
        print("-" * 60)
        print(f"✅ Scraping completed!")
        print(f"   URLs discovered: {len(self.discovered_urls)}")
        print(f"   URLs scraped: {len(self.visited_urls)}")
        print(f"   Unique content pages: {len(self.page_contents)}")
        print(f"   Failed URLs: {len(self.failed_urls)}")
        
        return self.page_contents
    
    def save_clean_content(self, filename='scraped_content.txt'):
        """Save cleaned content to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, page in enumerate(self.page_contents, 1):
                f.write(f"=== PAGE {i}: {page['title']} ===\n")
                f.write(f"URL: {page['url']}\n")
                f.write(f"Content Length: {page['content_length']} characters\n\n")
                f.write(f"{page['content']}\n\n")
                f.write("=" * 70 + "\n\n")
        
        print(f"💾 Saved {len(self.page_contents)} pages to {filename}")
        return filename
    
    def print_summary(self):
        """Print detailed summary of scraping results"""
        print("\n" + "="*70)
        print("SCRAPING SUMMARY")
        print("="*70)
        
        print(f"📊 Statistics:")
        print(f"   • URLs discovered: {len(self.discovered_urls)}")
        print(f"   • URLs successfully scraped: {len(self.visited_urls)}")
        print(f"   • Unique content pages: {len(self.page_contents)}")
        print(f"   • Failed URLs: {len(self.failed_urls)}")
        
        if self.page_contents:
            total_chars = sum(page['content_length'] for page in self.page_contents)
            avg_chars = total_chars // len(self.page_contents)
            print(f"   • Total content: {total_chars:,} characters")
            print(f"   • Average per page: {avg_chars:,} characters")
        
        print(f"\n📄 Successfully scraped pages:")
        for i, page in enumerate(self.page_contents, 1):
            print(f"   {i:2d}. {page['title']} ({page['content_length']:,} chars)")
        
        if self.failed_urls:
            print(f"\n❌ Failed URLs:")
            for url in sorted(self.failed_urls):
                print(f"   • {url}")

# Run the recursive scraper
if __name__ == "__main__":
    url = config['home_page']
    user_agent = config['message']
    max_pages = config.get('max_pages', 10)
    
    scraper = RecursiveScraper()
    
    # Scrape with discovery phase
    scraped_data = scraper.scrape_website(
        start_url=url,
        user_agent=user_agent,
        max_discovery=30,  # Discover up to 30 unique URLs
        max_pages=max_pages  # Then scrape up to max_pages
    )
    
    # Save clean content
    content_file = scraper.save_clean_content('scraped_content.txt')
    
    # Print detailed summary
    scraper.print_summary()