"""
Web scraper for collecting content from websites.
"""
import time
import yaml
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Set, Optional, Tuple
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chatbot.core.config import (
    SCRAPER_CONFIG_FILE,
    SCRAPER_TIMEOUT,
    SCRAPER_MAX_RETRIES,
    SCRAPER_DELAY,
    SCRAPER_CONTENT_FILE,
)
from chatbot.utils.logger import logger
from chatbot.utils.exceptions import ScraperError, ConfigurationError


def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing query parameters and fragments.
    
    Args:
        url: URL to normalize
    
    Returns:
        Normalized URL
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/') or '/'
        return urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))
    except Exception as e:
        logger.warning(f"Error normalizing URL {url}: {e}")
        return url


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    return ' '.join(lines)


class Scraper:
    """Web scraper with rate limiting and error handling."""
    
    def __init__(
        self,
        timeout: int = None,
        max_retries: int = None,
        delay: float = None,
        user_agent: str = None
    ):
        """
        Initialize the scraper.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            delay: Delay between requests in seconds
            user_agent: User agent string for requests
        """
        self.timeout = timeout or SCRAPER_TIMEOUT
        self.max_retries = max_retries or SCRAPER_MAX_RETRIES
        self.delay = delay or SCRAPER_DELAY
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Scraper initialized: timeout={self.timeout}s, retries={self.max_retries}, delay={self.delay}s")
    
    def _extract_text(self, soup: BeautifulSoup, url: str) -> Tuple[str, str]:
        """
        Extract title and body text from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            url: URL of the page (for logging)
        
        Returns:
            Tuple of (title, body_text)
        """
        # Extract title
        title = ""
        if soup.title:
            title = clean_text(soup.title.get_text())
        elif soup.find('h1'):
            title = clean_text(soup.find('h1').get_text())
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Extract body text
        body = ""
        if soup.body:
            body = clean_text(soup.body.get_text())
        else:
            body = clean_text(soup.get_text())
        
        if not title and not body:
            logger.warning(f"No content extracted from {url}")
        
        return title, body
    
    def scrape_page(self, url: str) -> Optional[Tuple[str, str, str]]:
        """
        Scrape a single page.
        
        Args:
            url: URL to scrape
        
        Returns:
            Tuple of (url, title, body) or None if failed
        """
        try:
            headers = {'User-Agent': self.user_agent}
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                logger.debug(f"Skipping non-HTML content at {url}: {content_type}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            title, body = self._extract_text(soup, url)
            
            return (url, title, body)
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout while scraping {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error while scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while scraping {url}: {e}", exc_info=True)
            return None
    
    def scrape_site(
        self,
        start_url: str,
        max_pages: int = 10,
        same_domain_only: bool = True
    ) -> str:
        """
        Scrape a website starting from a given URL.
        
        Args:
            start_url: Starting URL
            max_pages: Maximum number of pages to scrape
            same_domain_only: Only scrape pages from the same domain
        
        Returns:
            Combined content from all scraped pages
        """
        all_links_seen: Set[str] = set()
        to_visit: List[str] = [normalize_url(start_url)]
        all_content_parts: List[str] = []
        count = 0
        start_domain = urlparse(start_url).netloc
        
        logger.info(f"Starting scrape from {start_url} (max {max_pages} pages)")
        
        while to_visit and count < max_pages:
            url = to_visit.pop(0)
            
            if url in all_links_seen:
                continue
            
            all_links_seen.add(url)
            
            # Rate limiting
            if count > 0:
                time.sleep(self.delay)
            
            # Scrape the page
            result = self.scrape_page(url)
            
            if result:
                page_url, title, body = result
                if title or body:
                    content = f"URL: {page_url}\n"
                    if title:
                        content += f"Title: {title}\n"
                    if body:
                        content += f"Content: {body}\n"
                    content += "\n" + "="*80 + "\n\n"
                    all_content_parts.append(content)
                    count += 1
                    logger.info(f"Scraped page {count}/{max_pages}: {page_url}")
            
            # Extract links for next pages
            if count < max_pages:
                try:
                    headers = {'User-Agent': self.user_agent}
                    response = self.session.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        link_url = urljoin(url, link['href'])
                        link_url = normalize_url(link_url)
                        
                        # Filter by domain if requested
                        if same_domain_only:
                            if urlparse(link_url).netloc != start_domain:
                                continue
                        
                        # Add to queue if not seen
                        if link_url not in all_links_seen and link_url not in to_visit:
                            to_visit.append(link_url)
                            
                except Exception as e:
                    logger.debug(f"Error extracting links from {url}: {e}")
        
        all_content = "".join(all_content_parts)
        logger.info(f"Scraping complete: {count} pages scraped, {len(all_content)} characters collected")
        
        return all_content


def load_scraper_config(config_file: str = None) -> dict:
    """
    Load scraper configuration from YAML file.
    
    Args:
        config_file: Path to config file (defaults to SCRAPER_CONFIG_FILE)
    
    Returns:
        Configuration dictionary
    """
    config_file = config_file or SCRAPER_CONFIG_FILE
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise ConfigurationError(f"Scraper config file not found: {config_file}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate required keys
        required_keys = ['home_page', 'message', 'max_pages']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ConfigurationError(
                f"Missing required keys in config: {', '.join(missing_keys)}"
            )
        
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML config: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config: {e}")


def main():
    """Main entry point for the scraper."""
    try:
        # Load configuration
        config = load_scraper_config()
        url = config['home_page']
        user_agent = config['message']
        max_pages = config.get('max_pages', 10)
        output_file = config.get('output_file', SCRAPER_CONTENT_FILE)
        
        # Create scraper and scrape
        scraper = Scraper(user_agent=user_agent)
        scraped_content = scraper.scrape_site(url, max_pages=max_pages)
        
        # Save content
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(scraped_content)
        
        logger.info(f"Content saved to {output_file}")
        print(f"✅ Successfully scraped {max_pages} pages and saved to {output_file}")
        
    except (ConfigurationError, ScraperError) as e:
        logger.error(f"Scraper error: {e}")
        print(f"\n❌ Error: {e}\n")
    except KeyboardInterrupt:
        logger.info("Scraper interrupted by user")
        print("\n⚠️  Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
