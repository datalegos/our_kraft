from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import requests
import yaml

with open('scraper_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def normalize_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip('/') or '/'
    return urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))

class Scraper:
    def scrape_site(self, start_url, message, max_pages=10):
        all_links_seen = set()
        to_visit = [normalize_url(start_url)]
        all_content = ""
        count = 0

        while to_visit and count < max_pages:
            url = to_visit.pop(0)
            if url in all_links_seen:
                continue
            all_links_seen.add(url)
            try:
                headers = {'User-Agent': message}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.get_text(strip=True) if soup.title else ''
                body = soup.body.get_text(strip=True) if soup.body else ''
                all_content += f"{title}\n{body}\n\n"
                count += 1

                for link in soup.find_all('a', href=True):
                    link_url = urljoin(url, link['href'])
                    link_url = normalize_url(link_url)
                    if urlparse(link_url).netloc == urlparse(start_url).netloc:
                        if link_url not in all_links_seen:
                            to_visit.append(link_url)
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

        return all_content

url = config['home_page']
message = config['message']
max_pages = config['max_pages']
scraper = Scraper()
scraped_content = scraper.scrape_site(url, message, max_pages)

with open('content.txt', 'w', encoding='utf-8') as f:
    f.write(scraped_content)
