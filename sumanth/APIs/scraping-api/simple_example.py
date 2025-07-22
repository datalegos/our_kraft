#!/usr/bin/env python3
"""
Simple example of how to use the Civilian News Scraper
Shows basic usage and where content goes
"""

import requests
import json
import time

# API endpoint
BASE_URL = 'http://localhost:5004'

def discover_civilian_news():
    """Step 1: Discover civilian-relevant articles from home pages"""
    print("🔍 Step 1: Discovering civilian news articles...")
    
    # Available sources: bbc, cnn, reuters
    sources = ['bbc', 'cnn']
    all_articles = []
    
    for source in sources:
        print(f"\n📰 Discovering from {source.upper()}...")
        
        try:
            response = requests.post(f"{BASE_URL}/api/discover-blogs", 
                                   json={
                                       "source": source,
                                       "maxArticles": 5
                                   }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                all_articles.extend(articles)
                
                print(f"✅ Found {len(articles)} articles")
                for i, article in enumerate(articles[:3]):
                    print(f"   {i+1}. {article['title'][:60]}...")
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)  # Be respectful
    
    return all_articles

def scrape_full_articles(article_urls):
    """Step 2: Scrape full content from discovered articles"""
    print(f"\n📖 Step 2: Scraping full content from {len(article_urls)} articles...")
    
    scraped_articles = []
    
    try:
        # Batch scrape for efficiency
        response = requests.post(f"{BASE_URL}/api/scrape-batch",
                               json={
                                   "urls": article_urls,
                                   "maxWorkers": 2
                               }, timeout=90)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            print(f"✅ Batch scraping completed")
            print(f"   Successful: {data.get('successful', 0)}")
            print(f"   Failed: {data.get('failed', 0)}")
            
            for result in results:
                if result['status'] == 'success':
                    article = result['article']
                    scraped_articles.append(article)
                    
                    print(f"\n📄 Article: {article['title'][:50]}...")
                    print(f"   Content: {article['wordCount']} words")
                    print(f"   Authors: {', '.join(article.get('authors', ['Unknown']))}")
                    print(f"   Civilian relevant: {article.get('civilian_relevant', False)}")
        else:
            print(f"❌ Batch scraping failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Scraping error: {e}")
    
    return scraped_articles

def save_to_file(articles, filename="civilian_news.json"):
    """Step 3: Save scraped content to file"""
    print(f"\n💾 Step 3: Saving {len(articles)} articles to {filename}...")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Articles saved to {filename}")
        print(f"   File size: {len(json.dumps(articles, indent=2))} characters")
        
        # Show summary
        total_words = sum(article.get('wordCount', 0) for article in articles)
        civilian_count = sum(1 for article in articles if article.get('civilian_relevant', False))
        
        print(f"\n📊 Content Summary:")
        print(f"   Total articles: {len(articles)}")
        print(f"   Total words: {total_words:,}")
        print(f"   Civilian relevant: {civilian_count}/{len(articles)}")
        
    except Exception as e:
        print(f"❌ Save error: {e}")

def main():
    print("📰 Civilian News Scraper - Simple Example")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Scraper API is not running!")
            print("   Start it with: python app.py")
            return
    except:
        print("❌ Cannot connect to scraper API!")
        print("   Make sure it's running on port 5004")
        return
    
    print("✅ Scraper API is running\n")
    
    # Complete workflow
    try:
        # Step 1: Discover articles
        discovered_articles = discover_civilian_news()
        
        if not discovered_articles:
            print("❌ No articles discovered")
            return
        
        # Step 2: Scrape full content
        article_urls = [article['url'] for article in discovered_articles[:5]]  # Limit to 5
        scraped_articles = scrape_full_articles(article_urls)
        
        if not scraped_articles:
            print("❌ No articles scraped successfully")
            return
        
        # Step 3: Save to file
        save_to_file(scraped_articles)
        
        print("\n🎉 Complete! Your civilian news content is ready!")
        print("📁 Check the 'civilian_news.json' file for the scraped content")
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()