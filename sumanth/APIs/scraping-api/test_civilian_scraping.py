#!/usr/bin/env python3
"""
Test script for Civilian News Scraping API
Tests home page blog discovery and article extraction
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = 'http://localhost:5004'

def print_header(title):
    print(f"\n{'='*50}")
    print(f"📰 {title}")
    print(f"{'='*50}")

def print_test(test_name):
    print(f"\n🔍 Testing: {test_name}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def test_health_check():
    print_test("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Civilian News Scraper is healthy: {data}")
        else:
            print_error(f"Health check failed: {response.status_code}")
    except Exception as e:
        print_error(f"Health check failed: {e}")

def test_discover_blogs():
    print_test("Home Page Blog Discovery")
    
    # Test each civilian news source
    sources = ['bbc', 'cnn', 'reuters']
    
    for source in sources:
        print(f"\n📡 Discovering blogs from: {source.upper()}")
        
        try:
            response = requests.post(f"{BASE_URL}/api/discover-blogs",
                                   json={
                                       "source": source,
                                       "maxArticles": 5
                                   }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print_success(f"Found {len(articles)} civilian-relevant articles")
                
                # Show discovered articles
                for i, article in enumerate(articles[:3]):
                    title = article.get('title', 'N/A')[:60]
                    url = article.get('url', 'N/A')[:80]
                    print(f"   {i+1}. {title}...")
                    print(f"      URL: {url}")
            else:
                print_error(f"Blog discovery failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print_error(f"Blog discovery error: {e}")
        
        time.sleep(2)  # Be respectful with requests

def test_scrape_article():
    print_test("Full Article Scraping")
    
    # First discover some articles
    print("🔍 Discovering articles to scrape...")
    
    try:
        response = requests.post(f"{BASE_URL}/api/discover-blogs",
                               json={
                                   "source": "bbc",
                                   "maxArticles": 3
                               }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            if articles:
                # Scrape the first discovered article
                test_url = articles[0]['url']
                print(f"📰 Scraping article: {test_url}")
                
                scrape_response = requests.post(f"{BASE_URL}/api/scrape-article",
                                              json={"url": test_url}, timeout=30)
                
                if scrape_response.status_code == 200:
                    article_data = scrape_response.json()
                    article = article_data.get('article', {})
                    
                    print_success("Article scraped successfully")
                    print(f"   Title: {article.get('title', 'N/A')[:80]}...")
                    print(f"   Content length: {len(article.get('content', ''))}")
                    print(f"   Word count: {article.get('wordCount', 0)}")
                    print(f"   Authors: {article.get('authors', [])}")
                    print(f"   Summary: {article.get('summary', 'N/A')[:100]}...")
                    print(f"   Civilian relevant: {article.get('civilian_relevant', False)}")
                else:
                    print_error(f"Article scraping failed: {scrape_response.status_code}")
            else:
                print_error("No articles found to scrape")
        else:
            print_error("Failed to discover articles for scraping test")
            
    except Exception as e:
        print_error(f"Article scraping test error: {e}")

def test_batch_scraping():
    print_test("Batch Article Scraping")
    
    # Discover articles from multiple sources
    print("🔍 Discovering articles from multiple sources...")
    
    all_urls = []
    sources = ['bbc', 'cnn']
    
    for source in sources:
        try:
            response = requests.post(f"{BASE_URL}/api/discover-blogs",
                                   json={
                                       "source": source,
                                       "maxArticles": 2
                                   }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                for article in articles:
                    all_urls.append(article['url'])
        except:
            continue
    
    if all_urls:
        print(f"📰 Batch scraping {len(all_urls)} articles...")
        
        try:
            response = requests.post(f"{BASE_URL}/api/scrape-batch",
                                   json={
                                       "urls": all_urls[:4],  # Limit to 4 for testing
                                       "maxWorkers": 2
                                   }, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Batch scraping completed")
                print(f"   Total URLs: {data.get('totalUrls', 0)}")
                print(f"   Successful: {data.get('successful', 0)}")
                print(f"   Failed: {data.get('failed', 0)}")
                
                # Show results summary
                for result in data.get('results', [])[:3]:
                    status = result.get('status', 'unknown')
                    url = result.get('url', 'N/A')[:60]
                    if status == 'success':
                        article = result.get('article', {})
                        title = article.get('title', 'N/A')[:50]
                        print(f"   ✅ {title}...")
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"   ❌ {url}: {error}")
            else:
                print_error(f"Batch scraping failed: {response.status_code}")
        except Exception as e:
            print_error(f"Batch scraping error: {e}")
    else:
        print_error("No URLs found for batch scraping test")

def test_civilian_workflow():
    print_test("Complete Civilian News Workflow")
    
    print("🔄 Testing complete civilian news aggregation workflow...")
    
    try:
        # Step 1: Discover civilian-relevant blogs
        print("Step 1: Discovering civilian-relevant blogs...")
        
        discovery_response = requests.post(f"{BASE_URL}/api/discover-blogs",
                                         json={
                                             "source": "bbc",
                                             "maxArticles": 3
                                         }, timeout=30)
        
        if discovery_response.status_code == 200:
            articles = discovery_response.json().get('articles', [])
            print_success(f"✓ Found {len(articles)} civilian-relevant articles")
            
            if articles:
                # Step 2: Scrape the most relevant article
                print("Step 2: Scraping most relevant article...")
                top_article_url = articles[0]['url']
                
                scrape_response = requests.post(f"{BASE_URL}/api/scrape-article",
                                              json={"url": top_article_url}, timeout=30)
                
                if scrape_response.status_code == 200:
                    article_data = scrape_response.json()
                    article = article_data.get('article', {})
                    
                    print_success("✓ Article scraped and filtered for civilians")
                    print(f"   Title: {article.get('title', 'N/A')[:60]}...")
                    print(f"   Content: {len(article.get('content', ''))} characters")
                    print(f"   Summary: {article.get('summary', 'N/A')[:80]}...")
                    print(f"   Civilian Focus: {article.get('civilian_relevant', False)}")
                    print(f"   Content Type: {article.get('content_type', 'N/A')}")
                    
                    print_success("🎉 Civilian news workflow completed successfully!")
                    print("📋 Ready for news aggregator consumption!")
                else:
                    print_error("Step 2 failed: Article scraping")
            else:
                print_error("No civilian-relevant articles found")
        else:
            print_error("Step 1 failed: Blog discovery")
            
    except Exception as e:
        print_error(f"Civilian workflow error: {e}")

def main():
    print("📰 Starting Civilian News Scraping API Test Suite")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Testing API at: {BASE_URL}")
    print("🏠 Focus: Home page blogs with civilian-relevant content")
    
    print_header("CIVILIAN NEWS SCRAPER TESTS")
    
    # Run all tests
    test_health_check()
    test_discover_blogs()
    test_scrape_article()
    test_batch_scraping()
    test_civilian_workflow()
    
    print_header("TEST SUMMARY")
    print("🎯 All civilian news scraping tests completed!")
    print("📋 Check the results above for any failures")
    print("🔧 Make sure the scraping API is running on port 5004")
    
    print("\n💡 To start the civilian news scraper:")
    print("   cd scraping-api && python app.py")
    
    print("\n🎯 What this scraper provides:")
    print("   ✅ Home page blog discovery from trusted sources")
    print("   ✅ Full article content extraction")
    print("   ✅ Civilian-focused content filtering")
    print("   ✅ Batch processing for efficiency")
    print("   ✅ Clean, structured data for news aggregators")
    
    print("\n📰 Supported civilian news sources:")
    print("   • BBC News (politics, health, education)")
    print("   • CNN (national, world, politics)")
    print("   • Reuters (world, business, environment)")

if __name__ == "__main__":
    main()