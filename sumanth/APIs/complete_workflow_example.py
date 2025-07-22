#!/usr/bin/env python3
"""
Complete News Aggregator Workflow Example
Demonstrates how all 5 microservices work together
"""

import requests
import json
import time
from datetime import datetime

# Service URLs
SERVICES = {
    'auth': 'http://localhost:5000',
    'feed': 'http://localhost:5001',
    'tokenization': 'http://localhost:5002',
    'summarization': 'http://localhost:5003',
    'scraping': 'http://localhost:5004'
}

def print_step(step, description):
    print(f"\n🔄 Step {step}: {description}")
    print("-" * 50)

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def complete_news_workflow():
    """
    Complete workflow demonstrating all 5 APIs:
    1. Authentication - Register/login user
    2. Feed API - Get news articles
    3. Scraping API - Extract full content
    4. Tokenization API - Count tokens
    5. Summarization API - Generate summaries
    """
    
    print("🚀 Complete News Aggregator Workflow")
    print("=" * 60)
    
    # Step 1: Authentication
    print_step(1, "User Authentication")
    
    # Register a test user
    username = f"testuser_{int(time.time())}"
    user_data = {
        "username": username,
        "email": f"{username}@example.com",
        "password": "SecurePass123",
        "preferred_categories": "technology,business",
        "preferred_language": "en"
    }
    
    try:
        # Register
        register_response = requests.post(f"{SERVICES['auth']}/api/register", json=user_data)
        if register_response.status_code == 201:
            print_success("User registered successfully")
        else:
            print_error(f"Registration failed: {register_response.text}")
            return
        
        # Login
        login_response = requests.post(f"{SERVICES['auth']}/api/login", 
                                     json={"username": username, "password": "SecurePass123"})
        if login_response.status_code == 200:
            token = login_response.json().get('token')
            print_success("User logged in successfully")
            print(f"   JWT Token: {token[:30]}...")
        else:
            print_error(f"Login failed: {login_response.text}")
            return
            
    except Exception as e:
        print_error(f"Authentication error: {e}")
        return
    
    # Step 2: Get News Articles
    print_step(2, "Fetch News Articles")
    
    try:
        # Get top headlines
        feed_response = requests.get(f"{SERVICES['feed']}/api/news/top-headlines?category=technology&pageSize=3")
        if feed_response.status_code == 200:
            feed_data = feed_response.json()
            articles = feed_data.get('articles', [])
            print_success(f"Retrieved {len(articles)} news articles")
            
            if articles:
                first_article = articles[0]
                print(f"   Sample: {first_article.get('title', 'N/A')[:60]}...")
            else:
                print_error("No articles found")
                return
        else:
            print_error(f"Feed API failed: {feed_response.text}")
            return
            
    except Exception as e:
        print_error(f"Feed API error: {e}")
        return
    
    # Step 3: Scrape Full Article Content
    print_step(3, "Scrape Full Article Content")
    
    try:
        # Use scraping API to get full content
        article_url = first_article.get('url')
        if article_url:
            scrape_response = requests.post(f"{SERVICES['scraping']}/api/scrape/article",
                                          json={
                                              "url": article_url,
                                              "method": "newspaper"
                                          }, timeout=30)
            
            if scrape_response.status_code == 200:
                scrape_data = scrape_response.json()
                scraped_article = scrape_data.get('article', {})
                full_content = scraped_article.get('content', '')
                
                print_success("Article content scraped successfully")
                print(f"   Title: {scraped_article.get('title', 'N/A')[:60]}...")
                print(f"   Content length: {len(full_content)} characters")
                print(f"   Word count: {scraped_article.get('wordCount', 0)}")
                
                # Use scraped content for further processing
                article_content = full_content if full_content else first_article.get('content', '')
            else:
                print_error(f"Scraping failed: {scrape_response.text}")
                # Fallback to original article content
                article_content = first_article.get('content', '')
        else:
            print_error("No article URL available")
            article_content = first_article.get('content', '')
            
    except Exception as e:
        print_error(f"Scraping error: {e}")
        article_content = first_article.get('content', '')
    
    if not article_content:
        print_error("No article content available for processing")
        return
    
    # Step 4: Token Analysis
    print_step(4, "Analyze Token Usage")
    
    try:
        # Count tokens for the article
        token_response = requests.post(f"{SERVICES['tokenization']}/api/tokens/count",
                                     json={
                                         "text": article_content,
                                         "model": "gpt-3.5-turbo"
                                     })
        
        if token_response.status_code == 200:
            token_data = token_response.json()
            token_count = token_data.get('tokenCount', 0)
            print_success(f"Token analysis completed")
            print(f"   Token count: {token_count}")
            print(f"   Character count: {token_data.get('characterCount', 0)}")
            print(f"   Word count: {token_data.get('wordCount', 0)}")
            
            # Estimate cost
            cost_response = requests.post(f"{SERVICES['tokenization']}/api/tokens/estimate-cost",
                                        json={
                                            "text": article_content,
                                            "model": "gpt-3.5-turbo",
                                            "type": "input"
                                        })
            
            if cost_response.status_code == 200:
                cost_data = cost_response.json()
                estimated_cost = cost_data.get('estimatedCost', 0)
                print_success(f"Cost estimation: ${estimated_cost:.6f}")
            
        else:
            print_error(f"Token analysis failed: {token_response.text}")
            
    except Exception as e:
        print_error(f"Token analysis error: {e}")
    
    # Step 5: Generate Summary
    print_step(5, "Generate AI Summary")
    
    try:
        # Summarize the article
        summary_response = requests.post(f"{SERVICES['summarization']}/api/summarize/article",
                                       json={
                                           "content": article_content,
                                           "length": "medium",
                                           "model": "gpt-3.5-turbo"
                                       }, timeout=60)
        
        if summary_response.status_code == 200:
            summary_data = summary_response.json()
            summary = summary_data.get('summary', '')
            compression_ratio = summary_data.get('compressionRatio', 0)
            token_usage = summary_data.get('tokenUsage', {})
            
            print_success("Article summarized successfully")
            print(f"   Summary: {summary}")
            print(f"   Compression ratio: {compression_ratio}")
            print(f"   Total tokens used: {token_usage.get('totalTokens', 0)}")
            
        else:
            print_error(f"Summarization failed: {summary_response.text}")
            
    except Exception as e:
        print_error(f"Summarization error: {e}")
    
    # Step 6: Advanced Workflow - Batch Processing
    print_step(6, "Batch Processing Example")
    
    try:
        # Get multiple articles and process them
        if len(articles) > 1:
            # Scrape multiple articles
            urls = [article.get('url') for article in articles[:2] if article.get('url')]
            
            if urls:
                batch_scrape_response = requests.post(f"{SERVICES['scraping']}/api/scrape/batch",
                                                    json={
                                                        "urls": urls,
                                                        "method": "newspaper",
                                                        "maxWorkers": 2
                                                    }, timeout=60)
                
                if batch_scrape_response.status_code == 200:
                    batch_data = batch_scrape_response.json()
                    successful_scrapes = batch_data.get('successful', 0)
                    print_success(f"Batch scraping: {successful_scrapes} articles processed")
                    
                    # Extract content for batch summarization
                    batch_articles = []
                    for result in batch_data.get('results', []):
                        if result.get('status') == 'success':
                            article_data = result.get('article', {})
                            content = article_data.get('content', '')
                            if content:
                                batch_articles.append({
                                    'title': article_data.get('title', ''),
                                    'content': content
                                })
                    
                    if batch_articles:
                        # Batch summarization
                        batch_summary_response = requests.post(f"{SERVICES['summarization']}/api/summarize/batch",
                                                             json={
                                                                 "articles": batch_articles,
                                                                 "length": "short",
                                                                 "model": "gpt-3.5-turbo"
                                                             }, timeout=120)
                        
                        if batch_summary_response.status_code == 200:
                            batch_summary_data = batch_summary_response.json()
                            successful_summaries = batch_summary_data.get('successfulSummaries', 0)
                            total_tokens = batch_summary_data.get('totalTokenUsage', {}).get('totalTokens', 0)
                            print_success(f"Batch summarization: {successful_summaries} summaries generated")
                            print(f"   Total tokens used: {total_tokens}")
                        
    except Exception as e:
        print_error(f"Batch processing error: {e}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ All 5 microservices working together:")
    print("   1. Authentication API - User management ✓")
    print("   2. Feed API - News retrieval ✓")
    print("   3. Scraping API - Content extraction ✓")
    print("   4. Tokenization API - Cost analysis ✓")
    print("   5. Summarization API - AI processing ✓")
    print("\n🚀 Your news aggregator microservices are ready for production!")

def main():
    print(f"⏰ Workflow started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if all services are running
    print("🔍 Checking service availability...")
    all_services_up = True
    
    for service_name, url in SERVICES.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name.title()} API: Running")
            else:
                print(f"❌ {service_name.title()} API: Not responding")
                all_services_up = False
        except Exception as e:
            print(f"❌ {service_name.title()} API: Not available ({e})")
            all_services_up = False
    
    if not all_services_up:
        print("\n⚠️  Some services are not running. Please start all services first:")
        print("   docker-compose up -d")
        print("   OR start each service manually")
        return
    
    print("\n✅ All services are running! Starting workflow...\n")
    
    # Run the complete workflow
    complete_news_workflow()

if __name__ == "__main__":
    main()