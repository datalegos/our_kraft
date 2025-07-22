#!/usr/bin/env python3
"""
Comprehensive test script for all News Aggregator microservices
Tests authentication, feed fetching, tokenization, and summarization APIs
"""

import requests
import json
import time
import sys
from datetime import datetime

# Service URLs
SERVICES = {
    'auth': 'http://localhost:5000',
    'feed': 'http://localhost:5001', 
    'tokenization': 'http://localhost:5002',
    'summarization': 'http://localhost:5003',
    'scraping': 'http://localhost:5004'
}

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_test(test_name):
    print(f"\n🔍 Testing: {test_name}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def test_service_health():
    print_header("SERVICE HEALTH CHECKS")
    
    for service_name, base_url in SERVICES.items():
        print_test(f"{service_name.title()} API Health")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print_success(f"{service_name.title()} API is healthy: {data}")
            else:
                print_error(f"{service_name.title()} API health check failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print_error(f"{service_name.title()} API is not responding: {e}")

def test_authentication_api():
    print_header("AUTHENTICATION API TESTS")
    
    # Test user registration
    print_test("User Registration")
    register_data = {
        "username": "testuser_" + str(int(time.time())),
        "email": f"test_{int(time.time())}@example.com",
        "password": "SecurePass123",
        "preferred_categories": "technology,business,sports",
        "preferred_language": "en"
    }
    
    try:
        response = requests.post(f"{SERVICES['auth']}/api/register", json=register_data)
        if response.status_code == 201:
            print_success("User registered successfully")
            user_data = response.json()
        else:
            print_error(f"Registration failed: {response.text}")
            return None
    except Exception as e:
        print_error(f"Registration request failed: {e}")
        return None
    
    # Test user login
    print_test("User Login")
    login_data = {
        "username": register_data["username"],
        "password": register_data["password"]
    }
    
    try:
        response = requests.post(f"{SERVICES['auth']}/api/login", json=login_data)
        if response.status_code == 200:
            login_result = response.json()
            token = login_result.get('token')
            print_success(f"Login successful, token received")
            return token
        else:
            print_error(f"Login failed: {response.text}")
            return None
    except Exception as e:
        print_error(f"Login request failed: {e}")
        return None

def test_feed_api():
    print_header("FEED API TESTS")
    
    # Test top headlines
    print_test("Top Headlines")
    try:
        response = requests.get(f"{SERVICES['feed']}/api/news/top-headlines?category=technology&pageSize=3")
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print_success(f"Retrieved {len(articles)} top headlines")
            if articles:
                print(f"   Sample headline: {articles[0].get('title', 'N/A')[:80]}...")
                return articles[0]  # Return first article for summarization test
        else:
            print_error(f"Top headlines request failed: {response.text}")
    except Exception as e:
        print_error(f"Top headlines request failed: {e}")
    
    # Test news search
    print_test("News Search")
    try:
        response = requests.get(f"{SERVICES['feed']}/api/news/search?q=artificial intelligence&pageSize=2")
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print_success(f"Found {len(articles)} articles for 'artificial intelligence'")
        else:
            print_error(f"News search failed: {response.text}")
    except Exception as e:
        print_error(f"News search request failed: {e}")
    
    # Test news sources
    print_test("News Sources")
    try:
        response = requests.get(f"{SERVICES['feed']}/api/news/sources?category=technology")
        if response.status_code == 200:
            data = response.json()
            sources = data.get('sources', [])
            print_success(f"Retrieved {len(sources)} technology news sources")
        else:
            print_error(f"News sources request failed: {response.text}")
    except Exception as e:
        print_error(f"News sources request failed: {e}")
    
    return None

def test_tokenization_api():
    print_header("TOKENIZATION API TESTS")
    
    sample_text = "This is a sample news article about artificial intelligence and machine learning technologies that are transforming the world."
    
    # Test token counting
    print_test("Token Counting")
    try:
        response = requests.post(f"{SERVICES['tokenization']}/api/tokens/count", 
                               json={"text": sample_text, "model": "gpt-4"})
        if response.status_code == 200:
            data = response.json()
            token_count = data.get('tokenCount', 0)
            print_success(f"Token count: {token_count} tokens for GPT-4")
        else:
            print_error(f"Token counting failed: {response.text}")
    except Exception as e:
        print_error(f"Token counting request failed: {e}")
    
    # Test cost estimation
    print_test("Cost Estimation")
    try:
        response = requests.post(f"{SERVICES['tokenization']}/api/tokens/estimate-cost",
                               json={"text": sample_text, "model": "gpt-3.5-turbo", "type": "input"})
        if response.status_code == 200:
            data = response.json()
            cost = data.get('estimatedCost', 0)
            print_success(f"Estimated cost: ${cost:.6f} for GPT-3.5-turbo input")
        else:
            print_error(f"Cost estimation failed: {response.text}")
    except Exception as e:
        print_error(f"Cost estimation request failed: {e}")
    
    # Test batch counting
    print_test("Batch Token Counting")
    try:
        texts = [
            "First news article about technology.",
            "Second article about business trends.",
            "Third article covering sports news."
        ]
        response = requests.post(f"{SERVICES['tokenization']}/api/tokens/batch-count",
                               json={"texts": texts, "model": "gpt-4"})
        if response.status_code == 200:
            data = response.json()
            total_tokens = data.get('totalTokens', 0)
            print_success(f"Batch processing: {total_tokens} total tokens for {len(texts)} texts")
        else:
            print_error(f"Batch token counting failed: {response.text}")
    except Exception as e:
        print_error(f"Batch token counting request failed: {e}")
    
    # Test supported models
    print_test("Supported Models")
    try:
        response = requests.get(f"{SERVICES['tokenization']}/api/tokens/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get('supportedModels', [])
            print_success(f"Supported models: {', '.join(models)}")
        else:
            print_error(f"Models request failed: {response.text}")
    except Exception as e:
        print_error(f"Models request failed: {e}")

def test_summarization_api():
    print_header("SUMMARIZATION API TESTS")
    
    sample_article = """
    Artificial intelligence (AI) is rapidly transforming industries across the globe, with machine learning algorithms becoming increasingly sophisticated and capable of handling complex tasks that were once thought to be exclusively human domains. Recent breakthroughs in natural language processing, computer vision, and deep learning have opened up new possibilities for automation and intelligent decision-making.

    Companies are investing heavily in AI research and development, with tech giants like Google, Microsoft, and OpenAI leading the charge in creating more powerful and versatile AI systems. These advancements are not only changing how businesses operate but also raising important questions about the future of work, privacy, and the ethical implications of artificial intelligence.

    The integration of AI into everyday applications, from virtual assistants to recommendation systems, has made these technologies more accessible to consumers. As AI continues to evolve, experts predict that we will see even more innovative applications that could revolutionize healthcare, education, transportation, and many other sectors.
    """
    
    # Test article summarization
    print_test("Article Summarization")
    try:
        response = requests.post(f"{SERVICES['summarization']}/api/summarize/article",
                               json={
                                   "content": sample_article,
                                   "length": "medium",
                                   "model": "gpt-3.5-turbo"
                               })
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', '')
            compression_ratio = data.get('compressionRatio', 0)
            token_usage = data.get('tokenUsage', {})
            print_success(f"Article summarized successfully")
            print(f"   Summary: {summary[:100]}...")
            print(f"   Compression ratio: {compression_ratio}")
            print(f"   Token usage: {token_usage.get('totalTokens', 0)} tokens")
        else:
            print_error(f"Article summarization failed: {response.text}")
    except Exception as e:
        print_error(f"Article summarization request failed: {e}")
    
    # Test custom summarization
    print_test("Custom Summarization")
    try:
        response = requests.post(f"{SERVICES['summarization']}/api/summarize/custom",
                               json={
                                   "content": sample_article,
                                   "instructions": "Extract only the key companies mentioned and their AI initiatives",
                                   "model": "gpt-3.5-turbo",
                                   "maxTokens": 150
                               })
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', '')
            print_success(f"Custom summarization completed")
            print(f"   Result: {result[:100]}...")
        else:
            print_error(f"Custom summarization failed: {response.text}")
    except Exception as e:
        print_error(f"Custom summarization request failed: {e}")

def test_integration_workflow():
    print_header("INTEGRATION WORKFLOW TEST")
    
    print_test("Complete News Processing Workflow")
    
    # Step 1: Get news article
    try:
        response = requests.get(f"{SERVICES['feed']}/api/news/top-headlines?category=technology&pageSize=1")
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            if articles:
                article = articles[0]
                content = article.get('content') or article.get('description', '')
                if content and len(content) > 100:
                    print_success("✓ Step 1: Retrieved news article")
                    
                    # Step 2: Count tokens
                    token_response = requests.post(f"{SERVICES['tokenization']}/api/tokens/count",
                                                 json={"text": content, "model": "gpt-3.5-turbo"})
                    if token_response.status_code == 200:
                        token_count = token_response.json().get('tokenCount', 0)
                        print_success(f"✓ Step 2: Counted tokens ({token_count})")
                        
                        # Step 3: Summarize article
                        summary_response = requests.post(f"{SERVICES['summarization']}/api/summarize/article",
                                                       json={
                                                           "content": content,
                                                           "length": "short",
                                                           "model": "gpt-3.5-turbo"
                                                       })
                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            print_success("✓ Step 3: Generated summary")
                            print_success("🎉 Complete workflow successful!")
                            
                            print(f"\n📰 Original Article Title: {article.get('title', 'N/A')}")
                            print(f"📝 Summary: {summary_data.get('summary', 'N/A')}")
                            print(f"📊 Token Usage: {summary_data.get('tokenUsage', {}).get('totalTokens', 0)}")
                        else:
                            print_error("Step 3 failed: Summarization")
                    else:
                        print_error("Step 2 failed: Token counting")
                else:
                    print_error("No suitable article content found")
            else:
                print_error("No articles retrieved")
        else:
            print_error("Failed to retrieve news articles")
    except Exception as e:
        print_error(f"Integration workflow failed: {e}")

def test_scraping_api():
    print_header("SCRAPING API TESTS")
    
    # Test single article scraping
    print_test("Single Article Scraping")
    try:
        response = requests.post(f"{SERVICES['scraping']}/api/scrape/article",
                               json={
                                   "url": "https://techcrunch.com",
                                   "method": "newspaper"
                               }, timeout=30)
        if response.status_code == 200:
            data = response.json()
            article = data.get('article', {})
            print_success(f"Article scraped successfully")
            print(f"   Title: {article.get('title', 'N/A')[:60]}...")
            print(f"   Content length: {len(article.get('content', ''))}")
        else:
            print_error(f"Article scraping failed: {response.text}")
    except Exception as e:
        print_error(f"Article scraping error: {e}")
    
    # Test website discovery
    print_test("Website Link Discovery")
    try:
        response = requests.post(f"{SERVICES['scraping']}/api/scrape/website",
                               json={
                                   "url": "https://techcrunch.com",
                                   "maxLinks": 5
                               }, timeout=30)
        if response.status_code == 200:
            data = response.json()
            links = data.get('links', [])
            print_success(f"Found {len(links)} article links")
        else:
            print_error(f"Website discovery failed: {response.text}")
    except Exception as e:
        print_error(f"Website discovery error: {e}")

def main():
    print("🚀 Starting News Aggregator Microservices Test Suite")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_service_health()
    token = test_authentication_api()
    test_feed_api()
    test_tokenization_api()
    test_summarization_api()
    test_scraping_api()
    test_integration_workflow()
    
    print_header("TEST SUMMARY")
    print("🎯 All tests completed!")
    print("📋 Check the results above for any failures")
    print("🔧 Make sure all services are running and API keys are configured")
    print("\n💡 To start services:")
    print("   docker-compose up -d")
    print("   OR run each service manually with: python app.py")

if __name__ == "__main__":
    main()