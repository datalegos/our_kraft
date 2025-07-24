#!/usr/bin/env python3
"""
Setup verification script for DataLegos RAG Chatbot
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status"""
    if Path(dirpath).is_dir():
        print(f"✅ {description}: {dirpath}/")
        return True
    else:
        print(f"❌ {description}: {dirpath}/ (MISSING)")
        return False

def check_env_variable(var_name):
    """Check if environment variable is set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    value = os.getenv(var_name)
    if value and value != "sk-your-actual-api-key-here":
        print(f"✅ {var_name}: Set")
        return True
    else:
        print(f"❌ {var_name}: Not set or using placeholder")
        return False

def main():
    print("🔍 DataLegos RAG Chatbot - Setup Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check core files
    print("\n📁 Core Files:")
    all_good &= check_file_exists("app.py", "Main application")
    all_good &= check_file_exists("scraper.py", "Website scraper")
    all_good &= check_file_exists("create_embeddings.py", "Embedding creator")
    all_good &= check_file_exists("test_bot.py", "Test script")
    
    # Check configuration files
    print("\n⚙️ Configuration:")
    all_good &= check_file_exists(".env", "Environment variables")
    all_good &= check_file_exists("scraper_config.yaml", "Scraper config")
    all_good &= check_file_exists("environment.yml", "Conda environment")
    
    # Check data files
    print("\n📄 Data Files:")
    all_good &= check_file_exists("scraped_content.txt", "Scraped content")
    all_good &= check_file_exists("processed_chunks.txt", "Processed chunks")
    all_good &= check_directory_exists("vector_index", "Vector database")
    
    # Check documentation
    print("\n📚 Documentation:")
    all_good &= check_file_exists("README.md", "Project documentation")
    
    # Check environment variables
    print("\n🔑 Environment Variables:")
    try:
        all_good &= check_env_variable("OPENAI_API_KEY")
    except ImportError:
        print("❌ python-dotenv not installed")
        all_good = False
    
    # Check Python packages
    print("\n📦 Key Dependencies:")
    required_packages = [
        ("gradio", "gradio"),
        ("langchain", "langchain"), 
        ("langchain_community", "langchain_community"),
        ("langchain_huggingface", "langchain_huggingface"),
        ("faiss", "faiss"),
        ("openai", "openai"),
        ("beautifulsoup4", "bs4"),
        ("requests", "requests"),
        ("pyyaml", "yaml")
    ]
    
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} (MISSING)")
            all_good = False
    
    # Final status
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Setup verification PASSED!")
        print("\nYou can now run:")
        print("  python app.py          # Start web interface")
        print("  python test_bot.py     # Test command line")
    else:
        print("⚠️  Setup verification FAILED!")
        print("\nPlease fix the missing items above.")
        print("Refer to README.md for setup instructions.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())