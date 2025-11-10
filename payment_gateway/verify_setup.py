#!/usr/bin/env python3
"""
Setup Verification Script
Checks if your payment gateway POC is configured correctly
"""

import os
import sys
from dotenv import load_dotenv

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_status(check, status, message=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check}")
    if message:
        print(f"   → {message}")

def check_python_version():
    """Check if Python version is 3.7+"""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 7
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_status(
        "Python Version",
        is_valid,
        f"Found Python {version_str}" if is_valid else f"Python {version_str} found, need 3.7+"
    )
    return is_valid

def check_dependencies():
    """Check if required packages are installed"""
    required = ['flask', 'stripe', 'dotenv']
    all_installed = True
    
    for package in required:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print_status(f"Package: {package}", True, "Installed")
        except ImportError:
            print_status(f"Package: {package}", False, "Not installed")
            all_installed = False
    
    return all_installed

def check_env_file():
    """Check if .env file exists"""
    exists = os.path.exists('.env')
    print_status(
        ".env file",
        exists,
        "Found" if exists else "Not found - copy .env.example to .env"
    )
    return exists

def check_stripe_keys():
    """Check if Stripe keys are configured"""
    load_dotenv()
    
    secret_key = os.getenv('STRIPE_SECRET_KEY', '')
    publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY', '')
    
    secret_valid = secret_key.startswith('sk_test_')
    pub_valid = publishable_key.startswith('pk_test_')
    
    print_status(
        "Stripe Secret Key",
        secret_valid,
        "Valid test key" if secret_valid else "Missing or invalid (should start with sk_test_)"
    )
    
    print_status(
        "Stripe Publishable Key",
        pub_valid,
        "Valid test key" if pub_valid else "Missing or invalid (should start with pk_test_)"
    )
    
    return secret_valid and pub_valid

def check_files():
    """Check if required files exist"""
    files = [
        'app.py',
        'requirements.txt',
        'templates/index.html',
        'static/script.js',
        'static/style.css'
    ]
    
    all_exist = True
    for file in files:
        exists = os.path.exists(file)
        print_status(f"File: {file}", exists, "Found" if exists else "Missing")
        if not exists:
            all_exist = False
    
    return all_exist

def main():
    print_header("Payment Gateway POC - Setup Verification")
    
    print("\n📋 Checking Python Environment...")
    python_ok = check_python_version()
    
    print("\n📦 Checking Dependencies...")
    deps_ok = check_dependencies()
    
    print("\n📁 Checking Project Files...")
    files_ok = check_files()
    
    print("\n🔐 Checking Environment Configuration...")
    env_ok = check_env_file()
    
    if env_ok:
        print("\n🔑 Checking Stripe Configuration...")
        stripe_ok = check_stripe_keys()
    else:
        stripe_ok = False
    
    # Summary
    print_header("Summary")
    
    all_ok = python_ok and deps_ok and files_ok and env_ok and stripe_ok
    
    if all_ok:
        print("\n✅ Setup is complete! You can run the application.")
        print("\n🚀 To start the application, run:")
        print("   python app.py")
        print("\n🌐 Then open in browser:")
        print("   http://localhost:5000")
        print("\n💳 Test with card: 4242 4242 4242 4242")
    else:
        print("\n❌ Setup is incomplete. Please fix the issues above.")
        
        if not deps_ok:
            print("\n📦 To install dependencies, run:")
            print("   pip install -r requirements.txt")
        
        if not env_ok:
            print("\n🔐 To create .env file, run:")
            print("   copy .env.example .env")
            print("   Then edit .env and add your API keys")
        
        if env_ok and not stripe_ok:
            print("\n🔑 To get Stripe API keys:")
            print("   1. Sign up at https://stripe.com")
            print("   2. Go to Dashboard → Developers → API Keys")
            print("   3. Toggle 'Test mode' ON")
            print("   4. Copy both keys to .env file")
    
    print("\n📚 For detailed setup instructions, see:")
    print("   docs/SETUP_GUIDE.md")
    print("\n")

if __name__ == "__main__":
    main()
