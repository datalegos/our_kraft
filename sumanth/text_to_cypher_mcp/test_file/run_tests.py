#!/usr/bin/env python3
"""
Test Runner for Text-to-Cypher Application
Simple interface to run different test suites
"""

import sys
import subprocess
import requests
from colorama import Fore, Style, init

init(autoreset=True)

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import requests
        import colorama
        return True
    except ImportError:
        print(f"{Fore.YELLOW}📦 Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"])
            return True
        except subprocess.CalledProcessError:
            print(f"{Fore.RED}❌ Failed to install dependencies. Please run: pip install -r test_requirements.txt")
            return False

def check_application():
    """Check if the application is running"""
    try:
        response = requests.get("http://localhost:8081/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            print(f"{Fore.RED}❌ Application not responding properly (status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}❌ Application not running: {e}")
        print(f"{Fore.YELLOW}💡 Please start the application with: docker-compose up -d")
        return False

def show_menu():
    """Show test menu options"""
    print(f"\n{Fore.CYAN}🧪 Text-to-Cypher Test Suite")
    print(f"{'=' * 40}")
    print(f"{Fore.WHITE}Choose a test option:")
    print(f"{Fore.GREEN}  1. 🎯 Flow Demo - Quick demonstration of key features")
    print(f"{Fore.GREEN}  2. 🔍 Comprehensive Tests - Full test suite with all endpoints")
    print(f"{Fore.GREEN}  3. 🚀 Both - Run flow demo then comprehensive tests")
    print(f"{Fore.YELLOW}  4. ❌ Exit")
    print(f"{Fore.CYAN}{'=' * 40}")

def run_flow_demo():
    """Run the flow demonstration"""
    print(f"\n{Fore.CYAN}🎯 Running Flow Demonstration...")
    try:
        subprocess.run([sys.executable, "flow_demdso.py"], check=True)
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}❌ Flow demo failed")
    except FileNotFoundError:
        print(f"{Fore.RED}❌ flow_demo.py not found")

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print(f"\n{Fore.CYAN}🔍 Running Comprehensive Test Suite...")
    try:
        subprocess.run([sys.executable, "comprehensive_test.py"], check=True)
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}❌ Comprehensive tests failed")
    except FileNotFoundError:
        print(f"{Fore.RED}❌ comprehensive_test.py not found")

def main():
    """Main test runner"""
    print(f"{Fore.CYAN}🚀 Text-to-Cypher Application Test Runner")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if application is running
    if not check_application():
        return
    
    while True:
        show_menu()
        
        try:
            choice = input(f"\n{Fore.YELLOW}Enter your choice (1-4): {Style.RESET_ALL}").strip()
            
            if choice == "1":
                run_flow_demo()
            elif choice == "2":
                run_comprehensive_tests()
            elif choice == "3":
                run_flow_demo()
                print(f"\n{Fore.CYAN}{'=' * 50}")
                print(f"Flow demo complete. Starting comprehensive tests...")
                print(f"{'=' * 50}")
                run_comprehensive_tests()
            elif choice == "4":
                print(f"\n{Fore.GREEN}👋 Goodbye!")
                break
            else:
                print(f"{Fore.RED}❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}🛑 Test runner interrupted by user")
            break
        except EOFError:
            print(f"\n\n{Fore.YELLOW}🛑 Test runner terminated")
            break
        
        # Ask if user wants to continue
        try:
            continue_choice = input(f"\n{Fore.YELLOW}Run another test? (y/n): {Style.RESET_ALL}").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print(f"\n{Fore.GREEN}👋 Thanks for testing!")
                break
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Fore.GREEN}👋 Thanks for testing!")
            break

if __name__ == "__main__":
    main()