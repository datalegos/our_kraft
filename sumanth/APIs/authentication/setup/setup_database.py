#!/usr/bin/env python3
"""
Database setup script for News Aggregator Authentication API
This script will create the database and tables automatically
"""

import mysql.connector
from mysql.connector import errorcode
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create the database if it doesn't exist"""
    
    # Database configuration
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'news_auth_db')
    
    print("🗄️  Setting up database for News Aggregator Authentication API")
    print(f"📍 Host: {DB_HOST}")
    print(f"👤 User: {DB_USER}")
    print(f"🏷️  Database: {DB_NAME}")
    
    try:
        # Connect to MySQL server (without specifying database)
        print("\n1️⃣  Connecting to MySQL server...")
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()
        print("✅ Connected to MySQL server successfully")
        
        # Create database
        print(f"\n2️⃣  Creating database '{DB_NAME}'...")
        try:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Database '{DB_NAME}' created successfully")
        except mysql.connector.Error as err:
            print(f"⚠️  Database creation warning: {err}")
        
        # Use the database
        cursor.execute(f"USE {DB_NAME}")
        
        # Create users table
        print("\n3️⃣  Creating users table...")
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            preferred_language VARCHAR(10) DEFAULT 'en',
            preferred_categories TEXT DEFAULT 'general,technology,business',
            preferred_sources TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL,
            INDEX idx_username (username),
            INDEX idx_email (email),
            INDEX idx_active (is_active)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(create_users_table)
        print("✅ Users table created successfully")
        
        # Show table structure
        print("\n4️⃣  Verifying table structure...")
        cursor.execute("DESCRIBE users")
        columns = cursor.fetchall()
        
        print("📋 Users table structure:")
        for column in columns:
            print(f"   - {column[0]}: {column[1]} {'(Primary Key)' if column[3] == 'PRI' else ''}")
        
        # Check if database is ready
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"\n📊 Current users in database: {user_count}")
        
        cursor.close()
        connection.close()
        
        print("\n🎉 Database setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Update your .env file with the database credentials")
        print("   2. Run: python run.py")
        print("   3. Test the API: python test_api.py")
        
        return True
        
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("❌ Access denied. Check your username and password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("❌ Database does not exist")
        else:
            print(f"❌ Database error: {err}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_connection():
    """Test database connection with the created database"""
    
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'news_auth_db')
    
    try:
        print("\n🔍 Testing database connection...")
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"✅ Connected to MySQL version: {version[0]}")
        
        cursor.execute("SELECT DATABASE()")
        current_db = cursor.fetchone()
        print(f"✅ Using database: {current_db[0]}")
        
        cursor.close()
        connection.close()
        
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Connection test failed: {err}")
        return False

if __name__ == "__main__":
    print("🚀 News Aggregator Authentication API - Database Setup")
    print("=" * 60)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Please create one from .env.example")
        print("   cp .env.example .env")
        print("   Then edit .env with your database credentials")
        exit(1)
    
    # Create database
    if create_database():
        # Test connection
        if test_connection():
            print("\n🎯 Database is ready for the authentication API!")
        else:
            print("\n⚠️  Database created but connection test failed")
    else:
        print("\n❌ Database setup failed")
        print("\n🔧 Troubleshooting:")
        print("   - Make sure MySQL is running")
        print("   - Check your database credentials in .env")
        print("   - Verify MySQL user has CREATE DATABASE privileges")