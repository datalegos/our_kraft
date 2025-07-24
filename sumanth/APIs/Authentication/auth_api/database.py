# auth_api/database.py
# This file handles all database-related operations.

import mysql.connector
from mysql.connector import errorcode
from flask import current_app, g

def get_db_connection():
    """
    Establishes a connection to the database or gets it from the application context 'g'.
    'g' is a special object in Flask that is unique for each request.
    """
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(
                host=current_app.config['DB_HOST'],
                user=current_app.config['DB_USER'],
                password=current_app.config['DB_PASSWORD'],
                database=current_app.config['DB_NAME']
            )
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
            return None
    return g.db

def close_db(e=None):
    """Closes the database connection at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def create_database_and_tables():
    """Creates the database and tables if they don't exist."""
    db_name = current_app.config['DB_NAME']
    try:
        # Connect without specifying a database first to create it
        conn = mysql.connector.connect(
            host=current_app.config['DB_HOST'],
            user=current_app.config['DB_USER'],
            password=current_app.config['DB_PASSWORD']
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET UTF8")
        print(f"Database '{db_name}' created or already exists.")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Failed to create database: {err}")
        # exit(1) # Don't exit the app, just log the error

    # Now connect to the specific database to create tables
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            language VARCHAR(10) DEFAULT 'en',
            categories TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_query)
        print("Users table created or already exists.")
        cursor.close()
        # The connection will be closed automatically at the end of the request by close_db

def init_app(app):
    """
    Initializes the database functionality with the Flask app.
    This function is called from the app factory.
    """
    # This ensures that create_database_and_tables is run once before the first request
    with app.app_context():
        create_database_and_tables()
    # This tells Flask to call close_db when cleaning up after a request
    app.teardown_appcontext(close_db)
