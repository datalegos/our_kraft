#!/usr/bin/env python3
"""
Database Discovery and Data Extraction Tool
Scans for databases on specified ports and extracts sample data from tables.
"""

import yaml
import json
import csv
import logging
import socket
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
from dataclasses import dataclass

# Database drivers
try:
    import psycopg2
    import psycopg2.extras
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

@dataclass
class DatabaseInfo:
    """Database connection information"""
    host: str
    port: int
    db_type: str
    database: str
    username: str
    password: str

@dataclass
class TableInfo:
    """Table information"""
    database: str
    table_name: str
    row_count: int
    sample_size: int

class DatabaseExtractor:
    """Main database extraction class"""
    
    def __init__(self, config_file: str = "config/extraction_config.yml"):
        """Initialize the extractor with configuration"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_output_directory()
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_file}' not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        log_level = getattr(logging, log_config['level'].upper())
        
        # Create logs directory
        log_file = log_config['file'].format(timestamp=self.timestamp)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        handlers = [logging.FileHandler(log_file)]
        if log_config['console']:
            handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Database Extractor initialized")
    
    def setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        output_dir = self.config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {output_dir}")
    
    def check_port(self, host: str, port: int) -> bool:
        """Check if a port is open on the given host"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config['security']['connection_timeout'])
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception as e:
            self.logger.debug(f"Port check failed for {host}:{port} - {e}")
            return False
    
    def discover_databases(self) -> List[DatabaseInfo]:
        """Discover available databases on configured hosts and ports"""
        discovered = []
        seen_connections = set()  # Track unique connections to avoid duplicates
        
        for host in self.config['database']['hosts']:
            for db_type in self.config['database']['types']:
                ports = self.config['database']['port_scan'][db_type]
                
                for port in ports:
                    if self.check_port(host, port):
                        self.logger.info(f"Found open port: {host}:{port} ({db_type})")
                        
                        # Try different credentials
                        credentials = self.config['database']['credentials'][db_type]
                        for cred in credentials:
                            db_info = self.test_connection(
                                host, port, db_type, cred['username'], cred['password']
                            )
                            if db_info:
                                # Filter out duplicate connections (same server, different host names)
                                for db in db_info:
                                    connection_key = (port, db_type, db.database, cred['username'])
                                    if connection_key not in seen_connections:
                                        discovered.append(db)
                                        seen_connections.add(connection_key)
                                        self.logger.info(f"Added database: {db.database} on {db.host}:{db.port}")
                                    else:
                                        self.logger.info(f"Skipped duplicate: {db.database} on {db.host}:{db.port}")
                                break
        
        return discovered
    
    def test_connection(self, host: str, port: int, db_type: str, 
                       username: str, password: str) -> Optional[List[DatabaseInfo]]:
        """Test database connection and get list of databases"""
        try:
            if db_type == 'postgresql' and POSTGRESQL_AVAILABLE:
                return self.test_postgresql_connection(host, port, username, password)
            elif db_type == 'mysql' and MYSQL_AVAILABLE:
                return self.test_mysql_connection(host, port, username, password)
            else:
                self.logger.warning(f"Database type {db_type} not supported or driver not available")
                return None
        except Exception as e:
            self.logger.debug(f"Connection test failed for {host}:{port} ({username}) - {e}")
            return None
    
    def test_postgresql_connection(self, host: str, port: int, 
                                 username: str, password: str) -> Optional[List[DatabaseInfo]]:
        """Test PostgreSQL connection and get databases"""
        try:
            conn = psycopg2.connect(
                host=host, port=port, user=username, password=password,
                database='postgres', connect_timeout=self.config['security']['connection_timeout']
            )
            
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT datname FROM pg_database 
                    WHERE datistemplate = false AND datallowconn = true
                """)
                databases = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Filter excluded databases
            excluded = self.config['extraction']['exclude_databases']['postgresql']
            databases = [db for db in databases if not any(db.startswith(ex.rstrip('*')) for ex in excluded)]
            
            self.logger.info(f"PostgreSQL connection successful: {host}:{port} - Found {len(databases)} databases")
            
            return [
                DatabaseInfo(host, port, 'postgresql', db, username, password)
                for db in databases
            ]
            
        except Exception as e:
            self.logger.debug(f"PostgreSQL connection failed: {host}:{port} - {e}")
            return None
    
    def test_mysql_connection(self, host: str, port: int, 
                            username: str, password: str) -> Optional[List[DatabaseInfo]]:
        """Test MySQL connection and get databases"""
        try:
            conn = mysql.connector.connect(
                host=host, port=port, user=username, password=password,
                connection_timeout=self.config['security']['connection_timeout']
            )
            
            cursor = conn.cursor()
            cursor.execute("SHOW DATABASES")
            databases = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Filter excluded databases
            excluded = self.config['extraction']['exclude_databases']['mysql']
            databases = [db for db in databases if not any(db.startswith(ex.rstrip('*')) for ex in excluded)]
            
            self.logger.info(f"MySQL connection successful: {host}:{port} - Found {len(databases)} databases")
            
            return [
                DatabaseInfo(host, port, 'mysql', db, username, password)
                for db in databases
            ]
            
        except Exception as e:
            self.logger.debug(f"MySQL connection failed: {host}:{port} - {e}")
            return None
    
    def get_tables(self, db_info: DatabaseInfo) -> List[TableInfo]:
        """Get list of tables in a database with row counts"""
        try:
            if db_info.db_type == 'postgresql':
                return self.get_postgresql_tables(db_info)
            elif db_info.db_type == 'mysql':
                return self.get_mysql_tables(db_info)
        except Exception as e:
            self.logger.error(f"Failed to get tables from {db_info.database}: {e}")
            return []
    
    def get_postgresql_tables(self, db_info: DatabaseInfo) -> List[TableInfo]:
        """Get PostgreSQL tables and row counts"""
        tables = []
        try:
            conn = psycopg2.connect(
                host=db_info.host, port=db_info.port, 
                user=db_info.username, password=db_info.password,
                database=db_info.database
            )
            
            with conn.cursor() as cursor:
                # First get table names from information_schema
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
                
                table_names = [row[0] for row in cursor.fetchall()]
                
                # Get row count for each table
                for table_name in table_names:
                    try:
                        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                        row_count = cursor.fetchone()[0]
                        
                        sample_size = int(row_count * self.config['extraction']['sample_percentage'])
                        sample_size = min(sample_size, self.config['extraction']['max_records_per_table'])
                        
                        tables.append(TableInfo(db_info.database, table_name, row_count, sample_size))
                        self.logger.info(f"Found table: {table_name} ({row_count} rows)")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not get row count for table {table_name}: {e}")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to get PostgreSQL tables: {e}")
        
        return tables
    
    def get_mysql_tables(self, db_info: DatabaseInfo) -> List[TableInfo]:
        """Get MySQL tables and row counts"""
        tables = []
        try:
            conn = mysql.connector.connect(
                host=db_info.host, port=db_info.port,
                user=db_info.username, password=db_info.password,
                database=db_info.database
            )
            
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT table_name, table_rows 
                FROM information_schema.tables 
                WHERE table_schema = '{db_info.database}' 
                AND table_type = 'BASE TABLE'
            """)
            
            for table, row_count in cursor.fetchall():
                if row_count is None or row_count == 0:
                    # Get accurate count
                    cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                    row_count = cursor.fetchone()[0]
                
                sample_size = int(row_count * self.config['extraction']['sample_percentage'])
                sample_size = min(sample_size, self.config['extraction']['max_records_per_table'])
                
                tables.append(TableInfo(db_info.database, table, row_count, sample_size))
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to get MySQL tables: {e}")
        
        return tables
    
    def extract_table_data(self, db_info: DatabaseInfo, table_info: TableInfo) -> Optional[str]:
        """Extract sample data from a table"""
        try:
            self.logger.info(f"Extracting {table_info.sample_size} records from {table_info.table_name}")
            
            if db_info.db_type == 'postgresql':
                return self.extract_postgresql_data(db_info, table_info)
            elif db_info.db_type == 'mysql':
                return self.extract_mysql_data(db_info, table_info)
                
        except Exception as e:
            self.logger.error(f"Failed to extract data from {table_info.table_name}: {e}")
            return None
    
    def extract_postgresql_data(self, db_info: DatabaseInfo, table_info: TableInfo) -> Optional[str]:
        """Extract data from PostgreSQL table"""
        try:
            conn = psycopg2.connect(
                host=db_info.host, port=db_info.port,
                user=db_info.username, password=db_info.password,
                database=db_info.database
            )
            
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Use TABLESAMPLE for large tables, LIMIT for smaller ones
                if table_info.row_count > 1000:
                    sample_percent = (table_info.sample_size / table_info.row_count) * 100
                    query = f'SELECT * FROM "{table_info.table_name}" TABLESAMPLE SYSTEM ({sample_percent}) LIMIT {table_info.sample_size}'
                else:
                    query = f'SELECT * FROM "{table_info.table_name}" LIMIT {table_info.sample_size}'
                
                cursor.execute(query)
                rows = cursor.fetchall()
            
            conn.close()
            
            # Convert to list of dicts for JSON serialization
            data = [dict(row) for row in rows]
            return self.save_data(db_info, table_info, data)
            
        except Exception as e:
            self.logger.error(f"PostgreSQL extraction failed: {e}")
            return None
    
    def extract_mysql_data(self, db_info: DatabaseInfo, table_info: TableInfo) -> Optional[str]:
        """Extract data from MySQL table"""
        try:
            conn = mysql.connector.connect(
                host=db_info.host, port=db_info.port,
                user=db_info.username, password=db_info.password,
                database=db_info.database
            )
            
            cursor = conn.cursor(dictionary=True)
            
            # MySQL doesn't have TABLESAMPLE, use ORDER BY RAND() for sampling
            if table_info.row_count > 1000:
                query = f"SELECT * FROM `{table_info.table_name}` ORDER BY RAND() LIMIT {table_info.sample_size}"
            else:
                query = f"SELECT * FROM `{table_info.table_name}` LIMIT {table_info.sample_size}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            conn.close()
            
            return self.save_data(db_info, table_info, rows)
            
        except Exception as e:
            self.logger.error(f"MySQL extraction failed: {e}")
            return None
    
    def save_data(self, db_info: DatabaseInfo, table_info: TableInfo, data: List[Dict]) -> str:
        """Save extracted data to file"""
        # Generate filename
        filename = self.config['output']['filename_pattern'].format(
            timestamp=self.timestamp,
            host=db_info.host,
            port=db_info.port,
            database=db_info.database,
            table=table_info.table_name,
            format=self.config['output']['format']
        )
        
        filepath = os.path.join(self.config['output']['directory'], filename)
        
        # Add metadata
        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'database_info': {
                'host': db_info.host,
                'port': db_info.port,
                'type': db_info.db_type,
                'database': db_info.database
            },
            'table_info': {
                'name': table_info.table_name,
                'total_rows': table_info.row_count,
                'extracted_rows': len(data),
                'sample_percentage': self.config['extraction']['sample_percentage']
            },
            'data': data
        }
        
        # Save based on format
        if self.config['output']['format'] == 'json':
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        elif self.config['output']['format'] == 'csv':
            # Save metadata as separate file and data as CSV
            meta_file = filepath.replace('.csv', '_metadata.json')
            with open(meta_file, 'w') as f:
                json.dump({k: v for k, v in metadata.items() if k != 'data'}, f, indent=2, default=str)
            
            if data:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        
        self.logger.info(f"Saved {len(data)} records to {filepath}")
        return filepath
    
    def run(self):
        """Main execution method"""
        self.logger.info("Starting database discovery and extraction")
        
        # Check for required drivers
        if not POSTGRESQL_AVAILABLE:
            self.logger.warning("PostgreSQL driver (psycopg2) not available")
        if not MYSQL_AVAILABLE:
            self.logger.warning("MySQL driver (mysql-connector-python) not available")
        
        # Discover databases
        databases = self.discover_databases()
        if not databases:
            self.logger.error("No databases discovered!")
            return
        
        self.logger.info(f"Discovered {len(databases)} databases")
        
        # Extract data from each database
        total_files = 0
        for db_info in databases:
            self.logger.info(f"Processing database: {db_info.database} on {db_info.host}:{db_info.port}")
            
            tables = self.get_tables(db_info)
            if not tables:
                self.logger.warning(f"No tables found in {db_info.database}")
                continue
            
            self.logger.info(f"Found {len(tables)} tables in {db_info.database}")
            
            # Extract data from each table
            for table_info in tables:
                if table_info.row_count == 0:
                    self.logger.info(f"Skipping empty table: {table_info.table_name}")
                    continue
                
                filepath = self.extract_table_data(db_info, table_info)
                if filepath:
                    total_files += 1
        
        self.logger.info(f"Extraction completed. Generated {total_files} files in {self.config['output']['directory']}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Discovery and Data Extraction Tool")
    parser.add_argument("-c", "--config", default="config/extraction_config.yml",
                       help="Configuration file path (default: config/extraction_config.yml)")
    
    args = parser.parse_args()
    
    extractor = DatabaseExtractor(args.config)
    extractor.run()

if __name__ == "__main__":
    main()