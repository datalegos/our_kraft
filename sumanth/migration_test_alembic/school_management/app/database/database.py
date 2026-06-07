from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models import Base
import os
import time

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/school_db")

# Create engine with connection retry
def create_db_engine():
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            engine = create_engine(DATABASE_URL)
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✅ Database connection successful on attempt {attempt + 1}")
            return engine
        except Exception as e:
            print(f"❌ Database connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("💥 All database connection attempts failed")
                raise

# Create engine
engine = create_db_engine()

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables (for development - in production use Alembic)
def create_tables():
    print("🔨 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")