import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# Placeholder: Replace with your actual PostgreSQL connection string
# Example: postgresql+psycopg2://username:password@localhost:5432/yourdatabase
# DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/news_aggregator"
# postgresql://[user[:password]@][host][:port][/dbname][?param1=value1&param2=value2]

DATABASE_URL = "postgresql://postgres:password@localhost:5432/globalpulse"
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

# Set up the database engine and session
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

# Create tables (run once at startup)
def init_db():
    Base.metadata.create_all(engine)

# Registration logic
def register_user(username: str, email: str, password_hash: str):
    session = SessionLocal()
    try:
        user = User(username=username, email=email, password_hash=password_hash)
        session.add(user)
        session.commit()
        print(f"User '{username}' registered successfully.")
        return user
    except IntegrityError:
        session.rollback()
        print("Username or email already exists.")
        return None
    finally:
        session.close()

# Example usage/test function
def test_registration():
    init_db()
    # In a real app, hash the password before storing!
    user1 = register_user("alice", "alice@example.com", "hashedpassword123")
    user2 = register_user("bob", "bob@example.com", "hashedpassword456")
    # Try duplicate
    user3 = register_user("alice", "alice@example.com", "anotherhash")

if __name__ == "__main__":
    test_registration() 