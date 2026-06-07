from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, date

Base = declarative_base()

class Department(Base):
    __tablename__ = 'departments'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(500), nullable=False, unique=True, index=True)
    description = Column(Text)
    hod = Column(String(100), nullable=True)  # Head of Department
    hod_mobile= Column(String(100),nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to students
    students = relationship("Student", back_populates="department", cascade="all, delete-orphan")

class Student(Base):
    __tablename__ = 'students'
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    student_id = Column(String(20), unique=True, nullable=False, index=True)
    phone = Column(String(20))
    address = Column(Text)
    enrollment_date = Column(Date, default=date.today)
    department_id = Column(Integer, ForeignKey('departments.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to department
    department = relationship("Department", back_populates="students")