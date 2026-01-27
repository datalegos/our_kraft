from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, date

# Department Schemas
class DepartmentBase(BaseModel):
    name: str
    description: Optional[str] = None
    hod: Optional[str] = None  # Head of Department

class DepartmentCreate(DepartmentBase):
    pass

class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    hod: Optional[str] = None

class Department(DepartmentBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class DepartmentWithStudents(Department):
    students: List['Student'] = []

# Student Schemas
class StudentBase(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    student_id: str
    phone: Optional[str] = None
    address: Optional[str] = None
    department_id: Optional[int] = None

class StudentCreate(StudentBase):
    pass

class StudentUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    student_id: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    department_id: Optional[int] = None

class Student(StudentBase):
    id: int
    enrollment_date: date
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class StudentWithDepartment(Student):
    department: Optional[Department] = None

# Update forward references
DepartmentWithStudents.model_rebuild()