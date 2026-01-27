from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.crud import DepartmentCRUD, StudentCRUD
from app.schemas import (
    Department, DepartmentCreate, DepartmentUpdate, DepartmentWithStudents,
    Student, StudentCreate, StudentUpdate, StudentWithDepartment
)

router = APIRouter()

# Department endpoints
@router.post("/departments/", response_model=Department, status_code=status.HTTP_201_CREATED)
def create_department(department: DepartmentCreate, db: Session = Depends(get_db)):
    # Check if department already exists
    db_department = DepartmentCRUD.get_department_by_name(db, name=department.name)
    if db_department:
        raise HTTPException(
            status_code=400,
            detail="Department with this name already exists"
        )
    return DepartmentCRUD.create_department(db=db, department=department)

@router.get("/departments/", response_model=List[DepartmentWithStudents])
def read_departments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    departments = DepartmentCRUD.get_departments(db, skip=skip, limit=limit)
    return departments

@router.get("/departments/{department_id}", response_model=DepartmentWithStudents)
def read_department(department_id: int, db: Session = Depends(get_db)):
    db_department = DepartmentCRUD.get_department(db, department_id=department_id)
    if db_department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    return db_department

@router.put("/departments/{department_id}", response_model=Department)
def update_department(department_id: int, department_update: DepartmentUpdate, db: Session = Depends(get_db)):
    db_department = DepartmentCRUD.update_department(db, department_id=department_id, department_update=department_update)
    if db_department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    return db_department

@router.delete("/departments/{department_id}")
def delete_department(department_id: int, db: Session = Depends(get_db)):
    success = DepartmentCRUD.delete_department(db, department_id=department_id)
    if not success:
        raise HTTPException(status_code=404, detail="Department not found")
    return {"message": "Department deleted successfully"}

# Student endpoints
@router.post("/students/", response_model=Student, status_code=status.HTTP_201_CREATED)
def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    # Check if student email already exists
    db_student = StudentCRUD.get_student_by_email(db, email=student.email)
    if db_student:
        raise HTTPException(
            status_code=400,
            detail="Student with this email already exists"
        )
    
    # Check if student ID already exists
    db_student = StudentCRUD.get_student_by_student_id(db, student_id=student.student_id)
    if db_student:
        raise HTTPException(
            status_code=400,
            detail="Student with this student ID already exists"
        )
    
    # Check if department exists (if provided)
    if student.department_id:
        db_department = DepartmentCRUD.get_department(db, department_id=student.department_id)
        if not db_department:
            raise HTTPException(
                status_code=400,
                detail="Department not found"
            )
    
    return StudentCRUD.create_student(db=db, student=student)

@router.get("/students/", response_model=List[Student])
def read_students(skip: int = 0, limit: int = 100, department_id: int = None, db: Session = Depends(get_db)):
    students = StudentCRUD.get_students(db, skip=skip, limit=limit, department_id=department_id)
    return students

@router.get("/students/{student_id}", response_model=StudentWithDepartment)
def read_student(student_id: int, db: Session = Depends(get_db)):
    db_student = StudentCRUD.get_student(db, student_id=student_id)
    if db_student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return db_student

@router.put("/students/{student_id}", response_model=Student)
def update_student(student_id: int, student_update: StudentUpdate, db: Session = Depends(get_db)):
    db_student = StudentCRUD.update_student(db, student_id=student_id, student_update=student_update)
    if db_student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return db_student

@router.delete("/students/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db)):
    success = StudentCRUD.delete_student(db, student_id=student_id)
    if not success:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student deleted successfully"}

# Additional endpoints
@router.get("/departments/{department_id}/students", response_model=List[Student])
def read_students_by_department(department_id: int, db: Session = Depends(get_db)):
    # Check if department exists
    db_department = DepartmentCRUD.get_department(db, department_id=department_id)
    if not db_department:
        raise HTTPException(status_code=404, detail="Department not found")
    
    students = StudentCRUD.get_students(db, department_id=department_id)
    return students