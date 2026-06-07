from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_
from app.models import Department, Student
from app.schemas import DepartmentCreate, DepartmentUpdate, StudentCreate, StudentUpdate
from typing import List, Optional

# Department CRUD operations
class DepartmentCRUD:
    
    @staticmethod
    def get_department(db: Session, department_id: int) -> Optional[Department]:
        return db.query(Department).options(joinedload(Department.students)).filter(Department.id == department_id).first()
    
    @staticmethod
    def get_department_by_name(db: Session, name: str) -> Optional[Department]:
        return db.query(Department).filter(Department.name == name).first()
    
    @staticmethod
    def get_departments(db: Session, skip: int = 0, limit: int = 100) -> List[Department]:
        return db.query(Department).options(joinedload(Department.students)).offset(skip).limit(limit).all()
    
    @staticmethod
    def create_department(db: Session, department: DepartmentCreate) -> Department:
        db_department = Department(**department.dict())
        db.add(db_department)
        db.commit()
        db.refresh(db_department)
        return db_department
    
    @staticmethod
    def update_department(db: Session, department_id: int, department_update: DepartmentUpdate) -> Optional[Department]:
        db_department = db.query(Department).filter(Department.id == department_id).first()
        if db_department:
            update_data = department_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_department, field, value)
            db.commit()
            db.refresh(db_department)
        return db_department
    
    @staticmethod
    def delete_department(db: Session, department_id: int) -> bool:
        db_department = db.query(Department).filter(Department.id == department_id).first()
        if db_department:
            db.delete(db_department)
            db.commit()
            return True
        return False

# Student CRUD operations
class StudentCRUD:
    
    @staticmethod
    def get_student(db: Session, student_id: int) -> Optional[Student]:
        return db.query(Student).options(joinedload(Student.department)).filter(Student.id == student_id).first()
    
    @staticmethod
    def get_student_by_email(db: Session, email: str) -> Optional[Student]:
        return db.query(Student).filter(Student.email == email).first()
    
    @staticmethod
    def get_student_by_student_id(db: Session, student_id: str) -> Optional[Student]:
        return db.query(Student).filter(Student.student_id == student_id).first()
    
    @staticmethod
    def get_students(db: Session, skip: int = 0, limit: int = 100, department_id: Optional[int] = None) -> List[Student]:
        query = db.query(Student).options(joinedload(Student.department))
        if department_id:
            query = query.filter(Student.department_id == department_id)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def create_student(db: Session, student: StudentCreate) -> Student:
        db_student = Student(**student.dict())
        db.add(db_student)
        db.commit()
        db.refresh(db_student)
        return db_student
    
    @staticmethod
    def update_student(db: Session, student_id: int, student_update: StudentUpdate) -> Optional[Student]:
        db_student = db.query(Student).filter(Student.id == student_id).first()
        if db_student:
            update_data = student_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_student, field, value)
            db.commit()
            db.refresh(db_student)
        return db_student
    
    @staticmethod
    def delete_student(db: Session, student_id: int) -> bool:
        db_student = db.query(Student).filter(Student.id == student_id).first()
        if db_student:
            db.delete(db_student)
            db.commit()
            return True
        return False