# 🎓 School Management System

A professional, full-stack web application for managing students and departments with a beautiful UI, REST API, and automated database migrations.

## ✨ Features

- **🎨 Modern Web UI**: Beautiful, responsive interface with tabbed navigation
- **🔄 Complete CRUD Operations**: Create, Read, Update, Delete for students and departments
- **🚀 FastAPI Backend**: High-performance REST API with automatic documentation
- **🗄️ SQLAlchemy Models**: Type-safe database models with relationships
- **📦 Automated Migrations**: Alembic integration for database version control
- **🐳 Docker Database**: PostgreSQL running in Docker container
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **⚡ Real-time Updates**: Dynamic UI updates without page refresh

## 🏗️ Professional Project Structure

```
school_management/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── models/                   # Database models
│   │   ├── __init__.py
│   │   └── models.py            # SQLAlchemy models
│   ├── database/                # Database configuration
│   │   ├── __init__.py
│   │   └── database.py          # Connection & session management
│   ├── schemas/                 # Pydantic schemas
│   │   ├── __init__.py
│   │   └── schemas.py           # API validation schemas
│   ├── crud/                    # Database operations
│   │   ├── __init__.py
│   │   └── crud.py              # CRUD operations
│   ├── api/                     # API routes
│   │   ├── __init__.py
│   │   └── api.py               # FastAPI routes
│   └── ui/                      # User Interface
│       ├── __init__.py
│       ├── static/              # Static files
│       │   ├── css/
│       │   │   └── style.css    # Modern CSS styling
│       │   └── js/
│       │       └── app.js       # Frontend JavaScript
│       └── templates/           # HTML templates
│           └── index.html       # Main UI template
├── database/                    # Database setup
│   ├── docker-compose.yml       # PostgreSQL container
│   ├── alembic.ini             # Alembic configuration
│   └── migrations/             # Auto-generated migrations
│       ├── env.py              # Migration environment
│       └── script.py.mako      # Migration template
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_api.py             # API and UI tests
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
cd school_management

# Automated setup (Linux/Mac)
./docker-start.sh

# Or Windows
docker-start.bat

# Or manual Docker commands
docker-compose up -d --build
```

### Option 2: Manual Development Setup

#### 1. Start PostgreSQL Database
```bash
cd school_management/database
docker-compose up -d
```

#### 2. Install Dependencies
```bash
cd school_management
pip install -r requirements.txt
```

#### 3. Generate Database Migration
```bash
cd database
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

#### 4. Start the Application
```bash
python main.py
```

#### 5. Open Your Browser
Visit: **http://localhost:8000**

#### 6. Test Everything
```bash
python tests/test_api.py
```

## 🎯 What You'll See

### Beautiful Web Interface
- **📚 Departments Tab**: Manage departments with descriptions
- **👨‍🎓 Students Tab**: Manage student records with full details
- **➕ Add/Edit Modals**: Clean forms for data entry
- **🗑️ Delete Confirmations**: Safe deletion with confirmations
- **📱 Responsive Design**: Works perfectly on all devices

### Key UI Features
- Modern gradient design with professional styling
- Tabbed navigation between departments and students
- Modal dialogs for adding/editing records
- Real-time form validation
- Success/error notifications
- Loading states for better UX
- Empty state messages when no data exists

## 🐳 Docker Deployment

For production deployment or if you prefer containerized development:

### Quick Docker Start
```bash
cd school_management

# Linux/Mac
./docker-start.sh

# Windows
docker-start.bat
```

### Manual Docker Commands
```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Production Deployment
```bash
# Use production configuration
cp .env.example .env
# Edit .env with production values
docker-compose -f docker-compose.prod.yml up -d --build
```

**📖 For detailed Docker instructions, see [DOCKER.md](DOCKER.md)**

## 🔧 API Endpoints

### Departments
- `GET /api/departments/` - List all departments
- `POST /api/departments/` - Create new department
- `GET /api/departments/{id}` - Get department with students
- `PUT /api/departments/{id}` - Update department
- `DELETE /api/departments/{id}` - Delete department

### Students
- `GET /api/students/` - List all students
- `POST /api/students/` - Create new student
- `GET /api/students/{id}` - Get student with department
- `PUT /api/students/{id}` - Update student
- `DELETE /api/students/{id}` - Delete student

### Relationships
- `GET /api/departments/{id}/students` - Get students in department

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🗄️ Database Schema

### Departments Table
```sql
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Students Table
```sql
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    student_id VARCHAR(20) NOT NULL UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    enrollment_date DATE DEFAULT CURRENT_DATE,
    department_id INTEGER REFERENCES departments(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔄 Migration Workflow

### The Power of SQLAlchemy + Alembic

1. **Modify Models** (`app/models/models.py`):
```python
class Student(Base):
    # ... existing fields ...
    graduation_year = Column(Integer)  # Add new field
```

2. **Auto-Generate Migration**:
```bash
cd database
alembic revision --autogenerate -m "add graduation year"
```

3. **Alembic Creates This Automatically**:
```python
def upgrade() -> None:
    op.add_column('students', sa.Column('graduation_year', sa.Integer(), nullable=True))

def downgrade() -> None:
    op.drop_column('students', 'graduation_year')
```

4. **Apply Migration**:
```bash
alembic upgrade head
```

### Migration Commands
```bash
# Check current status
alembic current

# View history
alembic history

# Upgrade to latest
alembic upgrade head

# Rollback one version
alembic downgrade -1

# Generate SQL (review before applying)
alembic upgrade head --sql
```

## 🎨 UI Screenshots

### Departments Management
- Clean table view with department information
- Add/Edit modal with form validation
- Student count badges for each department
- Responsive design that works on all devices

### Students Management
- Comprehensive student information display
- Department relationship shown clearly
- Full contact information management
- Easy editing and deletion

### Modern Design Elements
- Gradient backgrounds and modern typography
- Smooth animations and hover effects
- Professional color scheme
- Intuitive navigation and user flow

## 🧪 Testing

Run the comprehensive test suite:
```bash
python tests/test_api.py
```

Tests include:
- ✅ API endpoint functionality
- ✅ CRUD operations
- ✅ Data validation
- ✅ Relationship management
- ✅ UI accessibility
- ✅ Static file serving

## 🔧 Development

### Adding New Features

1. **Add Model Fields** in `app/models/models.py`
2. **Update Schemas** in `app/schemas/schemas.py`
3. **Modify CRUD** in `app/crud/crud.py`
4. **Update API** in `app/api/api.py`
5. **Generate Migration**: `alembic revision --autogenerate`
6. **Apply Migration**: `alembic upgrade head`

### Customizing the UI

- **Styling**: Edit `app/ui/static/css/style.css`
- **Functionality**: Modify `app/ui/static/js/app.js`
- **Layout**: Update `app/ui/templates/index.html`

## 🚀 Production Deployment

### Environment Variables
```bash
export DATABASE_URL="postgresql://user:pass@host:port/dbname"
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t school-management .
docker run -p 8000:8000 school-management
```

## 🎯 Key Benefits

### For Developers
- **Clean Architecture**: Separation of concerns with proper layering
- **Type Safety**: Pydantic schemas and SQLAlchemy models
- **Auto-Documentation**: Swagger UI generated automatically
- **Migration Safety**: Version-controlled database changes

### For Users
- **Intuitive Interface**: Easy to use without training
- **Responsive Design**: Works on any device
- **Real-time Feedback**: Immediate success/error notifications
- **Professional Look**: Modern, clean design

### For Operations
- **Docker Integration**: Easy deployment and scaling
- **Health Checks**: Built-in monitoring endpoints
- **Migration Management**: Safe database updates
- **Comprehensive Logging**: Full request/response tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎓 Built with FastAPI, SQLAlchemy, PostgreSQL, and modern web technologies**