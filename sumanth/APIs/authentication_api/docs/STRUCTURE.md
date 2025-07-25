# 📁 News Aggregator Authentication API - Directory Structure

## 🎯 **Organized Project Structure**

```
authentication/
├── 🔧 **Core Application**
│   ├── core/
│   │   ├── app.py              # Flask application factory
│   │   ├── config.py           # Configuration management
│   │   ├── models.py           # Database models (User)
│   │   ├── routes.py           # API endpoints
│   │   └── utils.py            # Authentication utilities
│   │
├── 🛡️ **Security Features**
│   ├── security/
│   │   ├── token_blacklist.py  # JWT token blacklisting
│   │   └── security_monitor.py # Activity monitoring
│   │
├── ⚙️ **Setup & Configuration**
│   ├── setup/
│   │   ├── .env.example        # Environment template
│   │   ├── setup_database.py   # Database setup script
│   │   └── setup_database.bat  # Windows setup script
│   │
├── 🧪 **Testing & Documentation**
│   ├── tests/
│   │   ├── test_api.py         # API testing script
│   │   └── postman_collection.json # Postman tests
│   │
│   ├── docs/
│   │   └── API_DOCUMENTATION.md # Complete API reference
│   │
└── 📋 **Root Files**
    ├── .env                    # Your environment config
    ├── requirements.txt        # Python dependencies
    ├── README.md              # Setup instructions
    └── run.py                 # Application entry point
```

## 🚀 **How to Use**

### **Start the API:**
```bash
python run.py
```

### **Run Tests:**
```bash
python tests/test_api.py
```

### **Setup Database:**
```bash
python setup/setup_database.py
```

## 🎯 **Benefits of This Structure**

✅ **Organized** - Related files grouped together
✅ **Scalable** - Easy to add new features
✅ **Professional** - Industry-standard structure
✅ **Maintainable** - Clear separation of concerns
✅ **Clean** - No clutter in root directory

## 📝 **File Purposes**

### **Core Files:**
- `app.py` - Creates and configures Flask application
- `config.py` - Manages environment-based configuration
- `models.py` - Database models and relationships
- `routes.py` - API endpoints and business logic
- `utils.py` - Helper functions and decorators

### **Security Files:**
- `token_blacklist.py` - Prevents JWT token reuse
- `security_monitor.py` - Detects suspicious activity

### **Setup Files:**
- `setup_database.py` - Automated database creation
- `.env.example` - Configuration template

### **Test Files:**
- `test_api.py` - Comprehensive API testing
- `postman_collection.json` - Postman/Insomnia tests

This structure makes your authentication API professional, maintainable, and ready for production! 🎉