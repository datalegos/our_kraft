# Project Structure

```
payment-gateway-poc/
│
├── app.py                  # Flask backend with Stripe integration
├── requirements.txt        # Python dependencies
├── .env                   # Your API keys (not in git)
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore rules
├── README.md              # Project overview and quick start
├── PROJECT_STRUCTURE.md   # This file
│
├── verify_setup.py        # Setup verification script
├── test_stripe_keys.py    # Test Stripe API connection
│
├── docs/                  # Documentation
│   └── SETUP_GUIDE.md     # Detailed setup instructions
│
├── templates/             # HTML templates
│   └── index.html         # Main payment page
│
└── static/                # Static assets
    ├── script.js          # Frontend JavaScript (Stripe integration)
    └── style.css          # Styles
```

## File Descriptions

### Core Application Files

- **app.py** - Flask backend server with Stripe API endpoints
- **requirements.txt** - Python package dependencies
- **.env** - Your Stripe API keys (keep private, not in git)
- **.env.example** - Template showing what keys are needed

### Documentation

- **README.md** - Quick start guide and project overview
- **docs/SETUP_GUIDE.md** - Detailed step-by-step setup instructions
- **PROJECT_STRUCTURE.md** - This file explaining the project structure

### Utility Scripts

- **verify_setup.py** - Checks if everything is configured correctly
- **test_stripe_keys.py** - Tests if your Stripe keys work

### Frontend

- **templates/index.html** - Main payment page UI
- **static/script.js** - JavaScript for Stripe payment processing
- **static/style.css** - Styling for the application

### Configuration

- **.gitignore** - Tells git which files to ignore (like .env)
- **.vscode/** - VS Code settings (optional)

## Key Features

- ✅ Clean, organized structure
- ✅ Separation of concerns (backend/frontend/docs)
- ✅ Environment-based configuration
- ✅ Comprehensive documentation
- ✅ Utility scripts for testing
- ✅ Git-ready with proper .gitignore

## Getting Started

1. Read **README.md** for quick start
2. Follow **docs/SETUP_GUIDE.md** for detailed setup
3. Run **verify_setup.py** to check configuration
4. Run **app.py** to start the application
