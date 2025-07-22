@echo off
echo Setting up News Aggregator Authentication Database...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found
    echo Please copy .env.example to .env and configure your database settings
    echo.
    if exist .env.example (
        echo Copying .env.example to .env...
        copy .env.example .env
        echo Please edit .env file with your database credentials
        notepad .env
    )
    pause
    exit /b 1
)

echo Running database setup script...
python setup_database.py

echo.
echo Database setup completed!
echo You can now run the API with: python run.py
pause