Python Flask Authentication API (Modular Version)
This project is a modular Python Flask-based API that provides endpoints for user authentication and profile management. It uses MySQL as its database and JWT for securing endpoints. The code is organized into a package structure for better maintainability.
Project Structure
/project-folder
|-- auth_api/
|   |-- __init__.py
|   |-- routes.py
|   |-- database.py
|   |-- utils.py
|
|-- config.py
|-- run.py
|-- requirements.txt
|-- README.md


Prerequisites
Python and Pip: Ensure you have Python 3.6+ and pip installed.
MySQL Server: You must have a running MySQL server instance.
Setup Instructions
1. Create Project Files
Create the directory structure and files as shown above.
2. Create a Virtual Environment (Recommended)
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


3. Install Dependencies
Install all the required Python packages from requirements.txt:
pip install -r requirements.txt


4. Configure the Application
Before running, you must configure your database connection. Open config.py and update the DB_USER and DB_PASSWORD variables with your MySQL credentials.
# config.py

class Config:
    # ... other configs
    DB_USER = 'your_mysql_user'
    DB_PASSWORD = 'your_mysql_password'


It is highly recommended to use environment variables for these settings in a production environment.
5. Run the Application
Start the Flask development server using the run.py script:
python run.py


Your API will now be running at http://localhost:5000. The script will automatically attempt to create the database and the users table on the first run.
