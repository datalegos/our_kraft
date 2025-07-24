# run.py
# This is the main entry point to start the Flask application.

from auth_api import create_app

# Create an instance of the Flask application using the factory function
app = create_app()

if __name__ == '__main__':
    # Run the app
    # debug=True will reload the server on code changes and provide detailed error pages.
    # In a production environment, you would use a proper WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, port=5000)
