# News Aggregator Authentication API

A robust, production-ready authentication service for your news aggregator application. Built with Flask, SQLAlchemy, and JWT authentication.

## Features

🔐 **Secure Authentication**

- JWT token-based authentication
- Password hashing with Werkzeug
- Input validation and sanitization
- Token expiration and verification

👤 **User Management**

- User registration and login
- Profile management
- Account activation/deactivation
- Password strength validation

📰 **News Preferences**

- Customizable news categories
- Preferred news sources
- Language preferences
- Personalized content filtering

🛡️ **Security**

- CORS support for frontend integration
- Environment-based configuration
- SQL injection protection
- Secure password requirements

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

Make sure you have MySQL running and create a database:

```sql
CREATE DATABASE news_auth_db;
```

### 3. Environment Configuration

Copy the example environment file and update it:

```bash
cp .env.example .env
```

Edit `.env` with your database credentials:

```env
SECRET_KEY=your_super_secret_key_here
DB_HOST=localhost
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=news_auth_db
```

### 4. Run the Application

```bash
python run.py
```

The API will be available at `http://localhost:5000`

### 5. Test the API

Run the test script to verify everything works:

```bash
python test_api.py
```

## API Documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed endpoint documentation.

### Quick API Overview

| Endpoint                | Method  | Description              |
| ----------------------- | ------- | ------------------------ |
| `/api/register`         | POST    | Register new user        |
| `/api/login`            | POST    | User login               |
| `/api/logout`           | POST    | User logout              |
| `/api/user/profile`     | GET     | Get user profile         |
| `/api/user/update`      | PUT     | Update user profile      |
| `/api/user/preferences` | GET/PUT | Manage news preferences  |
| `/api/verify-token`     | GET     | Verify JWT token         |
| `/api/categories`       | GET     | Get available categories |
| `/health`               | GET     | Health check             |

## Project Structure

```
authentication/
├── app.py                 # Flask application factory
├── config.py             # Configuration management
├── models.py             # Database models
├── routes.py             # API endpoints
├── utils.py              # Utility functions
├── run.py                # Application entry point
├── requirements.txt      # Python dependencies
├── test_api.py          # API testing script
├── .env.example         # Environment template
├── README.md            # This file
└── API_DOCUMENTATION.md # Detailed API docs
```

## Integration with News Aggregator

This authentication API is designed to integrate seamlessly with your news aggregator:

### 1. User Registration Flow

```python
# Register user with news preferences
POST /api/register
{
    "username": "newsreader",
    "email": "user@example.com",
    "password": "SecurePass123",
    "preferred_categories": "technology,business,sports",
    "preferred_language": "en"
}
```

### 2. Authentication Flow

```python
# Login and get JWT token
POST /api/login
{
    "username": "newsreader",
    "password": "SecurePass123"
}

# Use token for authenticated requests
GET /api/user/preferences
Authorization: Bearer <jwt_token>
```

### 3. Personalized News Filtering

```python
# Get user preferences for news filtering
GET /api/user/preferences
Response: {
    "preferences": {
        "language": "en",
        "categories": ["technology", "business", "sports"],
        "sources": ["bbc", "reuters"]
    }
}
```

## Available News Categories

- `general` - General news
- `business` - Business and finance
- `entertainment` - Entertainment news
- `health` - Health and medical
- `science` - Science and research
- `sports` - Sports news
- `technology` - Technology news
- `politics` - Political news

## Security Best Practices

✅ **Implemented**

- Password hashing with salt
- JWT token expiration
- Input validation and sanitization
- Environment-based secrets
- CORS configuration
- SQL injection protection

🔧 **Recommended for Production**

- Use HTTPS only
- Implement rate limiting
- Add request logging
- Set up monitoring
- Use a reverse proxy (nginx)
- Implement token blacklisting for logout

## Environment Variables

| Variable      | Description       | Default          |
| ------------- | ----------------- | ---------------- |
| `SECRET_KEY`  | Flask secret key  | `dev-secret-key` |
| `DB_HOST`     | Database host     | `localhost`      |
| `DB_USER`     | Database user     | `root`           |
| `DB_PASSWORD` | Database password | ``               |
| `DB_NAME`     | Database name     | `news_auth_db`   |
| `FLASK_ENV`   | Environment       | `development`    |

## Development

### Running Tests

```bash
python test_api.py
```

### Database Migration

The app automatically creates tables on startup. For production, consider using Flask-Migrate for database versioning.

### Adding New Features

1. Update models in `models.py`
2. Add routes in `routes.py`
3. Update utilities in `utils.py`
4. Test with `test_api.py`
5. Update documentation

## Troubleshooting

### Common Issues

**Database Connection Error**

- Check MySQL is running
- Verify database credentials in `.env`
- Ensure database exists

**Import Errors**

- Install all requirements: `pip install -r requirements.txt`
- Check Python version (3.7+ recommended)

**JWT Token Issues**

- Verify SECRET_KEY is set
- Check token expiration (24 hours default)
- Ensure proper Authorization header format

### Debug Mode

Set `FLASK_ENV=development` in your `.env` file for detailed error messages.

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app('production')"
```

### Using Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:create_app('production')"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License.

---

**Ready to integrate with your news aggregator!** 🚀

For questions or support, please check the API documentation or create an issue.
