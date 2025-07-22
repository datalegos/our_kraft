# News Aggregator Authentication API

A robust authentication service for your news aggregator application with JWT-based authentication and user preference management.

## Features

- User registration and login
- JWT token-based authentication
- Password validation and hashing
- User profile management
- News preferences (categories, sources, language)
- Token verification
- Input validation and error handling

## Base URL

```
http://localhost:5000
```

## Authentication

Most endpoints require a JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### 1. User Registration

**POST** `/api/register`

Register a new user account.

**Request Body:**

```json
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123",
  "preferred_language": "en",
  "preferred_categories": "technology,business,sports",
  "preferred_sources": "bbc,cnn"
}
```

**Response (201):**

```json
{
  "message": "User registered successfully",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "username": "johndoe",
    "email": "john@example.com",
    "preferred_language": "en",
    "preferred_categories": ["technology", "business", "sports"],
    "preferred_sources": ["bbc", "cnn"],
    "is_active": true,
    "created_at": "2024-01-15T10:30:00",
    "last_login": null
  }
}
```

### 2. User Login

**POST** `/api/login`

Authenticate user and get JWT token.

**Request Body:**

```json
{
  "username": "johndoe",
  "password": "SecurePass123"
}
```

**Response (200):**

```json
{
  "message": "Login successful",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "username": "johndoe",
    "email": "john@example.com",
    "preferred_language": "en",
    "preferred_categories": ["technology", "business", "sports"],
    "preferred_sources": ["bbc", "cnn"],
    "is_active": true,
    "created_at": "2024-01-15T10:30:00",
    "last_login": "2024-01-15T11:45:00"
  }
}
```

### 3. User Logout

**POST** `/api/logout`

Logout user (client should remove token).

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "message": "Logged out successfully"
}
```

### 4. Get User Profile

**GET** `/api/user/profile`

Get current user's profile information.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "user": {
    "id": 1,
    "username": "johndoe",
    "email": "john@example.com",
    "preferred_language": "en",
    "preferred_categories": ["technology", "business", "sports"],
    "preferred_sources": ["bbc", "cnn"],
    "is_active": true,
    "created_at": "2024-01-15T10:30:00",
    "last_login": "2024-01-15T11:45:00"
  }
}
```

### 5. Update User Profile

**PUT** `/api/user/update`

Update user profile information.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "email": "newemail@example.com",
  "password": "NewSecurePass123",
  "preferred_language": "es",
  "preferred_categories": ["sports", "entertainment"],
  "preferred_sources": ["reuters", "ap"]
}
```

**Response (200):**

```json
{
  "message": "Profile updated successfully",
  "user": {
    "id": 1,
    "username": "johndoe",
    "email": "newemail@example.com",
    "preferred_language": "es",
    "preferred_categories": ["sports", "entertainment"],
    "preferred_sources": ["reuters", "ap"],
    "is_active": true,
    "created_at": "2024-01-15T10:30:00",
    "last_login": "2024-01-15T11:45:00"
  }
}
```

### 6. Get User Preferences

**GET** `/api/user/preferences`

Get user's news preferences.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "preferences": {
    "language": "en",
    "categories": ["technology", "business", "sports"],
    "sources": ["bbc", "cnn"]
  }
}
```

### 7. Update User Preferences

**PUT** `/api/user/preferences`

Update user's news preferences.

**Headers:** `Authorization: Bearer <token>`

**Request Body:**

```json
{
  "language": "fr",
  "categories": ["technology", "science"],
  "sources": ["lemonde", "figaro"]
}
```

**Response (200):**

```json
{
  "message": "Preferences updated successfully",
  "preferences": {
    "language": "fr",
    "categories": ["technology", "science"],
    "sources": ["lemonde", "figaro"]
  }
}
```

### 8. Get Available Categories

**GET** `/api/categories`

Get list of available news categories.

**Response (200):**

```json
{
  "categories": [
    "general",
    "business",
    "entertainment",
    "health",
    "science",
    "sports",
    "technology",
    "politics"
  ]
}
```

### 9. Verify Token

**GET** `/api/verify-token`

Verify if JWT token is valid.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**

```json
{
  "valid": true,
  "user_id": 1,
  "username": "johndoe"
}
```

### 10. Health Check

**GET** `/health`

Check if the API is running.

**Response (200):**

```json
{
  "status": "healthy",
  "service": "authentication-api"
}
```

### 11. API Information

**GET** `/api/info`

Get API information and available endpoints.

**Response (200):**

```json
{
  "service": "News Aggregator Authentication API",
  "version": "1.0.0",
  "endpoints": {
    "auth": {
      "register": "POST /api/register",
      "login": "POST /api/login",
      "logout": "POST /api/logout",
      "profile": "GET /api/user/profile",
      "update_profile": "PUT /api/user/update",
      "get_preferences": "GET /api/user/preferences",
      "update_preferences": "PUT /api/user/preferences",
      "verify_token": "GET /api/verify-token"
    },
    "utility": {
      "categories": "GET /api/categories",
      "health": "GET /health",
      "info": "GET /api/info"
    }
  }
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error message description",
  "details": "Additional error details (optional)"
}
```

### Common HTTP Status Codes:

- `200` - Success
- `201` - Created (registration)
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (invalid credentials/token)
- `404` - Not Found
- `409` - Conflict (username/email already exists)
- `500` - Internal Server Error

## Password Requirements

- Minimum 8 characters
- At least one letter
- At least one number

## Valid News Categories

- general
- business
- entertainment
- health
- science
- sports
- technology
- politics

## Usage Examples

### Register and Login Flow

```bash
# Register
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newsreader",
    "email": "reader@example.com",
    "password": "SecurePass123",
    "preferred_categories": "technology,business"
  }'

# Login
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newsreader",
    "password": "SecurePass123"
  }'

# Use token for authenticated requests
curl -X GET http://localhost:5000/api/user/profile \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Integration with News Aggregator

This authentication API is designed to work seamlessly with your news aggregator:

1. **User Registration**: Collect user preferences during signup
2. **Authentication**: Use JWT tokens for secure API access
3. **Personalization**: Use user preferences to filter news content
4. **Profile Management**: Allow users to update their news preferences

The user preferences (categories, sources, language) can be used by your news aggregator to:

- Filter news by preferred categories
- Prioritize content from preferred sources
- Serve news in the user's preferred language
- Provide personalized news recommendations
