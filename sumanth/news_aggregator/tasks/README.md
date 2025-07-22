# Modular Prompt Generator System

This directory contains a modular prompt generator system that creates personalized news prompts based on user types.

## Architecture

The system is organized into three user type modules and one orchestrator:

### User Type Modules

1. **`user_types/anonymous_user.py`** - Handles anonymous users
   - Generates general global news prompts
   - No user ID required
   - Returns 5 global news items

2. **`user_types/new_user.py`** - Handles newly registered users
   - Uses registration details (country, state, district)
   - Generates news based on location preferences
   - Default category: "general"

3. **`user_types/existing_user.py`** - Handles existing registered users
   - Analyzes bookmarks and likes history
   - Determines preferred category from user behavior
   - Provides personalized news feed

### Orchestrator

**`prompt_orchestrator.py`** - Main orchestrator that:
- Determines user type automatically
- Delegates to appropriate user type module
- Provides unified interface for all user types

## User Type Determination Logic

1. **Anonymous User**: `user_id = None`
2. **New User**: Registered user with no bookmarks and no likes
3. **Existing User**: Registered user with bookmarks or likes history

## Usage

### Basic Usage

```python
from prompt_orchestrator import generate_user_prompt_json

# Anonymous user
prompt = generate_user_prompt_json(user_id=None)

# Registered user (type determined automatically)
prompt = generate_user_prompt_json(user_id=123)
```

### Advanced Usage

```python
from prompt_orchestrator import PromptOrchestrator

orchestrator = PromptOrchestrator()

# Determine user type
user_type = orchestrator.determine_user_type(user_id=123)

# Generate prompt
prompt = orchestrator.generate_user_prompt(user_id=123)
```

### Direct Module Usage

```python
from user_types import AnonymousUserPromptGenerator, NewUserPromptGenerator, ExistingUserPromptGenerator

# Anonymous user
anon_generator = AnonymousUserPromptGenerator()
prompt = anon_generator.generate_prompt()

# New user
new_generator = NewUserPromptGenerator()
prompt = new_generator.generate_prompt(user_id=123)

# Existing user
existing_generator = ExistingUserPromptGenerator()
prompt = existing_generator.generate_prompt(user_id=123)
```

## Database Requirements

The system requires a PostgreSQL database with the following tables:

### Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    country VARCHAR(100),
    state VARCHAR(100),
    district VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Bookmarks Table
```sql
CREATE TABLE bookmarks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    news_id INTEGER REFERENCES news_articles(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Likes Table
```sql
CREATE TABLE likes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    news_id INTEGER REFERENCES news_articles(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### News Articles Table
```sql
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50),
    title TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

Database connection parameters are hardcoded in each module. To make them configurable, you can:

1. Use environment variables
2. Create a config file
3. Pass connection parameters to constructors

## Testing

Run the test file to verify the system:

```bash
python test_modular_prompt_generator.py
```

## Output Format

All modules return a consistent JSON structure:

```json
{
  "intent": "fetch_news",
  "user_type": "anonymous|new_user|existing_user",
  "user_id": 123,
  "timestamp": "2024-01-01T12:00:00",
  "user_profile": {
    "country": "India",
    "state": "Karnataka",
    "district": "Bangalore",
    "preferred_category": "technology"
  },
  "segments": [
    {
      "region_level": "global",
      "region": null,
      "category": "technology",
      "count": 3,
      "prompt": "Give me the top 3 global technology news headlines today."
    }
  ]
}
```

## Migration from Original System

The original `promptgenerator.py` functionality is preserved through the `generate_user_prompt_json()` function in `prompt_orchestrator.py`, ensuring backward compatibility. 