# DataLegos Dependencies

Complete guide for managing project dependencies with Poetry.

## 📦 Dependency Management

This project uses **Poetry** for dependency management.

### Files

- **pyproject.toml** - Project configuration and dependencies
- **poetry.lock** - Locked dependency versions (auto-generated)

## 🐳 Docker Deployment (Recommended)

Dependencies are automatically installed in the Docker container using Poetry.

```bash
cd deployment_scripts
./start_all.sh
# All dependencies installed automatically with Poetry
```

## 💻 Local Development Setup

### Prerequisites

- Python 3.12 or higher
- Poetry (will be installed by setup script)

### Quick Setup

**Linux/Mac:**
```bash
chmod +x setup_poetry.sh
./setup_poetry.sh
```

**Windows:**
```bash
setup_poetry.bat
```

### Manual Setup

1. **Install Poetry:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Configure Poetry:**
   ```bash
   poetry config virtualenvs.in-project true
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```

4. **Download spaCy model:**
   ```bash
   poetry run python -m spacy download en_core_web_lg
   ```

5. **Activate virtual environment:**
   ```bash
   poetry shell
   ```

## 📋 Core Dependencies

### Production Dependencies (requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| requests | >=2.31.0 | HTTP requests to Wazuh API |
| pyyaml | >=6.0.1 | YAML configuration files |
| neo4j | >=5.15.0 | Neo4j database driver |
| pandas | >=2.1.0 | Data processing |
| python-dotenv | >=1.0.0 | Environment variable management |
| python-dateutil | >=2.8.2 | Date/time handling |

### Presidio Dependencies (requirements_presidio.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| presidio-analyzer | 2.2.354 | PII detection engine |
| presidio-anonymizer | 2.2.354 | PII anonymization |
| spacy | >=3.7.0,<4.0.0 | NLP engine for Presidio |
| en-core-web-lg | 3.7.1 | English language model |

### Development Dependencies (requirements-dev.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.4.0 | Testing framework |
| pytest-cov | >=4.1.0 | Test coverage |
| black | >=23.12.0 | Code formatting |
| flake8 | >=7.0.0 | Linting |
| pylint | >=3.0.0 | Code analysis |
| mypy | >=1.8.0 | Type checking |
| ipython | >=8.18.0 | Interactive shell |

## 🔄 Updating Dependencies

### Update All Dependencies

```bash
# Update all packages to latest compatible versions
poetry update

# Update poetry.lock without installing
poetry lock --no-update
```

### Add New Dependency

```bash
# Add to main dependencies
poetry add <package-name>

# Add to dev dependencies
poetry add --group dev <package-name>

# Add specific version
poetry add <package-name>@^2.0.0
```

### Remove Dependency

```bash
poetry remove <package-name>
```

### Update Specific Package

```bash
poetry update <package-name>
```

## 📋 Poetry Commands

### Common Commands

```bash
# Show installed packages
poetry show

# Show dependency tree
poetry show --tree

# Show outdated packages
poetry show --outdated

# Activate virtual environment
poetry shell

# Run command in virtual environment
poetry run python scripts/main.py

# Install dependencies
poetry install

# Install only main dependencies (no dev)
poetry install --only main

# Update lock file
poetry lock

# Export to requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
```

## 🐍 Python Version

**Required:** Python 3.12 or higher

**Check your version:**
```bash
python --version
# or
python3 --version
```

**Install Python 3.12:**

- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install python3.12 python3.12-venv python3.12-dev
  ```

- **macOS (Homebrew):**
  ```bash
  brew install python@3.12
  ```

- **Windows:**
  Download from [python.org](https://www.python.org/downloads/)

## 🔍 Dependency Details

### Why These Dependencies?

**requests**
- Used for HTTP calls to Wazuh Manager API
- Reliable, well-maintained library
- Handles authentication and SSL

**pyyaml**
- Parses YAML configuration files
- All configs are in YAML format
- Easy to read and maintain

**neo4j**
- Official Neo4j Python driver
- Connects to Neo4j database
- Executes Cypher queries

**pandas**
- Data manipulation and analysis
- Used for aggregations
- Efficient data processing

**python-dotenv**
- Loads environment variables from .env
- Keeps secrets out of code
- Standard practice for configuration

**presidio-analyzer**
- Microsoft's PII detection engine
- Detects sensitive information
- Configurable entity recognition

**spacy**
- NLP engine required by Presidio
- Provides language understanding
- en_core_web_lg model for English

## 🚨 Common Issues

### Issue: spaCy model not found

**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Issue: pip install fails

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Try again
pip install -r requirements.txt
```

### Issue: Permission denied

**Solution:**
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use --user flag (not recommended)
pip install --user -r requirements.txt
```

### Issue: Conflicting dependencies

**Solution:**
```bash
# Create fresh virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Dependency Tree

```
DataLegos Pipeline
├── Core Dependencies
│   ├── requests (HTTP)
│   ├── pyyaml (Config)
│   ├── neo4j (Database)
│   ├── pandas (Data)
│   └── python-dotenv (Env)
│
├── PII Detection
│   ├── presidio-analyzer
│   ├── presidio-anonymizer
│   └── spacy + en_core_web_lg
│
└── Development (Optional)
    ├── Testing (pytest, pytest-cov)
    ├── Code Quality (black, flake8, pylint)
    └── Tools (ipython, mypy)
```

## 🔐 Security

### Keeping Dependencies Updated

```bash
# Check for security vulnerabilities
pip install safety
safety check

# Update vulnerable packages
pip install --upgrade <vulnerable-package>
```

### Best Practices

1. **Use virtual environments** - Isolate project dependencies
2. **Pin versions in production** - Use exact versions for stability
3. **Regular updates** - Keep dependencies up to date
4. **Security scanning** - Check for vulnerabilities regularly
5. **Minimal dependencies** - Only install what you need

## 📖 Additional Resources

- [pip documentation](https://pip.pypa.io/)
- [Virtual environments guide](https://docs.python.org/3/tutorial/venv.html)
- [Presidio documentation](https://microsoft.github.io/presidio/)
- [spaCy documentation](https://spacy.io/)
- [Neo4j Python driver](https://neo4j.com/docs/python-manual/current/)

## 🆘 Getting Help

### Check installed packages
```bash
pip list
```

### Check package version
```bash
pip show <package-name>
```

### Verify installation
```bash
python -c "import requests; print(requests.__version__)"
python -c "import neo4j; print(neo4j.__version__)"
python -c "import presidio_analyzer; print(presidio_analyzer.__version__)"
```

### Test imports
```bash
python -c "
import requests
import yaml
import neo4j
import pandas
from presidio_analyzer import AnalyzerEngine
print('All imports successful!')
"
```

## 📝 Notes

- **No Poetry/pipenv** - This project uses traditional pip requirements
- **Docker recommended** - Dependencies managed automatically
- **Local development** - Use virtual environment
- **Python 3.12+** - Required for all features
- **spaCy model** - Must be downloaded separately

---

**For Docker deployment:** Dependencies are handled automatically  
**For local development:** Run `./setup_local.sh` or `setup_local.bat`
