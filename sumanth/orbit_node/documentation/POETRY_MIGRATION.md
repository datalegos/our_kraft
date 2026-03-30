# Poetry Migration Guide

Complete guide for the migration from pip requirements to Poetry.

## 🎯 What Changed

### Before (pip requirements)
```
requirements.txt
requirements_presidio.txt
requirements-dev.txt
setup_local.sh
```

### After (Poetry)
```
pyproject.toml          # Project config + dependencies
poetry.lock             # Locked versions (auto-generated)
setup_poetry.sh         # Poetry setup script
```

## 📦 New Dependency Management

### Poetry Benefits

1. **Single File** - All dependencies in `pyproject.toml`
2. **Lock File** - `poetry.lock` ensures reproducible builds
3. **Dependency Resolution** - Automatic conflict resolution
4. **Virtual Environments** - Built-in venv management
5. **Project Metadata** - All project info in one place

## 🚀 Quick Start

### Docker (Recommended - No Changes Needed)

```bash
cd deployment_scripts
./start_all.sh
# Poetry is used automatically in Docker
```

### Local Development

**Linux/Mac:**
```bash
chmod +x setup_poetry.sh
./setup_poetry.sh
```

**Windows:**
```bash
setup_poetry.bat
```

## 📋 Poetry Commands

### Setup & Installation

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Install only main dependencies (no dev)
poetry install --only main

# Download spaCy model
poetry run python -m spacy download en_core_web_lg
```

### Running Scripts

```bash
# Activate virtual environment
poetry shell

# Run scripts directly
python scripts/main.py
python scripts/detect_pii.py

# Or run without activating
poetry run python scripts/main.py
poetry run python scripts/detect_pii.py
```

### Managing Dependencies

```bash
# Add new dependency
poetry add requests

# Add dev dependency
poetry add --group dev pytest

# Remove dependency
poetry remove requests

# Update all dependencies
poetry update

# Update specific package
poetry update requests

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree

# Show outdated packages
poetry show --outdated
```

### Virtual Environment

```bash
# Activate virtual environment
poetry shell

# Deactivate
exit  # or Ctrl+D

# Show environment info
poetry env info

# Remove virtual environment
poetry env remove python

# List environments
poetry env list
```

## 🔄 Migration Steps

### If You Have Existing pip Environment

1. **Remove old virtual environment:**
   ```bash
   rm -rf venv
   ```

2. **Install Poetry:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies with Poetry:**
   ```bash
   poetry install
   ```

4. **Download spaCy model:**
   ```bash
   poetry run python -m spacy download en_core_web_lg
   ```

5. **Activate Poetry environment:**
   ```bash
   poetry shell
   ```

## 📁 File Structure

### pyproject.toml Structure

```toml
[tool.poetry]
name = "orbit-node"
version = "1.0.0"
description = "..."
authors = ["..."]

[tool.poetry.dependencies]
python = "^3.12"
requests = "^2.31.0"
# ... main dependencies

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
# ... dev dependencies

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Dependencies Organization

- **Main dependencies** - `[tool.poetry.dependencies]`
- **Dev dependencies** - `[tool.poetry.group.dev.dependencies]`
- **Docs dependencies** - `[tool.poetry.group.docs.dependencies]`

## 🐳 Docker Changes

### Dockerfile

**Before:**
```dockerfile
COPY requirements.txt requirements_presidio.txt ./
RUN pip install -r requirements.txt
RUN pip install -r requirements_presidio.txt
```

**After:**
```dockerfile
RUN curl -sSL https://install.python-poetry.org | python3 -
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --only main
```

### Container Name

**Changed:** `datalegos_neo4j` → `orbit-neo4j`

All scripts updated to use new name.

## 🔍 Comparison

### Adding Dependencies

**pip:**
```bash
pip install requests
pip freeze > requirements.txt
```

**Poetry:**
```bash
poetry add requests
# poetry.lock updated automatically
```

### Installing Dependencies

**pip:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Poetry:**
```bash
poetry install
# Creates .venv automatically
```

### Running Scripts

**pip:**
```bash
source venv/bin/activate
python scripts/main.py
```

**Poetry:**
```bash
poetry run python scripts/main.py
# or
poetry shell
python scripts/main.py
```

## 🎨 Benefits

### For Development

- ✅ Single source of truth (`pyproject.toml`)
- ✅ Automatic dependency resolution
- ✅ Lock file for reproducibility
- ✅ Built-in virtual environment management
- ✅ Easy dependency updates

### For Production

- ✅ Reproducible builds with `poetry.lock`
- ✅ Faster Docker builds (better caching)
- ✅ No manual requirements.txt management
- ✅ Automatic conflict detection

### For Team

- ✅ Consistent environments across team
- ✅ Easy onboarding (one command: `poetry install`)
- ✅ Clear dependency groups (main, dev, docs)
- ✅ Modern Python packaging standard

## 🚨 Important Notes

### poetry.lock File

- **Auto-generated** - Don't edit manually
- **Commit to git** - Ensures reproducible builds
- **Update with** - `poetry lock` or `poetry update`

### Virtual Environment Location

Poetry creates `.venv` in project directory (configured in setup scripts).

### Compatibility

Old `requirements.txt` files are kept for reference but not used.

To export to requirements.txt:
```bash
poetry export -f requirements.txt --output requirements.txt
```

## 📖 Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry Commands](https://python-poetry.org/docs/cli/)
- [pyproject.toml Specification](https://python-poetry.org/docs/pyproject/)
- [Dependency Groups](https://python-poetry.org/docs/managing-dependencies/)

## 🆘 Troubleshooting

### Poetry not found after installation

```bash
# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or restart terminal
```

### Lock file out of date

```bash
poetry lock --no-update
```

### Dependency conflicts

```bash
# Show dependency tree
poetry show --tree

# Update conflicting package
poetry update <package-name>
```

### Virtual environment issues

```bash
# Remove and recreate
poetry env remove python
poetry install
```

### Cache issues

```bash
# Clear cache
poetry cache clear pypi --all
```

## ✅ Checklist

### For New Setup

- [ ] Install Poetry
- [ ] Run `poetry install`
- [ ] Download spaCy model
- [ ] Activate environment with `poetry shell`

### For Existing Setup

- [ ] Remove old `venv/` directory
- [ ] Install Poetry
- [ ] Run `poetry install`
- [ ] Download spaCy model
- [ ] Test with `poetry run python scripts/main.py`

### For Docker

- [ ] No changes needed
- [ ] Run `./start_all.sh` as usual
- [ ] Poetry used automatically

---

**Migration Status:** Complete  
**Recommended:** Use Poetry for all new development  
**Docker:** Automatically uses Poetry  
**Old files:** Kept for reference, not used
