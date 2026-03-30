# NJS Pipeline Container
# Uses Poetry for dependency management

FROM python:3.12-slim

LABEL maintainer="NJS Team"
LABEL description="NJS Node-to-Core Pipeline"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy Poetry files first (for layer caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --only main && \
    rm -rf $POETRY_CACHE_DIR

# Download spaCy model for Presidio
RUN poetry run python -m spacy download en_core_web_lg

# Copy application code
COPY . /app/

# Install project
RUN poetry install --only main

# Create .env from template if it doesn't exist
RUN if [ -f .env.example ] && [ ! -f .env ]; then \
        cp .env.example .env; \
    fi

# Set permissions for scripts
RUN chmod +x docker/entrypoint.sh docker/healthcheck.sh && \
    chmod +x scripts/*.py

# Create app user (non-root)
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/docker/healthcheck.sh

# Entry point
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["run-pipeline"]
