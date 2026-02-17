# DataLegos Pipeline Container
# Follows DataLegos maintainability standards

FROM python:3.12-slim

LABEL maintainer="DataLegos Team"
LABEL description="DataLegos Node-to-Core Pipeline"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt requirements_presidio.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_presidio.txt

# Download spaCy model for Presidio
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY . /app/

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
