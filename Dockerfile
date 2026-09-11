# Multi-stage / Production Dockerfile for Advanced RAG Microservice
# Optimized for Google Cloud Run, AWS App Runner, and Docker environments

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output for real-time Cloud Run logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install minimal OS dependencies for compiling C extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install lightweight CPU-only PyTorch to avoid massive ~3GB CUDA wheels in Cloud Run
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency definition
COPY requirements.txt .

# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY . .

# Ensure storage directories exist with appropriate permissions
RUN mkdir -p /app/documents /app/chroma_db /app/static

# Expose standard Cloud Run container port
EXPOSE 8080

# Healthcheck for container orchestrators (GCP Cloud Run / Kubernetes / AWS ECS)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch production ASGI server (Uvicorn) binding dynamically to Cloud Run's $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
