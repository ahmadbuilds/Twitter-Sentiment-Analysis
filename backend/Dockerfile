FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860 

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy & install dependencies
COPY backend/requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ .

# Hugging Face Spaces requires the app to run on $PORT
ENV PORT=${PORT}

# Optional: allow start.sh if exists
RUN chmod +x start.sh || true

# Expose port for local testing
EXPOSE ${PORT}

# HEALTHCHECK is optional — Spaces already manage health internally
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Start the app
CMD ["bash", "start.sh"]
