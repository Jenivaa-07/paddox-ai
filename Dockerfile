FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment setup
ENV PYTHONPATH="."
ENV HOST="0.0.0.0"
ENV PORT=8000

# Expose port
EXPOSE 8000

# Download models and start service
CMD python download_models.py && uvicorn main:app --host $HOST --port $PORT
