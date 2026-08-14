FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache the local embedding model at image-build time so RAG does not depend on
# a first-request download in production.
ENV HF_HOME="/opt/paddox-huggingface"
ENV SENTENCE_TRANSFORMERS_HOME="/opt/paddox-huggingface"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

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
