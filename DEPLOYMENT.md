# PADDOX AI Deployment Guide

## Prerequisites
- Docker
- External Cloud Storage bucket for Artifacts (e.g. AWS S3, Google Cloud Storage)

## Environment Variables
The container expects the following environment variables:
- `ARTIFACT_BUCKET_URL`: The URL to the bucket containing models.
- `EXPECTED_RF_SHA256`: Checksum for the fantasy RF model.
- `GROQ_API_KEY`: Key for Groq Whisper transcription.

## Deployment Steps
1. Push your model artifacts to the external bucket.
2. Build the Docker image:
   `docker build -t paddox-ai:latest .`
3. Run the container:
   `docker run -p 8000:8000 -e ARTIFACT_BUCKET_URL=... -e EXPECTED_RF_SHA256=... paddox-ai:latest`
4. Use `/health` for liveness checks and `/ready` for readiness checks (ensures models downloaded).

## FAISS Rebuild Process
If you need to rebuild the FAISS knowledge index deterministically:
1. Run `python scripts/rag/build_index.py`
2. Push the resulting `.pkl` file to the artifact storage alongside the models.
