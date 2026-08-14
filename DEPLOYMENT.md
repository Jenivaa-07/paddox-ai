# PADDOX AI Deployment Guide

## Prerequisites
- Docker
- External Cloud Storage bucket for Artifacts (e.g. AWS S3, Google Cloud Storage)

## Environment Variables
The container expects the following environment variables:
- `ARTIFACT_BUCKET_URL`: The URL to the bucket containing models.
- `EXPECTED_RF_SHA256`: Checksum for the fantasy RF model.
- `GROQ_API_KEY`: Key for Groq Whisper transcription.
- `AI_SERVICE_KEY`: Shared secret also configured on the Node backend. It protects `/chat` from direct public use.
- `RAG_MIN_RELEVANCE`: Retrieval threshold; defaults to `0.28`.
- `REQUIRE_PREDICTIVE_MODELS`: Defaults to `false`, allowing the RAG service to start without the optional race/fantasy artifacts. Set it to `true` when those predictors must be available.

## Deployment Steps
1. Push your model artifacts to the external bucket.
2. Build the Docker image:
   `docker build -t paddox-ai:latest .`
3. Run the container:
   `docker run -p 8000:8000 -e ARTIFACT_BUCKET_URL=... -e EXPECTED_RF_SHA256=... paddox-ai:latest`
4. Use `/health` for liveness checks and `/ready` for readiness checks (ensures models downloaded).
5. Use `/rag/health` to confirm the embedding model and knowledge index are ready.

## FAISS Rebuild Process
If you need to rebuild the FAISS knowledge index deterministically:
1. Run `python knowledge/ingest.py`.
2. Store both `index.faiss` and `index.pkl` together for the new version. If a complete pair is not present, the service safely rebuilds an in-memory index from the committed knowledge documents.
