# PADDOX AI Microservice

This repository hosts the AI backend for PADDOX, providing advanced machine learning capabilities for the platform.

## Features
- Deep Learning Race Prediction (LSTM)
- Fantasy Driver Analysis (Random Forest)
- Real-time Fan Sentiment Analysis (DistilBERT)
- RAG-powered Chat & Voice Assistance

## Setup
Install dependencies:
```bash
pip install -r requirements-lock.txt
```

Run locally:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
