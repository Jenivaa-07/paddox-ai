import os
import hmac
import json
import platform
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
import glob

# Guarded imports
try:
    import torch
    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    TORCH_VERSION = None
    CUDA_AVAILABLE = False

try:
    import transformers
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import fastf1
    HAS_FASTF1 = True
except ImportError:
    HAS_FASTF1 = False

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Detect CPU/GPU availability and save environment information
    env_info = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "has_torch": HAS_TORCH,
        "torch_version": TORCH_VERSION,
        "cuda_available": CUDA_AVAILABLE,
        "has_transformers": HAS_TRANSFORMERS,
        "has_fastf1": HAS_FASTF1
    }
    
    print("--- Environment Report ---")
    for k, v in env_info.items():
        print(f"{k}: {v}")
        
    with open("environment_report.json", "w") as f:
        json.dump(env_info, f, indent=4)
        
    print("Loading ML models...")
    # Load Sentiment Model if exists and is production-eligible
    artifacts_path = os.path.join(os.path.dirname(__file__), "artifacts", "sentiment")
    reports_path = os.path.join(os.path.dirname(__file__), "reports", "sentiment")
    if os.path.exists(artifacts_path) and os.path.exists(reports_path):
        runs = glob.glob(os.path.join(artifacts_path, "run_*"))
        if runs:
            # Sort by modification time
            runs.sort(key=os.path.getmtime, reverse=True)
            for run_path in runs:
                run_id = os.path.basename(run_path)
                manifest_file = os.path.join(reports_path, run_id, "dataset_manifest.json")
                
                is_prod = False
                if os.path.exists(manifest_file):
                    try:
                        with open(manifest_file, "r") as mf:
                            manifest = json.load(mf)
                            is_prod = manifest.get("eligible_for_production", False)
                    except Exception:
                        pass
                
                if is_prod:
                    print(f"Loading PADDOX production sentiment model from {run_id}")
                    try:
                        models["sentiment"] = pipeline("text-classification", model=run_path, tokenizer=run_path, device=-1)
                    except Exception as e:
                        print(f"Failed to load sentiment model: {e}")
                    break # Loaded the most recent production model
            
            if "sentiment" not in models:
                print("No production-eligible sentiment model found.")
    try:
        rag_status = get_rag_status()
        print(f"RAG ready: {rag_status['ready']} ({rag_status['index_version']}, {rag_status['chunk_count']} chunks)")
    except Exception as error:
        print(f"RAG warm-up failed: {error}")
    print("Models loaded successfully.")
    
    yield
    
    print("Shutting down AI microservice...")
    models.clear()

app = FastAPI(title="PADDOX AI Analytics Engine", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "paddox-ai"}

@app.get("/ready")
def ready_check(response: Response):
    deps = {
        "torch": HAS_TORCH,
        "transformers": HAS_TRANSFORMERS,
        "fastf1": HAS_FASTF1
    }
    
    race_ready = race_predictor_svc.model is not None if hasattr(race_predictor_svc, 'model') else False
    fantasy_ready = fantasy_predictor_svc.pipeline is not None if hasattr(fantasy_predictor_svc, 'pipeline') else False
    
    # Do not treat the smoke-test sentiment model as production-critical
    is_ready = race_ready and fantasy_ready
    
    if not is_ready:
        response.status_code = 503
        
    return {
        "status": "ready" if is_ready else "unavailable",
        "dependencies": deps,
        "models_loaded": {
            "race": race_ready,
            "fantasy": fantasy_ready,
            "sentiment": "sentiment" in models
        }
    }

# --- RAG & Voice ---

from services.rag_chatbot import generate_rag_response, get_rag_status

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=600)

def verify_chat_service_key(x_paddox_ai_key: str | None = Header(default=None)):
    expected = os.getenv("AI_SERVICE_KEY", "").strip()
    if expected and (not x_paddox_ai_key or not hmac.compare_digest(x_paddox_ai_key, expected)):
        raise HTTPException(status_code=401, detail="invalid_service_key")

@app.post("/chat", dependencies=[Depends(verify_chat_service_key)])
def chat(request: ChatRequest):
    return generate_rag_response(request.query)

@app.get("/rag/health")
def rag_health():
    status = get_rag_status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail="rag_not_ready")
    return {"status": "ready", **status}

# --- Local ML Models ---
class SentimentRequest(BaseModel):
    text: str

@app.post("/analyze-sentiment")
def analyze_sentiment(request: SentimentRequest, response: Response):
    if "sentiment" not in models:
        response.status_code = 503
        return {"status": "model_not_ready"}
    try:
        res = models["sentiment"](request.text)[0]
        return {"status": "ok", "sentiment": res["label"], "score": res["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import time
import uuid

from services.race_predictor import RacePredictorService
from services.fantasy_predictor import FantasyPredictorService

race_predictor_svc = RacePredictorService()
fantasy_predictor_svc = FantasyPredictorService()

@app.post("/predict-race")
def predict_race(request: dict, response: Response):
    start_time = time.time()
    try:
        res = race_predictor_svc.predict(request)
        latency_ms = (time.time() - start_time) * 1000
        res["inference_latency_ms"] = round(latency_ms, 2)
        res["request_id"] = str(uuid.uuid4())
        res["data_as_of"] = "2026-07-30T00:00:00Z"
        return res
    except ValueError as e:
        if str(e) == "model_not_ready":
            response.status_code = 503
            return {"status": "model_not_ready"}
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-fantasy")
def predict_fantasy(request: dict, response: Response):
    start_time = time.time()
    try:
        res = fantasy_predictor_svc.predict_batch(request)
        latency_ms = (time.time() - start_time) * 1000
        res["inference_latency_ms"] = round(latency_ms, 2)
        res["request_id"] = str(uuid.uuid4())
        return res
    except ValueError as e:
        if str(e) == "model_not_ready":
            response.status_code = 503
            return {"status": "model_not_ready"}
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from services.recommendation_service import recommendation_service
from services.highlight_service import highlight_service

class RecommendRequest(BaseModel):
    user_id: str
    context: dict = None
    k: int = 10
    exclude_item_ids: list = []

@app.post("/recommend")
def recommend(request: RecommendRequest, response: Response):
    try:
        recs, strategy, model_version, latency = recommendation_service.get_recommendations(
            request.user_id, request.context, request.k, request.exclude_item_ids
        )
        return {
            "recommendations": recs,
            "strategy": strategy,
            "model_version": model_version,
            "data_as_of": "2026-07-31T00:00:00Z",
            "inference_latency_ms": round(latency, 2),
            "request_id": str(uuid.uuid4())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RankHighlightsRequest(BaseModel):
    user_id: str
    race_id: str
    candidate_highlight_ids: list = []
    k: int = 10

from schemas.voice import VoiceTranscribeResponse, VoiceAskResponse
from services.voice.transcription_service import transcription_service
from services.voice.voice_assistant_service import voice_assistant_service
from typing import List, Dict, Tuple, Optional
from fastapi import File, UploadFile, Form
from fastapi.concurrency import run_in_threadpool

@app.post("/voice/transcribe", response_model=VoiceTranscribeResponse)
async def voice_transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    try:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio_too_large")
            
        from services.voice.voice_privacy import VoicePrivacy
        duration = VoicePrivacy.get_audio_duration(content)
        if duration is None:
            raise HTTPException(status_code=400, detail="invalid_audio_format")
        if duration > 30.0:
            raise HTTPException(status_code=413, detail="audio_too_long")
            
        result = await run_in_threadpool(
            transcription_service.transcribe,
            content,
            file.filename,
            language,
        )
        
        return VoiceTranscribeResponse(
            transcript=result["text"],
            language=result["language"],
            stt_provider=result["provider"],
            stt_model=result["model"],
            stt_latency_ms=result["latency_ms"],
            audio_retained=False,
            request_id=str(uuid.uuid4())
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="invalid_request")

@app.post("/voice/ask", response_model=VoiceAskResponse)
async def voice_ask(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    race_id: Optional[str] = Form(None)
):
    try:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio_too_large")
            
        from services.voice.voice_privacy import VoicePrivacy
        duration = VoicePrivacy.get_audio_duration(content)
        if duration is None:
            raise HTTPException(status_code=400, detail="invalid_audio_format")
        if duration > 30.0:
            raise HTTPException(status_code=413, detail="audio_too_long")
            
        response = await run_in_threadpool(
            voice_assistant_service.process_voice_query,
            content,
            file.filename,
            language,
            race_id,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="invalid_request")

@app.post("/rank-highlights")
def rank_highlights(request: RankHighlightsRequest, response: Response):
    try:
        res = highlight_service.rank_highlights(
            request.user_id, request.race_id, request.candidate_highlight_ids, request.k
        )
        if res.get("status") == "authorized_inventory_unavailable":
            return res
            
        return {
            "highlights": res["highlights"],
            "filtered_for_rights": res["filtered_for_rights"],
            "duplicate_suppressed": res["duplicate_suppressed"],
            "model_version": res["model_version"],
            "inference_latency_ms": round(res["inference_latency_ms"], 2),
            "request_id": str(uuid.uuid4())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
