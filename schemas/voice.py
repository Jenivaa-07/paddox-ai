from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class VoiceTranscribeResponse(BaseModel):
    transcript: str
    language: str
    stt_provider: str
    stt_model: str
    stt_latency_ms: float
    audio_retained: bool
    request_id: str

class VoiceLatencyMetrics(BaseModel):
    stt_ms: float
    retrieval_ms: float
    llm_ms: float
    total_ms: float

class VoiceAskResponse(BaseModel):
    transcript: str
    intent: str
    answer: str
    spoken_answer: str
    grounded: bool
    citations: List[str]
    data_as_of: Optional[str] = None
    stt_provider: str
    stt_model: str
    llm_provider: str
    latency: VoiceLatencyMetrics
    audio_retained: bool
    request_id: str
