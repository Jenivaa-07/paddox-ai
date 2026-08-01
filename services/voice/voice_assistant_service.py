import time
import os
import uuid
import logging
from typing import Dict, Any, Tuple
from fastapi import HTTPException
from schemas.voice import VoiceAskResponse, VoiceLatencyMetrics
from services.voice.transcription_service import transcription_service
from services.voice.voice_intent_router import VoiceIntentRouter
from services.rag_chatbot import generate_rag_response
from groq import Groq

logger = logging.getLogger(__name__)

class VoiceAssistantService:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
    def _clean_for_speech(self, detailed_answer: str) -> str:
        import re
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', detailed_answer)
        # Remove markdown formatting (**, *, #, `)
        text = re.sub(r'[*#`]', '', text)
        # Remove citation identifiers like [1], [citation]
        text = re.sub(r'\[.*?\]', '', text)
        
        # Split into sentences (rudimentary)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        # Keep first 1-3 useful sentences
        kept_sentences = sentences[:3]
        spoken = " ".join(kept_sentences)
        
        # Enforce character limit
        if len(spoken) > 400:
            spoken = spoken[:397] + "..."
            
        return spoken

    def process_voice_query(self, audio_bytes: bytes, filename: str, language: str = None, race_id: str = None) -> VoiceAskResponse:
        total_start = time.time()
        
        # 1. Transcription
        stt_start = time.time()
        stt_result = transcription_service.transcribe(audio_bytes, filename, language)
        stt_ms = (time.time() - stt_start) * 1000
        transcript = stt_result["text"]
        
        # 2. Intent Routing
        intent = VoiceIntentRouter.route_intent(transcript)
        
        detailed_answer = ""
        spoken_answer = ""
        grounded = False
        citations = []
        data_as_of = None
        retrieval_ms = 0.0
        llm_ms = 0.0
        
        if intent == "action_refusal":
            detailed_answer = "I can explain how to do that, but I cannot perform account, payment or administrative actions through voice."
            spoken_answer = detailed_answer
        elif intent == "live_race":
            detailed_answer = "Live race data is currently unavailable. No active session found."
            spoken_answer = detailed_answer
            grounded = False
            citations = []
        else:
            # RAG (Historical/General)
            rag_start = time.time()
            rag_resp = generate_rag_response(transcript)
            rag_duration = (time.time() - rag_start) * 1000
            retrieval_ms = rag_duration * 0.4 # Estimate retrieval vs llm split
            llm_ms = rag_duration * 0.6
            
            detailed_answer = rag_resp["answer"]
            grounded = rag_resp["grounded"]
            citations = rag_resp.get("retrieved_context_sources", [])
            
            llm_summary_start = time.time()
            spoken_answer = self._clean_for_speech(detailed_answer)
            llm_ms += (time.time() - llm_summary_start) * 1000
            
        total_ms = (time.time() - total_start) * 1000
        
        return VoiceAskResponse(
            transcript=transcript,
            intent=intent,
            answer=detailed_answer,
            spoken_answer=spoken_answer,
            grounded=grounded,
            citations=citations,
            data_as_of=data_as_of,
            stt_provider=stt_result["provider"],
            stt_model=stt_result["model"],
            llm_provider="groq",
            latency=VoiceLatencyMetrics(
                stt_ms=round(stt_ms, 2),
                retrieval_ms=round(retrieval_ms, 2),
                llm_ms=round(llm_ms, 2),
                total_ms=round(total_ms, 2)
            ),
            audio_retained=False,
            request_id=str(uuid.uuid4())
        )

voice_assistant_service = VoiceAssistantService()
