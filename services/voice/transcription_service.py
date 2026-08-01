import os
import time
import logging
from groq import Groq
from services.voice.voice_privacy import VoicePrivacy
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("VOICE_STT_MODEL", "whisper-large-v3-turbo")
        self.default_language = os.getenv("VOICE_LANGUAGE", "en")
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set. Transcription will fail.")
            
        try:
            with open("config/motorsport_vocabulary.txt", "r") as f:
                self.vocabulary = f.read().replace('\n', ', ')
        except Exception:
            self.vocabulary = "Formula 1, PADDOX, Verstappen, Hamilton, Leclerc"

    def transcribe(self, audio_bytes: bytes, filename: str, language: str = None):
        if not self.api_key:
            raise HTTPException(status_code=503, detail="transcription_provider_unavailable")
            
        start_time = time.time()
        temp_path = None
        
        try:
            client = Groq(api_key=self.api_key)
            
            transcription = client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model=self.model,
                prompt=self.vocabulary,
                language=language if language else self.default_language,
                response_format="json",
                temperature=0.0
            )
                
            text = transcription.text.strip()
            if not text:
                raise HTTPException(status_code=400, detail="no_speech_detected")
                
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "text": text,
                "language": language if language else self.default_language,
                "latency_ms": latency_ms,
                "model": self.model,
                "provider": "groq"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            if "timeout" in str(e).lower():
                raise HTTPException(status_code=503, detail="transcription_provider_unavailable")
            raise HTTPException(status_code=500, detail="transcription_provider_unavailable")

transcription_service = TranscriptionService()
