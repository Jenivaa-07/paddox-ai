import os
from typing import List, Dict
from fastapi import HTTPException
from .base_provider import ProviderResult
from .groq_provider import GroqProvider

class ProviderRouter:
    def __init__(self):
        self.primary_provider_name = os.getenv("LLM_PRIMARY_PROVIDER", "groq")
        
        self.groq = GroqProvider()

    def _is_fallbackable_error(self, e: Exception) -> bool:
        error_str = str(e).lower()
        # Fallback only for: timeout, network failure, 429, 500, 502, 503, 504, empty response, invalid structured response
        fallback_keywords = [
            "timeout", "network", "429", "500", "502", "503", "504",
            "empty provider response", "invalid structured response",
            "connection error", "rate limit"
        ]
        
        # Do NOT fallback for: safety refusal, prompt-injection, missing retrieval context, request validation error, or unsupported question.
        non_fallback_keywords = [
            "safety refusal", "prompt-injection", "no sufficient context", 
            "validation error", "unsupported", "invalid api key", "unauthorized", "invalid_argument"
        ]
        
        for k in non_fallback_keywords:
            if k in error_str:
                return False
                
        for k in fallback_keywords:
            if k in error_str:
                return True
                
        return False

    def route_query(
        self,
        query: str,
        context_docs: List[Dict[str, str]],
        history: List[Dict[str, str]] = None,
    ) -> ProviderResult:
        # Since fallback is disabled and we use only groq as active provider
        primary = self.groq
        
        if not primary.client:
            raise HTTPException(status_code=503, detail="llm_unavailable")
            
        try:
            result = primary.generate_answer(query, context_docs, history=history)
            result.fallback_used = False
            return result
        except Exception as e:
            # Check for transitent error, try once more
            if self._is_fallbackable_error(e):
                try:
                    result = primary.generate_answer(query, context_docs, history=history)
                    result.fallback_used = False
                    return result
                except Exception as retry_e:
                    raise HTTPException(status_code=503, detail=f"primary_failed: {str(retry_e)}")
            else:
                # Non-fallbackable error (e.g. prompt injection, missing key at runtime, etc)
                raise HTTPException(status_code=503, detail=f"primary_error: {str(e)}")
