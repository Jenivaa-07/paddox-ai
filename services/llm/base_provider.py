from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pydantic import BaseModel

class ProviderResult(BaseModel):
    answer: str
    refused: bool = False
    refusal_reason: Optional[str] = None
    provider: str
    model: str
    fallback_used: bool = False
    latency_ms: float

class LLMProvider(ABC):
    @abstractmethod
    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ProviderResult:
        """
        Generates an answer using the specific LLM provider.
        context_docs should be a list of dictionaries with 'content' and 'metadata'.
        """
        pass
