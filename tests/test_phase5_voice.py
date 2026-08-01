import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

os.environ["GROQ_API_KEY"] = "fake_test_key"

from main import app
from services.voice.voice_intent_router import VoiceIntentRouter

client = TestClient(app)

def test_intent_routing():
    assert VoiceIntentRouter.route_intent("I want to buy a ticket") == "action_refusal"
    assert VoiceIntentRouter.route_intent("Who won the 2021 championship?") == "general_rag"
    assert VoiceIntentRouter.route_intent("What is the gap to Norris live?") == "live_race"

@patch('services.voice.voice_privacy.VoicePrivacy.get_audio_duration')
@patch('services.voice.transcription_service.Groq')
def test_transcribe_endpoint(mock_groq, mock_dur):
    mock_dur.return_value = 5.0
    from services.voice.transcription_service import transcription_service
    transcription_service.api_key = "fake_test_key"
    
    mock_client = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "This is a test transcript."
    mock_client.audio.transcriptions.create.return_value = mock_transcription
    mock_groq.return_value = mock_client

    response = client.post("/voice/transcribe", files={"file": ("test.webm", b"fake audio data", "audio/webm")})
    print("DEBUG 503:", response.text)
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == "This is a test transcript."
    assert data["stt_provider"] == "groq"
    assert data["audio_retained"] == False

@patch('services.voice.transcription_service.Groq')
@patch('services.voice.voice_assistant_service.Groq')
def test_ask_endpoint(mock_groq_llm, mock_groq_stt):
    from services.voice.transcription_service import transcription_service
    transcription_service.api_key = "fake_test_key"
    from services.voice.voice_assistant_service import voice_assistant_service
    voice_assistant_service.groq_api_key = "fake_test_key"
    
    mock_stt_client = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "How does DRS work?"
    mock_stt_client.audio.transcriptions.create.return_value = mock_transcription
    mock_groq_stt.return_value = mock_stt_client
    
    mock_llm_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "DRS reduces drag."
    mock_completion.choices = [mock_choice]
    mock_llm_client.chat.completions.create.return_value = mock_completion
    mock_groq_llm.return_value = mock_llm_client
    
    # We also need to mock the RAG service, but let's see if it works without mocking if it uses local FAISS
    # Actually, RAG service uses LLM factory. Since we only want to test voice integration, this is fine.
    
    # Wait, rag_chatbot internally calls llm_factory, which calls groq or local. 
    # Let's mock rag_chatbot directly for simplicity in this unit test.
    with patch('services.voice.voice_assistant_service.generate_rag_response') as mock_rag, \
         patch('services.voice.voice_privacy.VoicePrivacy.get_audio_duration') as mock_dur:
        mock_dur.return_value = 5.0
        mock_rag.return_value = {
            "answer": "DRS is Drag Reduction System which opens the rear wing flap to reduce drag and increase top speed.",
            "grounded": True,
            "retrieved_context_sources": ["doc_1"]
        }
        
        response = client.post("/voice/ask", files={"file": ("test.webm", b"fake audio data", "audio/webm")})
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "How does DRS work?"
        assert data["intent"] == "general_rag"
        assert data["spoken_answer"] == "DRS is Drag Reduction System which opens the rear wing flap to reduce drag and increase top speed."

def test_spoken_answer_deterministic():
    from services.voice.voice_assistant_service import voice_assistant_service
    answer = "Hamilton finished P1 at lap 52 with 98% accuracy. **This is bold**. [1] http://f1.com."
    cleaned = voice_assistant_service._clean_for_speech(answer)
    assert "Hamilton finished P1 at lap 52 with 98% accuracy." in cleaned
    assert "bold" in cleaned
    assert "**" not in cleaned
    assert "[1]" not in cleaned
    assert "http" not in cleaned

@patch('services.voice.voice_privacy.VoicePrivacy.get_audio_duration')
def test_audio_too_long(mock_dur):
    mock_dur.return_value = 35.0
    response = client.post("/voice/transcribe", files={"file": ("test.webm", b"fake audio", "audio/webm")})
    assert response.status_code == 413
    assert response.json()["detail"] == "audio_too_long"
    
@patch('services.voice.voice_privacy.VoicePrivacy.get_audio_duration')
def test_audio_invalid_format(mock_dur):
    mock_dur.return_value = None
    response = client.post("/voice/transcribe", files={"file": ("test.webm", b"fake audio", "audio/webm")})
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_audio_format"

