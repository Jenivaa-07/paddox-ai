# Phase 5: Voice-Based Race Assistant (Closure Report)

## 1. Executive Summary
Phase 5 introduces a privacy-conscious, push-to-talk voice assistant to the PADDOX ecosystem. It allows fans to query historical F1 statistics and contextual RAG knowledge securely through a web interface. The pipeline routes audio to Groq Whisper Large V3 Turbo for transcription, processes the query through the existing RAG pipeline, and returns both detailed text and a deterministic summary suitable for browser-based speech synthesis. 

This phase strictly adhered to all privacy constraints, operating entirely as a user-initiated feature with no continuous background listening and no disk-based temporary file persistence.

## 2. Architecture & Data Flow

### A. Frontend (`assets/js/voice-assistant.js`)
- WebRTC `MediaRecorder` captures audio upon explicit user interaction ("push-to-talk").
- The Alt+Shift+V keyboard shortcut is safely isolated (ignoring inputs/textareas) and requires visible confirmation.
- Maximum recording time is strictly bounded at 30 seconds.
- Audio is transmitted securely to the Node.js backend API.
- Synthesizes the summarized `spoken_answer` using the browser's `SpeechSynthesis` API.

### B. Node.js Gateway (`routes/voice.routes.js`)
- Authenticates and terminates the client connection securely.
- Uses `multer` in-memory storage to bound uploads to a strict 10MB limit.
- Validates supported audio MIME types natively (`audio/webm`, `audio/mp4`, `audio/ogg`, `audio/wav`).
- Forwards the payload natively to the FastAPI backend without writing to disk.

### C. FastAPI Backend (`services/voice/`)
- **Voice Privacy (`voice_privacy.py`)**: All processing operates strictly on in-memory `BytesIO` buffers. No `services/voice/temp/` directories or temporary files are used. The `get_audio_duration` method uses `mutagen` to securely inspect byte durations before processing. Requests exceeding 30.0 seconds yield an HTTP 413 `audio_too_long` error, and undecipherable files yield HTTP 400 `invalid_audio_format`.
- **Transcription Service (`transcription_service.py`)**: Uses the Groq SDK targeting `whisper-large-v3-turbo` entirely via in-memory bytes.
- **Intent Router (`voice_intent_router.py`)**: Classifies text safely (e.g. `action_refusal`, `general_rag`, `live_race`).
- **Assistant Service (`voice_assistant_service.py`)**: Generates the RAG context and deterministically strips markdown, citations, and URLs via `_clean_for_speech` to produce the `spoken_answer`. No secondary LLM condensation is used, preventing latency and hallucination drift.

## 3. Supported Features vs. Experimental Capabilities
- **Static FAISS RAG**: Grounding, citation-validation and refusal controls.
- **Live Race Routing**: Handled safely. If queried, returns `live_data_unavailable` as timestamped server-side race integration is currently PENDING.
- **Fan Hub Sentiment**: Experimental/smoke-test only. The existing DistilBERT model is not production-validated and is not utilized in the voice assistant channel.
- **Latency**: Groq transcription integration implemented; real-provider latency pending measurement under load.

## 4. Privacy & Security Compliance
- **No Wake Words**: The system explicitly does NOT implement hotword detection or persistent microphone polling.
- **In-Memory Audio**: PADDOX does not intentionally persist uploaded audio locally. Audio is transmitted to the configured Groq transcription provider, and provider-side processing is governed by that provider’s applicable data-handling terms.
- **Data Minimization**: The Node.js gateway drops client IP information and limits the payload to the audio buffer and language tag.
- **Grounding, citation-validation and refusal controls**: The pipeline relies completely on Static FAISS knowledge. Numerical values, positions, and entities are deterministically retained in speech generation.

## 5. Test Evidence & Exact Metrics

**Node.js Syntax and Validation**
- Syntax checks (`node --check` on JS frontend and gateway routes): PASSED
- `node test_voice_gateway.js` results: 4 assertions passed (Valid forwarding, 503 timeout handling, 413 file too large handling, 400 validation handling).

**Pytest Full Regression Suite**
Command executed: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest.exe -v`
- **Total Tests Collected**: 43
- **Passed**: 40
- **Skipped**: 3 (due to `RUN_REAL_GROQ_VOICE_TESTS=false`)
- **Warnings**: 2 (standard library deprecations)
- **Failed/Errors**: 0

*Note: Skipped tests target real Groq APIs. No genuine consented human audio datasets were fabricated; real-provider Opt-In testing relies on `synthetic_audio_fixture` datasets which are explicitly labelled as ineligible for production evaluation.*

## 6. Final Statuses
- **Phase 5 Engineering Status**: COMPLETE
- **Phase 5 Mocked STT Validation Status**: PASSED
- **Phase 5 Real Groq STT Integration Status**: PENDING
- **Phase 5 Human Audio Evaluation Status**: PENDING – NO CONSENTED AUDIOSET
- **Phase 5 Live Race Voice Status**: PENDING
- **Phase 5 Privacy Validation Status**: PASSED (In-memory strict bounds)
- **Phase 5 Frontend Integration Status**: LOCALLY VALIDATED
- **Phase 5 Production Readiness Status**: LIMITED
- **Phase 5 IEEE Result Eligibility**: PENDING
