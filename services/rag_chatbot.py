import json
import os
import re
import time
import uuid
from functools import lru_cache

import yaml

from .llm.provider_router import ProviderRouter

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_BASE_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "rag",
    "faiss",
    "sentence-transformers_all-MiniLM-L6-v2",
)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge", "docs")
SOURCES_PATH = os.path.join(BASE_DIR, "knowledge", "sources.yaml")
EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

_active_index_version = "unavailable"

INJECTION_PATTERNS = (
    r"ignore\s+(all|any|the|previous|prior)\s+(instructions?|prompts?)",
    r"(show|reveal|print|return|output)\s+(your|the)\s+(system\s+)?prompt",
    r"developer\s+message",
    r"jailbreak",
)

FOLLOW_UP_PATTERN = re.compile(
    r"^(and\b|also\b|what about\b|how about\b|tell me more\b|why\b|when\b|where\b|"
    r"who\b|which one\b|can you explain\b)|\b(it|that|this|they|them|those)\b",
    flags=re.IGNORECASE,
)


def _embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _source_catalog():
    if not os.path.exists(SOURCES_PATH):
        return {}
    with open(SOURCES_PATH, "r", encoding="utf-8") as handle:
        configured = yaml.safe_load(handle) or {}
    return {
        source.get("file"): source
        for source in configured.get("sources", [])
        if source.get("file")
    }


def _knowledge_chunks():
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = []
    catalog = _source_catalog()
    if not os.path.isdir(KNOWLEDGE_DIR):
        return []

    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(KNOWLEDGE_DIR, filename)
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if not content:
            continue
        source = catalog.get(filename, {})
        documents.append(Document(
            page_content=content,
            metadata={
                "source": filename,
                "title": source.get("title", filename),
                "version": source.get("version", ""),
                "date": source.get("date", ""),
                "origin": source.get("origin", ""),
            },
        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def load_vectorstore():
    """Load a complete saved index or rebuild deterministically from committed docs."""
    global _active_index_version
    from langchain_community.vectorstores import FAISS

    embeddings = _embedding_model()
    if os.path.isdir(INDEX_BASE_PATH):
        versions = [
            name for name in os.listdir(INDEX_BASE_PATH)
            if name.startswith("v") and name[1:].isdigit()
        ]
        versions.sort(key=lambda name: int(name[1:]), reverse=True)
        for version in versions:
            version_path = os.path.join(INDEX_BASE_PATH, version)
            faiss_path = os.path.join(version_path, "index.faiss")
            document_store_path = os.path.join(version_path, "index.pkl")
            if not (os.path.isfile(faiss_path) and os.path.isfile(document_store_path)):
                continue
            try:
                store = FAISS.load_local(
                    version_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                _active_index_version = version
                return store
            except Exception as error:
                print(f"Skipping incomplete RAG index {version}: {error}")

    chunks = _knowledge_chunks()
    if not chunks:
        _active_index_version = "unavailable"
        return None
    _active_index_version = "runtime-docs"
    return FAISS.from_documents(chunks, embeddings)


def get_rag_status():
    store = load_vectorstore()
    chunk_count = int(getattr(getattr(store, "index", None), "ntotal", 0)) if store else 0
    return {
        "ready": store is not None and chunk_count > 0,
        "index_version": _active_index_version,
        "chunk_count": chunk_count,
        "embedding_model": EMBEDDING_MODEL,
    }


def _source_from_metadata(metadata):
    return {
        "source": str(metadata.get("source", "unknown")),
        "title": str(metadata.get("title", metadata.get("source", "Verified PADDOX source"))),
        "version": str(metadata.get("version", "")),
        "date": str(metadata.get("date", "")),
    }


def _default_suggestions():
    return [
        "When is the next Formula 1 race?",
        "Who leads the driver standings?",
        "What can I do on PADDOX?",
    ]


def _refusal(answer, request_id, reason):
    return {
        "status": "success",
        "answer": answer,
        "grounded": False,
        "sources": [],
        "retrieved_context_sources": [],
        "provider": "retrieval",
        "model": "none",
        "fallback_used": False,
        "latency_ms": 0.0,
        "data_as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": request_id,
        "refusal_reason": reason,
        "suggestions": _default_suggestions(),
        "retrieval": {
            "index_version": _active_index_version,
            "chunks_used": 0,
        },
    }


def _looks_like_prompt_injection(query):
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS)


def _normalise_history(history):
    turns = []
    for turn in list(history or [])[-8:]:
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
            continue
        content = re.sub(r"\s+", " ", str(turn.get("content", ""))).strip()[:1000]
        if not content or _looks_like_prompt_injection(content):
            continue
        turns.append({"role": turn["role"], "content": content})
    return turns


def _retrieval_query(query, history):
    if not history or not FOLLOW_UP_PATTERN.search(query):
        return query
    previous_user_turn = next(
        (turn["content"] for turn in reversed(history) if turn["role"] == "user"),
        "",
    )
    return f"{previous_user_turn}\nFollow-up: {query}" if previous_user_turn else query


def _context_content(value, max_length):
    if not isinstance(value, dict) or not value:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)[:max_length]


def _follow_up_suggestions(sources, live_context=None, user_context=None):
    source_ids = {source.get("source") for source in sources}
    if live_context or "live_f1_context" in source_ids:
        return [
            "When is qualifying?",
            "Who leads the driver standings?",
            "Who won the last race?",
        ]
    if user_context and user_context.get("signedIn"):
        return [
            "How many Fan Points do I have?",
            "What can I do in Fan Pulse?",
            "When is the next Formula 1 race?",
        ]
    if "paddox_shop_policy.md" in source_ids:
        return [
            "Which items cannot be returned?",
            "How long does shipping take?",
            "What can I do on PADDOX?",
        ]
    if "f1_terminology.md" in source_ids or "fia_regulations.md" in source_ids:
        return [
            "What is an overcut?",
            "Explain dirty air",
            "What changed in the 2026 power units?",
        ]
    return _default_suggestions()


def generate_rag_response(
    query: str,
    history: list = None,
    live_context: dict = None,
    user_context: dict = None,
) -> dict:
    request_id = str(uuid.uuid4())
    clean_query = str(query or "").strip()

    if len(clean_query) < 2 or len(clean_query) > 600:
        return _refusal(
            "Please ask a PADDOX or Formula 1 question between 2 and 600 characters.",
            request_id,
            "invalid_query",
        )
    if _looks_like_prompt_injection(clean_query):
        return _refusal(
            "I can’t follow instructions that try to override the AI Pit Wall. Ask me a PADDOX or Formula 1 question instead.",
            request_id,
            "prompt_injection",
        )

    try:
        clean_history = _normalise_history(history)
        search_query = _retrieval_query(clean_query, clean_history)
        vectorstore = load_vectorstore()
        documents = []
        if vectorstore:
            threshold = min(0.95, max(0.0, float(os.getenv("RAG_MIN_RELEVANCE", "0.28"))))
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 4, "score_threshold": threshold},
            )
            documents = retriever.invoke(search_query)

        context_docs = []
        sources = []
        seen_sources = set()
        for document in documents:
            metadata = dict(document.metadata or {})
            context_docs.append({"content": document.page_content, "metadata": metadata})
            source = _source_from_metadata(metadata)
            if source["source"] not in seen_sources:
                seen_sources.add(source["source"])
                sources.append(source)

        # These contexts are accepted only from the authenticated Node gateway.
        live_content = _context_content(live_context, 8000)
        if live_content:
            context_docs.append({
                "content": live_content,
                "metadata": {
                    "source": "live_f1_context",
                    "title": "Current Formula 1 data feed",
                    "version": "live",
                    "date": time.strftime("%Y-%m-%d", time.gmtime()),
                },
            })
            sources.append(_source_from_metadata(context_docs[-1]["metadata"]))

        user_content = _context_content(user_context, 2000)
        if user_content:
            context_docs.append({
                "content": user_content,
                "metadata": {
                    "source": "paddox_profile",
                    "title": "Your PADDOX fan profile",
                    "version": "live",
                    "date": time.strftime("%Y-%m-%d", time.gmtime()),
                },
            })
            sources.append(_source_from_metadata(context_docs[-1]["metadata"]))

        if not context_docs:
            return _refusal(
                "I couldn’t find enough verified PADDOX or F1 evidence for that question. Try asking about PADDOX features, F1 terms, historical data, merch, or policies.",
                request_id,
                "insufficient_retrieval_context",
            )

        result = ProviderRouter().route_query(clean_query, context_docs, history=clean_history)
        grounded = bool(context_docs) and not result.refused
        return {
            "status": "success",
            "answer": result.answer,
            "grounded": grounded,
            "sources": sources if grounded else [],
            "retrieved_context_sources": [source["source"] for source in sources] if grounded else [],
            "provider": result.provider,
            "model": result.model,
            "fallback_used": result.fallback_used,
            "latency_ms": result.latency_ms,
            "data_as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "refusal_reason": result.refusal_reason,
            "suggestions": _follow_up_suggestions(
                sources if grounded else [],
                live_context=live_context,
                user_context=user_context,
            ),
            "retrieval": {
                "index_version": _active_index_version,
                "chunks_used": len(context_docs),
            },
        }
    except Exception as error:
        if hasattr(error, "status_code"):
            raise error
        print(f"RAG request failed: {error}")
        return {
            "status": "error",
            "answer": "The AI Pit Wall could not process that request.",
            "grounded": False,
            "sources": [],
            "retrieved_context_sources": [],
            "provider": "error",
            "model": "error",
            "fallback_used": False,
            "latency_ms": 0.0,
            "data_as_of": None,
            "request_id": request_id,
            "refusal_reason": "internal_error",
            "suggestions": _default_suggestions(),
            "retrieval": {
                "index_version": _active_index_version,
                "chunks_used": 0,
            },
        }
