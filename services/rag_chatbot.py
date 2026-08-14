import os
import json
import uuid
import time
from functools import lru_cache
from .llm.provider_router import ProviderRouter

# Constants
INDEX_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "rag", "faiss", "sentence-transformers_all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def load_vectorstore():
    # Load the latest versioned FAISS index
    if not os.path.exists(INDEX_BASE_PATH):
        return None
        
    versions = [d for d in os.listdir(INDEX_BASE_PATH) if os.path.isdir(os.path.join(INDEX_BASE_PATH, d)) and d.startswith("v")]
    if not versions:
        return None
        
    versions.sort(key=lambda x: int(x[1:]))
    latest_version = versions[-1]
    
    versioned_path = os.path.join(INDEX_BASE_PATH, latest_version)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = FAISS.load_local(versioned_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def generate_rag_response(query: str, live_context: dict = None) -> dict:
    request_id = str(uuid.uuid4())
    
    try:
        vectorstore = load_vectorstore()
        context_docs = []
        citations = []
        
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                if source not in citations:
                    citations.append(source)
                context_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        if live_context:
            context_docs.append({
                "content": json.dumps(live_context),
                "metadata": {"source": "live_context"}
            })
            citations.append("live_context")
            
        router = ProviderRouter()
        result = router.route_query(query, context_docs)
        
        return {
            "status": "success",
            "answer": result.answer,
            "grounded": not result.refused,
            "retrieved_context_sources": citations,
            "provider": result.provider,
            "model": result.model,
            "fallback_used": result.fallback_used,
            "latency_ms": result.latency_ms,
            "data_as_of": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "request_id": request_id
        }
        
    except Exception as e:
        # Standardize the error response without falling back silently on non-provider issues
        # Or letting the router's 503 HTTPExceptions bubble up
        if hasattr(e, 'status_code'):
            raise e
        return {
            "status": "error",
            "answer": f"Error: Unable to process request. ({str(e)})",
            "grounded": False,
            "retrieved_context_sources": [],
            "provider": "error",
            "model": "error",
            "fallback_used": False,
            "latency_ms": 0.0,
            "data_as_of": None,
            "request_id": request_id
        }

# --- OpenAI Voice Implementation Placeholder ---
# Any future OpenAI Voice implementation should be moved to a dedicated 
# services/voice module and not mixed with the LLM RAG chatbot configuration.
