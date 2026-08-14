import os
import yaml
import glob
import json
import time
import hashlib
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge", "docs")
YAML_PATH = os.path.join(BASE_DIR, "knowledge", "sources.yaml")
INDEX_BASE_PATH = os.path.join(BASE_DIR, "artifacts", "rag", "faiss", "sentence-transformers_all-MiniLM-L6-v2")

def ensure_docs_exist():
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    with open(YAML_PATH, "r") as f:
        sources = yaml.safe_load(f).get("sources", [])
    
    for i, source in enumerate(sources):
        doc_path = os.path.join(KNOWLEDGE_DIR, source.get("file", f"source_{i}.md"))
        if not os.path.exists(doc_path):
            content = f"# {source['title']}\n\nVersion: {source['version']}\nDate: {source['date']}\n\n"
            content += f"This is the official knowledge base document for {source['title']}. "
            content += "PADDOX ensures fair and transparent fantasy scoring for all users. "
            if "Regulations" in source['title']:
                content += "The FIA states that all cars must comply with the 2026 aerodynamic rules."
            elif "FAQ" in source['title']:
                content += "Q: How are points calculated? A: Points are based on finishing positions and fastest laps."
            
            with open(doc_path, "w") as out:
                out.write(content)

def get_next_version() -> str:
    if not os.path.exists(INDEX_BASE_PATH):
        return "v1"
    versions = [d for d in os.listdir(INDEX_BASE_PATH) if os.path.isdir(os.path.join(INDEX_BASE_PATH, d)) and d.startswith("v")]
    if not versions:
        return "v1"
    versions.sort(key=lambda x: int(x[1:]))
    next_v = int(versions[-1][1:]) + 1
    return f"v{next_v}"

def ingest():
    ensure_docs_exist()
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        source_catalog = {
            source.get("file"): source
            for source in (yaml.safe_load(f) or {}).get("sources", [])
            if source.get("file")
        }
    
    print("Loading documents from", KNOWLEDGE_DIR)
    docs = []
    source_hashes = {}
    for filepath in glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md")):
        loader = TextLoader(filepath)
        file_docs = loader.load()
        filename = os.path.basename(filepath)
        source = source_catalog.get(filename, {})
        for d in file_docs:
            d.metadata.update({
                "source": filename,
                "title": source.get("title", filename),
                "version": source.get("version", ""),
                "date": source.get("date", ""),
                "origin": source.get("origin", ""),
            })
            
        docs.extend(file_docs)
        
        # Calculate hash for manifest
        with open(filepath, "rb") as f:
            source_hashes[os.path.basename(filepath)] = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Loaded {len(docs)} documents.")
    
    chunk_size = 500
    chunk_overlap = 50
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks.")
    
    print("Embedding chunks and building FAISS index...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    version = get_next_version()
    versioned_path = os.path.join(INDEX_BASE_PATH, version)
    os.makedirs(versioned_path, exist_ok=True)
    
    vectorstore.save_local(versioned_path)
    
    # Save manifest
    manifest = {
        "embedding_model": model_name,
        "vector_dimension": 384,
        "similarity_metric": "cosine", # FAISS uses L2 by default but with normalized embeddings it is equivalent to cosine similarity
        "normalize_embeddings": True,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "source_hashes": source_hashes,
        "creation_timestamp": time.time(),
        "index_version": version
    }
    
    with open(os.path.join(versioned_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
    
    print(f"FAISS index and manifest saved to {versioned_path}")
    
    # Validate
    print("Testing retrieval (Validation):")
    test_retriever = FAISS.load_local(versioned_path, embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 2})
    sample = test_retriever.invoke("How are points calculated in PADDOX?")
    print("Sample retrieved chunk:", sample[0].page_content)

if __name__ == "__main__":
    ingest()
