"""
FastAPI Server for Advanced RAG System.
Provides RESTful endpoints for querying, streaming, and document management.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.rag_system import SimpleRAG

app = FastAPI(
    title="Advanced RAG API",
    description="Production-ready RAG microservice with hybrid search, re-ranking, and validation",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static and documents directory exist
STATIC_DIR = Path(__file__).parent / "static"
DOCUMENTS_DIR = Path(__file__).parent / "documents"
STATIC_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize global RAG instance
rag_system: Optional[SimpleRAG] = None

def get_rag() -> SimpleRAG:
    """Lazy initialize RAG system to prevent startup delay."""
    global rag_system
    if rag_system is None:
        print(" Initializing SimpleRAG backend...")
        rag_system = SimpleRAG()
        # Auto-load existing documents if ChromaDB doesn't have them
        existing_docs = list(DOCUMENTS_DIR.glob("*.*"))
        valid_files = [str(f) for f in existing_docs if f.suffix.lower() in [".pdf", ".txt", ".md"]]
        if valid_files and (not rag_system.vectorstore or len(rag_system.get_existing_documents()) == 0):
            try:
                print(f" Ingesting {len(valid_files)} documents on startup...")
                rag_system.load_from_files(valid_files)
            except Exception as e:
                print(f" Document auto-ingestion warning: {e}")
    return rag_system

# Request / Response Schemas
class ChatRequest(BaseModel):
    question: str
    use_hybrid: Optional[bool] = None
    hybrid_alpha: Optional[float] = None
    use_reranking: Optional[bool] = None
    use_validation: Optional[bool] = None

@app.get("/")
async def serve_frontend():
    """Serve the modern single-page application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Advanced RAG API is running. index.html not found in static/."}

@app.get("/api/health")
async def health_check():
    """Check API health and current provider status."""
    rag = get_rag()
    return {
        "status": "healthy",
        "provider": rag.provider,
        "llm_model": rag.llm_model,
        "embedding_model": rag.embedding_model,
        "indexed_documents": len(rag.get_existing_documents()) if rag.vectorstore else 0
    }

@app.get("/api/documents")
async def list_documents():
    """List documents in knowledge base and indexed files."""
    rag = get_rag()
    disk_files = []
    for f in DOCUMENTS_DIR.glob("*.*"):
        if f.suffix.lower() in [".pdf", ".txt", ".md"]:
            disk_files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "is_indexed": rag.check_document_exists(str(f)) if rag.vectorstore else False
            })
    return {"documents": disk_files}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new document (PDF, TXT, MD) and index it."""
    valid_exts = [".pdf", ".txt", ".md"]
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(valid_exts)}"
        )
    
    dest_path = DOCUMENTS_DIR / file.filename
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Index document into RAG
        rag = get_rag()
        res = rag.load_from_files([str(dest_path)])
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File uploaded and indexed successfully into ChromaDB."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Execute a RAG query and return answer, citations, and validation metadata."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    rag = get_rag()
    
    # Apply runtime parameter overrides if requested
    if request.use_hybrid is not None:
        rag.use_hybrid_search = request.use_hybrid
    if request.hybrid_alpha is not None:
        rag.hybrid_search_alpha = request.hybrid_alpha
        if rag.hybrid_retriever:
            rag.hybrid_retriever.hybrid_search_alpha = request.hybrid_alpha
    if request.use_reranking is not None:
        rag.use_reranking = request.use_reranking
        if rag.hybrid_retriever:
            rag.hybrid_retriever.use_reranking = request.use_reranking
            
    try:
        query_result = rag.query(
            request.question,
            enable_validation=request.use_validation
        )
        
        return {
            "question": request.question,
            "answer": query_result.get("result", ""),
            "citations": query_result.get("citations", []),
            "validation": query_result.get("validation", {}),
            "routing": query_result.get("routing", {}),
            "timings": getattr(rag, "_last_query_timings", {}),
            "is_general_answer": query_result.get("is_general_answer", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/clear")
async def clear_chat_history():
    """Clear conversational memory buffer."""
    rag = get_rag()
    if rag.memory_manager:
        rag.memory_manager.clear()
    return {"status": "success", "message": "Conversation memory reset."}

if __name__ == "__main__":
    import uvicorn
    # Pre-initialize RAG on startup
    get_rag()
    print("\n" + "="*60)
    print(" 🚀 Starting Advanced RAG FastAPI Server")
    print(" 🌐 Web UI: http://127.0.0.1:8000")
    print(" 📚 API Docs: http://127.0.0.1:8000/docs")
    print("="*60 + "\n")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
