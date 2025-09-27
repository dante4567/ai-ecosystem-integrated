#!/usr/bin/env python3

import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Module",
    description="Retrieval-Augmented Generation service",
    version="1.0.0"
)

PORT = int(os.getenv("PORT", 8001))

# Sample knowledge base
KNOWLEDGE_BASE = {
    "ai": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "rag": "Retrieval-Augmented Generation combines retrieval of relevant documents with text generation.",
    "llm": "Large Language Models are AI systems trained on vast amounts of text data.",
    "docker": "Docker is a platform for developing, shipping, and running applications in containers.",
    "fastapi": "FastAPI is a modern, fast web framework for building APIs with Python."
}

class QueryRequest(BaseModel):
    query: str
    max_results: int = 5

class QueryResponse(BaseModel):
    query: str
    results: list
    timestamp: str
    processing_time: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "RAG Module",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "knowledge_base_size": len(KNOWLEDGE_BASE)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Module API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "documents": "/documents"
        }
    }

@app.post("/search", response_model=QueryResponse)
async def search_documents(request: QueryRequest):
    """Search the knowledge base"""
    start_time = datetime.now()

    query_lower = request.query.lower()
    results = []

    for key, value in KNOWLEDGE_BASE.items():
        if query_lower in key.lower() or query_lower in value.lower():
            results.append({
                "id": key,
                "content": value,
                "relevance_score": 0.95 if query_lower in key.lower() else 0.75
            })

    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results = results[:request.max_results]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return QueryResponse(
        query=request.query,
        results=results,
        timestamp=datetime.now().isoformat(),
        processing_time=processing_time
    )

@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base"""
    return {
        "total_documents": len(KNOWLEDGE_BASE),
        "documents": [
            {"id": key, "preview": value[:100] + "..." if len(value) > 100 else value}
            for key, value in KNOWLEDGE_BASE.items()
        ]
    }

@app.post("/documents/{doc_id}")
async def add_document(doc_id: str, content: dict):
    """Add a new document to the knowledge base"""
    if "content" not in content:
        raise HTTPException(status_code=400, detail="Content field is required")

    KNOWLEDGE_BASE[doc_id] = content["content"]
    return {
        "message": f"Document {doc_id} added successfully",
        "document_id": doc_id,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )