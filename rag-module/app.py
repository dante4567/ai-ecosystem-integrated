#!/usr/bin/env python3

import os
import json
import sys
sys.path.append('../shared-config')

from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging

from database import get_db, db_config
from models import Document, DocumentChunk, SearchQuery
from vector_search import vector_search
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced RAG Module",
    description="Advanced Retrieval-Augmented Generation service with vector search and persistence",
    version="2.0.0"
)

PORT = int(os.getenv("PORT", 8001))

# Sample knowledge base for initial seeding
INITIAL_KNOWLEDGE = {
    "ai": "Artificial Intelligence is the simulation of human intelligence in machines. It encompasses machine learning, neural networks, natural language processing, computer vision, and robotics. AI systems can learn, reason, perceive, and make decisions.",
    "rag": "Retrieval-Augmented Generation combines retrieval of relevant documents with text generation. It enhances language models by providing them with external knowledge from a vector database, improving accuracy and reducing hallucinations.",
    "llm": "Large Language Models are AI systems trained on vast amounts of text data. Examples include GPT-4, Claude, Gemini, and LLaMA. They can understand and generate human-like text for various applications.",
    "docker": "Docker is a platform for developing, shipping, and running applications in containers. It provides lightweight virtualization, enabling consistent deployment across different environments.",
    "fastapi": "FastAPI is a modern, fast web framework for building APIs with Python. It features automatic OpenAPI documentation, type hints support, and high performance comparable to NodeJS and Go.",
    "vector_search": "Vector search uses embeddings to find semantically similar documents. It converts text into high-dimensional vectors and uses cosine similarity or other distance metrics for retrieval.",
    "microservices": "Microservices architecture breaks applications into small, independent services that communicate over APIs. It enables scalability, maintainability, and technology diversity."
}

class QueryRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    use_vector_search: bool = True
    include_metadata: bool = False

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    timestamp: str
    processing_time: float
    search_method: str
    stats: Dict[str, Any]

class DocumentCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    content_type: str = "text"
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    content_type: str
    source_url: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    content_type: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BulkUploadRequest(BaseModel):
    documents: List[DocumentCreateRequest]

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Enhanced RAG Module v2.0")

    # Initialize database
    if await db_config.initialize():
        await db_config.create_tables()
        logger.info("Database initialized")

    # Initialize vector search
    if await vector_search.initialize():
        logger.info("Vector search initialized")

        # Seed initial knowledge if database is empty
        async with db_config.get_session() as session:
            await seed_initial_knowledge(session)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await vector_search.close()
    await db_config.close()
    logger.info("Enhanced RAG Module shutdown complete")

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Enhanced health check endpoint"""
    start_time = datetime.now()

    # Check database connection
    db_healthy = False
    db_doc_count = 0
    try:
        result = await db.execute(select(func.count(Document.id)))
        db_doc_count = result.scalar()
        db_healthy = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    # Check vector search
    vector_stats = await vector_search.get_collection_stats()
    vector_healthy = vector_stats.get("status") == "healthy"

    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()

    overall_status = "healthy" if db_healthy and vector_healthy else "degraded"

    return {
        "service": "Enhanced RAG Module",
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "response_time": f"{response_time:.3f}s",
        "components": {
            "database": {
                "status": "healthy" if db_healthy else "unhealthy",
                "document_count": db_doc_count
            },
            "vector_search": vector_stats
        }
    }

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "Enhanced RAG Module API",
        "version": "2.0.0",
        "description": "Advanced RAG with vector search, persistence, and semantic retrieval",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "documents": "/documents",
            "documents_create": "/documents (POST)",
            "documents_bulk": "/documents/bulk (POST)",
            "documents_by_id": "/documents/{doc_id}",
            "stats": "/stats",
            "reindex": "/reindex"
        },
        "features": [
            "Vector similarity search",
            "PostgreSQL persistence",
            "Semantic document retrieval",
            "Batch document upload",
            "Full-text search fallback"
        ]
    }

@app.post("/search", response_model=QueryResponse)
async def search_documents(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """Enhanced search with vector similarity and fallback options"""
    start_time = datetime.now()

    # Store search query for analytics
    search_query = SearchQuery(
        query_text=request.query,
        result_count=0,
        metadata={"threshold": request.threshold, "max_results": request.max_results}
    )
    db.add(search_query)

    results = []
    search_method = "none"

    try:
        if request.use_vector_search:
            # Try vector search first
            vector_results = await vector_search.search_similar(
                request.query,
                n_results=request.max_results,
                threshold=request.threshold
            )

            if vector_results:
                results = vector_results
                search_method = "vector"
            else:
                # Fallback to database full-text search
                results = await fallback_text_search(db, request)
                search_method = "text_fallback"
        else:
            # Direct database search
            results = await fallback_text_search(db, request)
            search_method = "text"

        # Enhance results with database metadata if requested
        if request.include_metadata and results:
            results = await enhance_results_with_metadata(db, results)

    except Exception as e:
        logger.error(f"Search error: {e}")
        # Ultimate fallback to simple text matching
        results = await simple_text_search(db, request)
        search_method = "simple_fallback"

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Update search query record
    search_query.result_count = len(results)
    search_query.processing_time = processing_time
    await db.commit()

    # Generate search statistics
    stats = {
        "vector_search_available": vector_search.embedding_model is not None,
        "database_documents": await get_document_count(db),
        "vector_documents": await vector_search.get_document_count()
    }

    return QueryResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        timestamp=datetime.now().isoformat(),
        processing_time=processing_time,
        search_method=search_method,
        stats=stats
    )

async def fallback_text_search(db: AsyncSession, request: QueryRequest) -> List[Dict[str, Any]]:
    """Fallback text search using database LIKE queries"""
    try:
        query = select(Document).where(
            and_(
                Document.is_active == True,
                or_(
                    Document.title.ilike(f"%{request.query}%"),
                    Document.content.ilike(f"%{request.query}%")
                )
            )
        ).limit(request.max_results)

        result = await db.execute(query)
        documents = result.scalars().all()

        return [
            {
                "id": str(doc.id),
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "title": doc.title,
                "relevance_score": 0.6,  # Fixed score for text search
                "similarity_score": 0.6,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Fallback text search error: {e}")
        return []

async def simple_text_search(db: AsyncSession, request: QueryRequest) -> List[Dict[str, Any]]:
    """Simple text matching as ultimate fallback"""
    try:
        query = select(Document).where(Document.is_active == True).limit(50)
        result = await db.execute(query)
        documents = result.scalars().all()

        query_lower = request.query.lower()
        matches = []

        for doc in documents:
            content_lower = doc.content.lower()
            title_lower = doc.title.lower()

            if query_lower in content_lower or query_lower in title_lower:
                score = 0.8 if query_lower in title_lower else 0.5
                matches.append({
                    "id": str(doc.id),
                    "content": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                    "title": doc.title,
                    "relevance_score": score,
                    "similarity_score": score,
                    "metadata": doc.metadata
                })

        matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        return matches[:request.max_results]

    except Exception as e:
        logger.error(f"Simple text search error: {e}")
        return []

async def enhance_results_with_metadata(db: AsyncSession, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhance results with additional database metadata"""
    try:
        doc_ids = [result["id"] for result in results if "id" in result]
        if not doc_ids:
            return results

        # Convert string IDs to UUID objects for database query
        uuid_ids = []
        for doc_id in doc_ids:
            try:
                uuid_ids.append(uuid.UUID(doc_id))
            except ValueError:
                continue

        if not uuid_ids:
            return results

        query = select(Document).where(Document.id.in_(uuid_ids))
        result = await db.execute(query)
        documents = {str(doc.id): doc for doc in result.scalars().all()}

        # Enhance results
        enhanced_results = []
        for result in results:
            doc_id = result.get("id")
            if doc_id and doc_id in documents:
                doc = documents[doc_id]
                result.update({
                    "title": doc.title,
                    "source_url": doc.source_url,
                    "content_type": doc.content_type,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                    "database_metadata": doc.metadata
                })

            enhanced_results.append(result)

        return enhanced_results

    except Exception as e:
        logger.error(f"Error enhancing results: {e}")
        return results

async def get_document_count(db: AsyncSession) -> int:
    """Get total number of active documents"""
    try:
        result = await db.execute(
            select(func.count(Document.id)).where(Document.is_active == True)
        )
        return result.scalar() or 0
    except Exception:
        return 0

async def seed_initial_knowledge(db: AsyncSession = None):
    """Seed initial knowledge base if empty"""
    try:
        if not db:
            async with db_config.get_session() as session:
                db = session

        # Check if we already have documents
        count = await get_document_count(db)
        if count > 0:
            logger.info(f"Database already has {count} documents, skipping seed")
            return

        logger.info("Seeding initial knowledge base...")

        # Add initial documents
        documents_added = 0
        for key, content in INITIAL_KNOWLEDGE.items():
            try:
                # Create database record
                doc = Document(
                    title=key.replace("_", " ").title(),
                    content=content,
                    content_type="text",
                    metadata={"source": "initial_seed", "category": "general"}
                )
                db.add(doc)
                await db.flush()  # Get the ID

                # Add to vector store
                await vector_search.add_document(
                    str(doc.id),
                    content,
                    {"title": doc.title, "source": "initial_seed"}
                )

                documents_added += 1
                logger.info(f"Added document: {key}")

            except Exception as e:
                logger.error(f"Failed to add document {key}: {e}")

        await db.commit()
        logger.info(f"Seeded {documents_added} initial documents")

    except Exception as e:
        logger.error(f"Failed to seed initial knowledge: {e}")

@app.get("/documents")
async def list_documents(
    limit: int = 20,
    offset: int = 0,
    include_content: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """List documents with pagination and filtering"""
    try:
        # Get total count
        total_count = await get_document_count(db)

        # Get documents with pagination
        query = select(Document).where(Document.is_active == True)
        query = query.offset(offset).limit(limit).order_by(Document.created_at.desc())

        result = await db.execute(query)
        documents = result.scalars().all()

        # Format response
        doc_list = []
        for doc in documents:
            doc_data = {
                "id": str(doc.id),
                "title": doc.title,
                "content_type": doc.content_type,
                "source_url": doc.source_url,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
                "metadata": doc.metadata
            }

            if include_content:
                doc_data["content"] = doc.content
            else:
                # Provide preview
                preview_length = 200
                doc_data["preview"] = (
                    doc.content[:preview_length] + "..."
                    if len(doc.content) > preview_length
                    else doc.content
                )
                doc_data["content_length"] = len(doc.content)

            doc_list.append(doc_data)

        return {
            "documents": doc_list,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.post("/documents", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a new document with vector embedding"""
    try:
        # Create database record
        doc = Document(
            title=document.title,
            content=document.content,
            content_type=document.content_type,
            source_url=document.source_url,
            metadata=document.metadata
        )
        db.add(doc)
        await db.flush()  # Get the ID before commit

        # Add to vector store in background
        background_tasks.add_task(
            add_to_vector_store,
            str(doc.id),
            document.content,
            {"title": document.title, **document.metadata}
        )

        await db.commit()
        await db.refresh(doc)

        logger.info(f"Created document: {doc.title} (ID: {doc.id})")

        return DocumentResponse(
            id=str(doc.id),
            title=doc.title,
            content=doc.content,
            content_type=doc.content_type,
            source_url=doc.source_url,
            metadata=doc.metadata,
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )

    except Exception as e:
        logger.error(f"Error creating document: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create document")

@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific document by ID"""
    try:
        doc_uuid = uuid.UUID(doc_id)
        query = select(Document).where(
            and_(Document.id == doc_uuid, Document.is_active == True)
        )
        result = await db.execute(query)
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            id=str(document.id),
            title=document.title,
            content=document.content,
            content_type=document.content_type,
            source_url=document.source_url,
            metadata=document.metadata,
            created_at=document.created_at,
            updated_at=document.updated_at
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Error retrieving document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@app.put("/documents/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: str,
    update: DocumentUpdateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Update an existing document"""
    try:
        doc_uuid = uuid.UUID(doc_id)
        query = select(Document).where(
            and_(Document.id == doc_uuid, Document.is_active == True)
        )
        result = await db.execute(query)
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Track if content changed for vector update
        content_changed = False

        # Update fields
        if update.title is not None:
            document.title = update.title
        if update.content is not None:
            document.content = update.content
            content_changed = True
        if update.content_type is not None:
            document.content_type = update.content_type
        if update.source_url is not None:
            document.source_url = update.source_url
        if update.metadata is not None:
            document.metadata = update.metadata

        document.updated_at = datetime.utcnow()

        # Update vector store if content changed
        if content_changed:
            background_tasks.add_task(
                update_vector_store,
                str(document.id),
                document.content,
                {"title": document.title, **document.metadata}
            )

        await db.commit()
        await db.refresh(document)

        logger.info(f"Updated document: {document.title} (ID: {document.id})")

        return DocumentResponse(
            id=str(document.id),
            title=document.title,
            content=document.content,
            content_type=document.content_type,
            source_url=document.source_url,
            metadata=document.metadata,
            created_at=document.created_at,
            updated_at=document.updated_at
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Error updating document {doc_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update document")

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Soft delete a document"""
    try:
        doc_uuid = uuid.UUID(doc_id)
        query = select(Document).where(
            and_(Document.id == doc_uuid, Document.is_active == True)
        )
        result = await db.execute(query)
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Soft delete
        document.is_active = False
        document.updated_at = datetime.utcnow()

        # Remove from vector store
        background_tasks.add_task(remove_from_vector_store, str(document.id))

        await db.commit()

        logger.info(f"Deleted document: {document.title} (ID: {document.id})")

        return {
            "message": f"Document {doc_id} deleted successfully",
            "document_id": doc_id,
            "timestamp": datetime.now().isoformat()
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/documents/bulk")
async def bulk_upload_documents(
    upload: BulkUploadRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Bulk upload multiple documents"""
    try:
        created_docs = []
        failed_docs = []

        for doc_request in upload.documents:
            try:
                # Create database record
                doc = Document(
                    title=doc_request.title,
                    content=doc_request.content,
                    content_type=doc_request.content_type,
                    source_url=doc_request.source_url,
                    metadata=doc_request.metadata
                )
                db.add(doc)
                await db.flush()

                created_docs.append(str(doc.id))

                # Add to vector store in background
                background_tasks.add_task(
                    add_to_vector_store,
                    str(doc.id),
                    doc_request.content,
                    {"title": doc_request.title, **doc_request.metadata}
                )

            except Exception as e:
                failed_docs.append({"title": doc_request.title, "error": str(e)})

        await db.commit()

        logger.info(f"Bulk upload: {len(created_docs)} created, {len(failed_docs)} failed")

        return {
            "message": "Bulk upload completed",
            "created_count": len(created_docs),
            "failed_count": len(failed_docs),
            "created_ids": created_docs,
            "failed_documents": failed_docs,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Bulk upload error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Bulk upload failed")

@app.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get comprehensive statistics about the knowledge base"""
    try:
        # Database stats
        total_docs = await get_document_count(db)

        # Recent activity
        recent_query = select(func.count(Document.id)).where(
            and_(
                Document.is_active == True,
                Document.created_at >= datetime.utcnow() - timedelta(days=7)
            )
        )
        recent_result = await db.execute(recent_query)
        recent_docs = recent_result.scalar() or 0

        # Vector search stats
        vector_stats = await vector_search.get_collection_stats()

        # Search query stats
        search_stats_query = select(
            func.count(SearchQuery.id),
            func.avg(SearchQuery.processing_time),
            func.sum(SearchQuery.result_count)
        ).where(SearchQuery.created_at >= datetime.utcnow() - timedelta(days=1))

        search_result = await db.execute(search_stats_query)
        search_count, avg_time, total_results = search_result.first()

        return {
            "knowledge_base": {
                "total_documents": total_docs,
                "recent_documents": recent_docs,
                "vector_documents": vector_stats.get("document_count", 0)
            },
            "search_analytics": {
                "queries_24h": search_count or 0,
                "avg_response_time": float(avg_time) if avg_time else 0.0,
                "total_results_returned": total_results or 0
            },
            "vector_search": vector_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.post("/reindex")
async def reindex_vector_store(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Reindex all documents in the vector store"""
    try:
        # Get all active documents
        query = select(Document).where(Document.is_active == True)
        result = await db.execute(query)
        documents = result.scalars().all()

        # Start reindexing in background
        background_tasks.add_task(reindex_all_documents, documents)

        return {
            "message": "Reindexing started",
            "document_count": len(documents),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start reindexing")

# Background task functions
async def add_to_vector_store(doc_id: str, content: str, metadata: Dict[str, Any]):
    """Background task to add document to vector store"""
    try:
        await vector_search.add_document(doc_id, content, metadata)
        logger.info(f"Added document {doc_id} to vector store")
    except Exception as e:
        logger.error(f"Failed to add document {doc_id} to vector store: {e}")

async def update_vector_store(doc_id: str, content: str, metadata: Dict[str, Any]):
    """Background task to update document in vector store"""
    try:
        await vector_search.update_document(doc_id, content, metadata)
        logger.info(f"Updated document {doc_id} in vector store")
    except Exception as e:
        logger.error(f"Failed to update document {doc_id} in vector store: {e}")

async def remove_from_vector_store(doc_id: str):
    """Background task to remove document from vector store"""
    try:
        await vector_search.delete_document(doc_id)
        logger.info(f"Removed document {doc_id} from vector store")
    except Exception as e:
        logger.error(f"Failed to remove document {doc_id} from vector store: {e}")

async def reindex_all_documents(documents: List[Document]):
    """Background task to reindex all documents"""
    try:
        logger.info(f"Starting reindex of {len(documents)} documents")
        batch_docs = []

        for doc in documents:
            batch_docs.append({
                "id": str(doc.id),
                "content": doc.content,
                "metadata": {"title": doc.title, **doc.metadata}
            })

        result = await vector_search.batch_add_documents(batch_docs)
        logger.info(f"Reindex completed: {result}")

    except Exception as e:
        logger.error(f"Reindex failed: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )