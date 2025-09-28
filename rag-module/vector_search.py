#!/usr/bin/env python3

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class VectorSearchService:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.collection_name = "documents"

    async def initialize(self):
        """Initialize the vector search service"""
        try:
            # Initialize embedding model in thread pool
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                self.executor,
                self._load_embedding_model
            )

            # Initialize ChromaDB
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))

            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

            logger.info("Vector search service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vector search service: {e}")
            return False

    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        return SentenceTransformer(self.model_name)

    async def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document to the vector store"""
        try:
            # Generate embedding
            embedding = await self._generate_embedding(content)

            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "content_length": len(content),
                "added_at": str(asyncio.get_event_loop().time())
            })

            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )

            logger.info(f"Added document {doc_id} to vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False

    async def search_similar(self, query: str, n_results: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )

            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
                distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
                ids = results["ids"][0] if results["ids"] else [f"doc_{i}" for i in range(len(documents))]

                for i, (doc, metadata, distance, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                    # Convert distance to similarity score (cosine distance to similarity)
                    similarity_score = 1.0 - distance

                    if similarity_score >= threshold:
                        formatted_results.append({
                            "id": doc_id,
                            "content": doc,
                            "similarity_score": similarity_score,
                            "relevance_score": similarity_score,  # For compatibility
                            "metadata": metadata,
                            "rank": i + 1
                        })

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Update an existing document"""
        try:
            # Delete existing document
            await self.delete_document(doc_id)

            # Add updated document
            return await self.add_document(doc_id, content, metadata)

        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id} from vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def get_document_count(self) -> int:
        """Get the total number of documents in the store"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            self.embedding_model.encode,
            text
        )

        return embedding

    async def batch_add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple documents in batch"""
        try:
            embeddings = []
            doc_contents = []
            doc_metadatas = []
            doc_ids = []

            # Generate embeddings for all documents
            for doc in documents:
                doc_id = doc.get("id")
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                if not doc_id or not content:
                    continue

                embedding = await self._generate_embedding(content)

                embeddings.append(embedding.tolist())
                doc_contents.append(content)
                doc_metadatas.append(metadata)
                doc_ids.append(doc_id)

            # Add to ChromaDB in batch
            if embeddings:
                self.collection.add(
                    embeddings=embeddings,
                    documents=doc_contents,
                    metadatas=doc_metadatas,
                    ids=doc_ids
                )

            return {
                "success": True,
                "added_count": len(embeddings),
                "skipped_count": len(documents) - len(embeddings)
            }

        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "added_count": 0,
                "skipped_count": len(documents)
            }

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = await self.get_document_count()

            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "embedding_model": self.model_name,
                "status": "healthy" if self.embedding_model and self.collection else "unhealthy"
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "document_count": 0,
                "collection_name": self.collection_name,
                "embedding_model": self.model_name,
                "status": "unhealthy",
                "error": str(e)
            }

    async def close(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Vector search service closed")

# Global vector search instance
vector_search = VectorSearchService()