"""
OpenWebUI Functions for RAG Service Only
Configure OpenWebUI to use just the RAG service for document search and chat
"""

import json
import requests
from typing import List, Dict, Any

# Configuration
RAG_SERVICE_URL = "http://localhost:8001"

def search_documents(query: str, max_results: int = 5) -> str:
    """
    Search documents in the RAG knowledge base.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results
    """
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/search",
            json={"text": query, "top_k": max_results},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return f"No documents found for query: {query}"

            formatted_results = f"📄 **Search Results for '{query}':**\n\n"

            for i, result in enumerate(results, 1):
                content = result.get("content", "No content")[:200]
                score = result.get("relevance_score", 0)
                title = result.get("metadata", {}).get("title", "Untitled")

                formatted_results += f"**{i}. {title}** (Score: {score:.3f})\n"
                formatted_results += f"{content}...\n\n"

            return formatted_results
        else:
            return f"❌ Search failed with status {response.status_code}"

    except Exception as e:
        return f"❌ Search error: {str(e)}"

def chat_with_documents(question: str, model: str = "groq/llama-3.1-8b-instant") -> str:
    """
    Chat with your documents using RAG.

    Args:
        question: Question to ask about your documents
        model: LLM model to use (groq/llama-3.1-8b-instant, anthropic/claude-3-haiku-20240307, etc.)

    Returns:
        AI response based on document context
    """
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/chat",
            json={"question": question, "llm_model": model},
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer generated")
            sources = data.get("sources", [])
            model_used = data.get("model_used", model)

            result = f"🤖 **AI Response** (via {model_used}):\n\n{answer}\n\n"

            if sources:
                result += f"📚 **Sources:** {len(sources)} documents used\n"
                for i, source in enumerate(sources[:3], 1):
                    title = source.get("title", "Unknown")
                    result += f"{i}. {title}\n"

            return result
        else:
            return f"❌ Chat failed with status {response.status_code}"

    except Exception as e:
        return f"❌ Chat error: {str(e)}"

def upload_document_text(title: str, content: str) -> str:
    """
    Upload text content as a document to the RAG service.

    Args:
        title: Document title
        content: Document content

    Returns:
        Upload result message
    """
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/ingest",
            json={"content": content, "title": title},
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            doc_id = data.get("doc_id", "unknown")
            chunks = data.get("chunks", 0)

            return f"✅ Successfully uploaded '{title}'\n📄 Document ID: {doc_id}\n🔢 Chunks created: {chunks}"
        else:
            return f"❌ Upload failed with status {response.status_code}"

    except Exception as e:
        return f"❌ Upload error: {str(e)}"

def get_rag_stats() -> str:
    """
    Get RAG service statistics and health information.

    Returns:
        Service statistics
    """
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()

            result = "📊 **RAG Service Status:**\n\n"
            result += f"✅ Status: {data.get('status', 'unknown')}\n"
            result += f"🐳 Docker: {data.get('docker', False)}\n"
            result += f"🗄️ ChromaDB: {data.get('chromadb', 'unknown')}\n"
            result += f"👁️ File Watcher: {data.get('file_watcher', 'unknown')}\n"
            result += f"📷 OCR Available: {data.get('ocr_available', False)}\n\n"

            llm_providers = data.get('llm_providers', {})
            result += "🤖 **LLM Providers:**\n"
            for provider, status in llm_providers.items():
                emoji = "✅" if status else "❌"
                result += f"{emoji} {provider.title()}\n"

            return result
        else:
            return f"❌ Health check failed with status {response.status_code}"

    except Exception as e:
        return f"❌ Health check error: {str(e)}"

# Available LLM models for chat
AVAILABLE_MODELS = [
    "groq/llama-3.1-8b-instant",
    "groq/llama3-70b-8192",
    "anthropic/claude-3-haiku-20240307",
    "anthropic/claude-3-5-sonnet-20241022",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "google/gemini-1.5-pro"
]

def list_available_models() -> str:
    """
    List available LLM models for chat.

    Returns:
        List of available models
    """
    result = "🤖 **Available LLM Models:**\n\n"

    for model in AVAILABLE_MODELS:
        provider, model_name = model.split("/", 1)
        result += f"• **{provider.title()}**: {model_name}\n"

    result += "\nUse these model names in the `chat_with_documents()` function."

    return result

# OpenWebUI Function Registration
def get_openwebui_functions():
    """Return OpenWebUI function definitions"""
    return {
        "search_documents": {
            "function": search_documents,
            "description": "Search documents in RAG knowledge base",
            "parameters": {
                "query": {"type": "str", "description": "Search query"},
                "max_results": {"type": "int", "description": "Max results (default: 5)"}
            }
        },
        "chat_with_documents": {
            "function": chat_with_documents,
            "description": "Chat with documents using AI",
            "parameters": {
                "question": {"type": "str", "description": "Question about documents"},
                "model": {"type": "str", "description": "LLM model to use"}
            }
        },
        "upload_document_text": {
            "function": upload_document_text,
            "description": "Upload text content as document",
            "parameters": {
                "title": {"type": "str", "description": "Document title"},
                "content": {"type": "str", "description": "Document content"}
            }
        },
        "get_rag_stats": {
            "function": get_rag_stats,
            "description": "Get RAG service statistics",
            "parameters": {}
        },
        "list_available_models": {
            "function": list_available_models,
            "description": "List available LLM models",
            "parameters": {}
        }
    }