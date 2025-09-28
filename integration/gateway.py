#!/usr/bin/env python3

import os
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
from dateutil import parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Ecosystem Gateway",
    description="Unified gateway orchestrating RAG and Personal services",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")
PERSONAL_SERVICE_URL = os.getenv("PERSONAL_SERVICE_URL", "http://personal-service:8002")
CHROMA_URL = os.getenv("CHROMA_URL", "http://chromadb:8000")
PORT = int(os.getenv("PORT", 8003))

# Request/Response Models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)
    include_personal: bool = True
    include_context: bool = True

class SearchResponse(BaseModel):
    query: str
    rag_results: List[Dict[str, Any]]
    personal_context: List[Dict[str, Any]]
    unified_results: List[Dict[str, Any]]
    processing_time: float
    timestamp: str

class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None
    include_actions: bool = True

class AssistantResponse(BaseModel):
    message: str
    response: str
    intent: str
    confidence: float
    actions_taken: List[Dict[str, Any]]
    data_sources: List[str]
    processing_time: float

class ScheduleRequest(BaseModel):
    query: str = Field(..., min_length=1)
    schedule_time: Optional[datetime] = None
    duration_minutes: int = Field(default=30, ge=15, le=480)
    priority: str = Field(default="medium", regex="^(low|medium|high|urgent)$")

class ServiceHealthChecker:
    def __init__(self):
        self.session = None

    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session

    async def make_request(self, url: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        try:
            session = await self.get_session()
            kwargs = {}
            if data:
                kwargs["json"] = data

            async with session.request(method, url, **kwargs) as response:
                if response.content_type == "application/json":
                    return await response.json()
                else:
                    text = await response.text()
                    return {"data": text, "status_code": response.status}
        except Exception as e:
            logger.error(f"Request failed to {url}: {e}")
            return {"error": str(e), "status_code": 500}

    async def check_service_health(self, url: str) -> Dict[str, Any]:
        try:
            result = await self.make_request(f"{url}/health")
            if "error" not in result:
                return {"status": "healthy", "details": result}
            else:
                return {"status": "unhealthy", "error": result["error"]}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self):
        if self.session:
            await self.session.close()

health_checker = ServiceHealthChecker()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Ecosystem Gateway")
    logger.info(f"RAG Service: {RAG_SERVICE_URL}")
    logger.info(f"Personal Service: {PERSONAL_SERVICE_URL}")
    logger.info(f"ChromaDB: {CHROMA_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    await health_checker.close()
    logger.info("Gateway shutdown complete")

@app.get("/health")
async def health_check():
    """Gateway health check with downstream service status"""
    start_time = datetime.now()

    # Check downstream services
    rag_health = await health_checker.check_service_health(RAG_SERVICE_URL)
    personal_health = await health_checker.check_service_health(PERSONAL_SERVICE_URL)
    chroma_health = await health_checker.check_service_health(f"{CHROMA_URL}/api/v1/heartbeat")

    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()

    overall_status = "healthy"
    if any(service["status"] != "healthy" for service in [rag_health, personal_health, chroma_health]):
        overall_status = "degraded"

    return {
        "service": "AI Ecosystem Gateway",
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "response_time": f"{response_time:.3f}s",
        "services": {
            "rag": rag_health,
            "personal": personal_health,
            "chromadb": chroma_health
        }
    }

@app.get("/")
async def root():
    """Gateway information and available endpoints"""
    return {
        "service": "AI Ecosystem Gateway",
        "version": "1.0.0",
        "description": "Unified gateway orchestrating RAG and Personal services",
        "endpoints": {
            "health": "/health",
            "search": "/api/search",
            "assistant": "/api/assistant",
            "smart_schedule": "/api/smart-schedule",
            "services": "/api/services",
            "functions": "/api/functions"
        },
        "services": {
            "rag": RAG_SERVICE_URL,
            "personal": PERSONAL_SERVICE_URL,
            "chromadb": CHROMA_URL
        }
    }

@app.post("/api/search", response_model=SearchResponse)
async def unified_search(request: SearchRequest):
    """Search RAG then enrich with personal context"""
    start_time = datetime.now()

    try:
        # Search RAG service
        rag_results = []
        rag_response = await health_checker.make_request(
            f"{RAG_SERVICE_URL}/search",
            "POST",
            {
                "query": request.query,
                "max_results": request.max_results,
                "use_vector_search": True
            }
        )

        if "error" not in rag_response and "results" in rag_response:
            rag_results = rag_response["results"]

        # Get personal context if requested
        personal_context = []
        if request.include_personal:
            # Search tasks related to query
            tasks_response = await health_checker.make_request(f"{PERSONAL_SERVICE_URL}/tasks")
            if "error" not in tasks_response and "tasks" in tasks_response:
                for task in tasks_response["tasks"][:5]:
                    if request.query.lower() in task.get("title", "").lower():
                        personal_context.append({
                            "type": "task",
                            "id": task.get("id"),
                            "title": task.get("title"),
                            "status": task.get("status"),
                            "relevance": "high" if request.query.lower() in task.get("title", "").lower() else "medium"
                        })

            # Get today's schedule
            schedule_response = await health_checker.make_request(f"{PERSONAL_SERVICE_URL}/schedule")
            if "error" not in schedule_response and "schedule" in schedule_response:
                for date_str, day_info in schedule_response["schedule"].items():
                    for task in day_info.get("tasks", []):
                        if request.query.lower() in task.get("title", "").lower():
                            personal_context.append({
                                "type": "scheduled_task",
                                "title": task.get("title"),
                                "date": date_str,
                                "relevance": "medium"
                            })

        # Create unified results combining both sources
        unified_results = []

        # Add RAG results with enhanced context
        for rag_item in rag_results:
            unified_item = {
                "source": "knowledge_base",
                "id": rag_item.get("id"),
                "title": rag_item.get("title", rag_item.get("id", "Unknown")),
                "content": rag_item.get("content", ""),
                "relevance_score": rag_item.get("similarity_score", rag_item.get("relevance_score", 0)),
                "personal_connections": []
            }

            # Add personal connections if any
            for context_item in personal_context:
                if any(word in context_item.get("title", "").lower()
                      for word in rag_item.get("content", "").lower().split()[:10]):
                    unified_item["personal_connections"].append(context_item)

            unified_results.append(unified_item)

        # Add high-relevance personal items
        for context_item in personal_context:
            if context_item.get("relevance") == "high":
                unified_results.append({
                    "source": "personal",
                    "type": context_item.get("type"),
                    "title": context_item.get("title"),
                    "relevance_score": 0.8,
                    "personal_data": context_item
                })

        # Sort by relevance
        unified_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return SearchResponse(
            query=request.query,
            rag_results=rag_results,
            personal_context=personal_context,
            unified_results=unified_results[:request.max_results],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/assistant", response_model=AssistantResponse)
async def unified_assistant(request: AssistantRequest):
    """Unified assistant using both RAG and Personal modules"""
    start_time = datetime.now()

    try:
        message = request.message.lower()
        intent = "general"
        confidence = 0.6
        actions_taken = []
        data_sources = []
        response_text = ""

        # Intent detection
        if any(word in message for word in ["search", "find", "lookup", "what is", "explain", "tell me"]):
            intent = "search"
            confidence = 0.9

            # Search knowledge base
            search_response = await health_checker.make_request(
                f"{RAG_SERVICE_URL}/search",
                "POST",
                {"query": request.message, "max_results": 3}
            )

            if "error" not in search_response and search_response.get("results"):
                results = search_response["results"]
                data_sources.append("knowledge_base")
                actions_taken.append({"action": "knowledge_search", "results_count": len(results)})

                response_text = f"I found {len(results)} relevant results:\n\n"
                for i, result in enumerate(results, 1):
                    title = result.get("title", result.get("id", "Unknown"))
                    content = result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
                    response_text += f"{i}. **{title}**\n{content}\n\n"
            else:
                response_text = f"I couldn't find specific information about '{request.message}'. Could you try rephrasing your question?"

        elif any(word in message for word in ["task", "todo", "schedule", "reminder", "calendar"]):
            intent = "task_management"
            confidence = 0.9

            if any(word in message for word in ["create", "add", "make", "new"]):
                # Extract task title
                task_title = request.message
                for word in ["create", "add", "make", "new", "task", "todo", "reminder"]:
                    task_title = task_title.replace(word, "").strip()

                if task_title:
                    task_response = await health_checker.make_request(
                        f"{PERSONAL_SERVICE_URL}/tasks",
                        "POST",
                        {"title": task_title, "priority": "medium"}
                    )

                    if "error" not in task_response:
                        data_sources.append("personal")
                        actions_taken.append({"action": "task_created", "task_id": task_response.get("id")})
                        response_text = f"✅ Created task: '{task_title}'"
                    else:
                        response_text = f"❌ Failed to create task: {task_response.get('error', 'Unknown error')}"
                else:
                    response_text = "I need more details about the task you want to create."
            else:
                # Show tasks
                tasks_response = await health_checker.make_request(f"{PERSONAL_SERVICE_URL}/tasks")

                if "error" not in tasks_response and "tasks" in tasks_response:
                    tasks = tasks_response["tasks"]
                    pending_tasks = [t for t in tasks if t.get("status") != "completed"]

                    data_sources.append("personal")
                    actions_taken.append({"action": "tasks_retrieved", "count": len(pending_tasks)})

                    if pending_tasks:
                        response_text = f"📋 You have {len(pending_tasks)} pending tasks:\n\n"
                        for i, task in enumerate(pending_tasks[:5], 1):
                            title = task.get("title", "Untitled")
                            due_date = task.get("due_date", "No due date")
                            response_text += f"{i}. {title} (Due: {due_date})\n"

                        if len(pending_tasks) > 5:
                            response_text += f"\n... and {len(pending_tasks) - 5} more tasks"
                    else:
                        response_text = "🎉 You have no pending tasks!"
                else:
                    response_text = "❌ Unable to retrieve tasks at the moment."

        elif any(word in message for word in ["productivity", "stats", "progress", "summary"]):
            intent = "analytics"
            confidence = 0.8

            stats_response = await health_checker.make_request(f"{PERSONAL_SERVICE_URL}/stats")

            if "error" not in stats_response:
                summary = stats_response.get("summary", {})
                data_sources.append("personal")
                actions_taken.append({"action": "stats_retrieved"})

                response_text = f"📊 **Productivity Summary**\n\n"
                response_text += f"• Total tasks: {summary.get('total_tasks', 0)}\n"
                response_text += f"• Completion rate: {summary.get('completion_rate', 0):.1f}%\n"
                response_text += f"• Recent activity: {summary.get('recent_activity', 0)} tasks this week"
            else:
                response_text = "❌ Unable to retrieve productivity stats at the moment."
        else:
            # General response with service capabilities
            response_text = """I'm your AI assistant! I can help you with:

🔍 **Knowledge Search**: "What is machine learning?" or "Find information about Docker"
📝 **Task Management**: "Create a task to review documents" or "Show my tasks"
📊 **Productivity**: "Show my stats" or "What's my progress?"
📅 **Scheduling**: "What's my schedule today?"

What would you like me to help you with?"""

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return AssistantResponse(
            message=request.message,
            response=response_text,
            intent=intent,
            confidence=confidence,
            actions_taken=actions_taken,
            data_sources=data_sources,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Assistant error: {e}")
        raise HTTPException(status_code=500, detail=f"Assistant request failed: {str(e)}")

@app.post("/api/smart-schedule")
async def smart_schedule(request: ScheduleRequest):
    """Search knowledge then create events based on findings"""
    start_time = datetime.now()

    try:
        # First, search for relevant documents
        search_response = await health_checker.make_request(
            f"{RAG_SERVICE_URL}/search",
            "POST",
            {"query": request.query, "max_results": 5}
        )

        events_created = []

        if "error" not in search_response and search_response.get("results"):
            results = search_response["results"]

            # Schedule time for reviewing each document
            base_time = request.schedule_time or datetime.now() + timedelta(hours=1)

            for i, result in enumerate(results):
                doc_title = result.get("title", result.get("id", "Unknown Document"))

                # Create a task for reviewing this document
                task_title = f"Review: {doc_title}"
                due_time = base_time + timedelta(minutes=i * request.duration_minutes)

                task_response = await health_checker.make_request(
                    f"{PERSONAL_SERVICE_URL}/tasks",
                    "POST",
                    {
                        "title": task_title,
                        "description": f"Review document found for query: {request.query}",
                        "due_date": due_time.isoformat(),
                        "priority": request.priority,
                        "tags": ["review", "research", "scheduled"]
                    }
                )

                if "error" not in task_response:
                    events_created.append({
                        "task_id": task_response.get("id"),
                        "title": task_title,
                        "scheduled_time": due_time.isoformat(),
                        "document_id": result.get("id"),
                        "estimated_duration": request.duration_minutes
                    })

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return {
            "query": request.query,
            "documents_found": len(search_response.get("results", [])),
            "events_created": events_created,
            "total_scheduled_time": len(events_created) * request.duration_minutes,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Smart schedule error: {e}")
        raise HTTPException(status_code=500, detail=f"Smart scheduling failed: {str(e)}")

@app.get("/api/services")
async def service_status():
    """Get status of all connected services"""
    rag_health = await health_checker.check_service_health(RAG_SERVICE_URL)
    personal_health = await health_checker.check_service_health(PERSONAL_SERVICE_URL)
    chroma_health = await health_checker.check_service_health(f"{CHROMA_URL}/api/v1/heartbeat")

    return {
        "services": {
            "rag": {
                "url": RAG_SERVICE_URL,
                "health": rag_health,
                "capabilities": ["document_search", "vector_similarity", "knowledge_retrieval"]
            },
            "personal": {
                "url": PERSONAL_SERVICE_URL,
                "health": personal_health,
                "capabilities": ["task_management", "scheduling", "productivity_tracking"]
            },
            "chromadb": {
                "url": CHROMA_URL,
                "health": chroma_health,
                "capabilities": ["vector_storage", "similarity_search", "embeddings"]
            }
        },
        "gateway": {
            "status": "healthy",
            "capabilities": ["unified_search", "smart_assistant", "cross_service_orchestration"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/functions")
async def get_openwebui_functions():
    """Get OpenWebUI function definitions"""
    from openwebui_functions import OPENWEBUI_FUNCTIONS

    return {
        "functions": OPENWEBUI_FUNCTIONS,
        "total": len(OPENWEBUI_FUNCTIONS),
        "description": "AI Ecosystem integration functions for OpenWebUI",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )