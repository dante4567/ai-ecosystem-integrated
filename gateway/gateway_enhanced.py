#!/usr/bin/env python3

import os
import asyncio
import aiohttp
import json
import sys
sys.path.append('../shared-config')

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from smart_assistant import get_smart_assistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Ecosystem Gateway",
    description="Enhanced integration gateway with OpenWebUI functions",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-module:8001")
PERSONAL_SERVICE_URL = os.getenv("PERSONAL_SERVICE_URL", "http://personal-module:8002")
PORT = int(os.getenv("PORT", 8003))

# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    include_rag: bool = True
    include_personal: bool = True
    max_results: int = 10

class AssistantRequest(BaseModel):
    message: str
    context: Optional[Dict] = None
    user_id: Optional[str] = None

class OpenWebUIFunction(BaseModel):
    name: str
    description: str
    parameters: Dict
    function: str

class ServiceHealthChecker:
    def __init__(self):
        self.session = None

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def check_service_health(self, url: str) -> Dict[str, Any]:
        try:
            session = await self.get_session()
            async with session.get(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "healthy", "service": data.get("service", "unknown"), "response_time": data.get("response_time", "unknown")}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except asyncio.TimeoutError:
            return {"status": "unhealthy", "error": "timeout"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def make_request(self, url: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        try:
            session = await self.get_session()
            kwargs = {"timeout": 10}
            if data:
                kwargs["json"] = data

            async with session.request(method, url, **kwargs) as response:
                if response.content_type == "application/json":
                    return await response.json()
                else:
                    return {"data": await response.text(), "status_code": response.status}
        except Exception as e:
            return {"error": str(e), "status_code": 500}

    async def close(self):
        if self.session:
            await self.session.close()

health_checker = ServiceHealthChecker()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Ecosystem Gateway v2.0")
    logger.info(f"RAG Service URL: {RAG_SERVICE_URL}")
    logger.info(f"Personal Service URL: {PERSONAL_SERVICE_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    await health_checker.close()
    logger.info("Shutting down AI Ecosystem Gateway")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    start_time = datetime.now()

    # Check downstream services
    rag_health = await health_checker.check_service_health(RAG_SERVICE_URL)
    personal_health = await health_checker.check_service_health(PERSONAL_SERVICE_URL)

    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()

    gateway_status = "healthy"
    if rag_health["status"] != "healthy" or personal_health["status"] != "healthy":
        gateway_status = "degraded"

    return {
        "service": "AI Ecosystem Gateway",
        "status": gateway_status,
        "timestamp": datetime.now().isoformat(),
        "response_time": f"{response_time:.3f}s",
        "version": "2.0.0",
        "dependencies": {
            "rag_module": rag_health,
            "personal_module": personal_health
        }
    }

@app.get("/")
async def root():
    """Enhanced root endpoint with all available endpoints"""
    return {
        "message": "AI Ecosystem Gateway v2.0",
        "version": "2.0.0",
        "description": "Enhanced integration gateway with OpenWebUI functions",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "search": "/search",
            "assistant": "/assistant",
            "chat": "/chat",
            "capabilities": "/capabilities",
            "rag": "/rag/*",
            "personal": "/personal/*",
            "functions": "/functions",
            "openwebui": "/openwebui/*"
        }
    }

@app.get("/status")
async def service_status():
    """Comprehensive service status endpoint"""
    rag_health = await health_checker.check_service_health(RAG_SERVICE_URL)
    personal_health = await health_checker.check_service_health(PERSONAL_SERVICE_URL)

    # Get additional service info
    rag_info = await health_checker.make_request(f"{RAG_SERVICE_URL}/documents")
    personal_info = await health_checker.make_request(f"{PERSONAL_SERVICE_URL}/stats")

    return {
        "gateway": {
            "status": "healthy",
            "version": "2.0.0",
            "uptime": datetime.now().isoformat()
        },
        "services": {
            "rag_module": {
                "health": rag_health,
                "url": RAG_SERVICE_URL,
                "info": rag_info if "error" not in rag_info else None
            },
            "personal_module": {
                "health": personal_health,
                "url": PERSONAL_SERVICE_URL,
                "info": personal_info if "error" not in personal_info else None
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/search")
async def unified_search(request: SearchRequest):
    """Unified search endpoint calling both RAG and personal services"""
    start_time = datetime.now()
    results = {"rag": None, "personal": None, "combined": []}

    tasks = []

    # Search RAG module
    if request.include_rag:
        rag_search = health_checker.make_request(
            f"{RAG_SERVICE_URL}/search",
            "POST",
            {"query": request.query, "max_results": request.max_results // 2}
        )
        tasks.append(("rag", rag_search))

    # Search personal module
    if request.include_personal:
        personal_search = health_checker.make_request(f"{PERSONAL_SERVICE_URL}/tasks")
        tasks.append(("personal", personal_search))

    # Execute searches in parallel
    if tasks:
        task_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

        for i, (service_name, _) in enumerate(tasks):
            result = task_results[i]
            if not isinstance(result, Exception) and "error" not in result:
                results[service_name] = result

    # Combine results
    combined_results = []

    if results["rag"] and "results" in results["rag"]:
        for item in results["rag"]["results"]:
            combined_results.append({
                "source": "rag",
                "type": "knowledge",
                "content": item.get("content", ""),
                "relevance": item.get("relevance_score", 0)
            })

    if results["personal"] and "tasks" in results["personal"]:
        for task in results["personal"]["tasks"]:
            if request.query.lower() in task.get("title", "").lower():
                combined_results.append({
                    "source": "personal",
                    "type": "task",
                    "content": task.get("title", ""),
                    "relevance": 0.8,
                    "completed": task.get("completed", False)
                })

    # Sort by relevance
    combined_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    results["combined"] = combined_results[:request.max_results]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return {
        "query": request.query,
        "results": results,
        "total_results": len(combined_results),
        "processing_time": f"{processing_time:.3f}s",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/assistant")
async def smart_assistant_endpoint(request: AssistantRequest):
    """Enhanced smart assistant endpoint with advanced intent recognition"""
    try:
        # Get smart assistant instance
        assistant = get_smart_assistant(RAG_SERVICE_URL, PERSONAL_SERVICE_URL)

        # Process the message
        result = await assistant.process_message(
            request.message,
            context=request.context,
            health_checker=health_checker
        )

        # Format response
        response = {
            "message": request.message,
            "intent": {
                "name": result.intent.name,
                "confidence": result.intent.confidence,
                "action": result.intent.action,
                "entities": result.intent.entities
            },
            "response": result.response_text,
            "actions": result.actions,
            "data": result.data,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp,
            "user_id": request.user_id or "default_user"
        }

        logger.info(f"Assistant processed: {request.message} -> {result.intent.name} ({result.intent.confidence:.2f})")
        return response

    except Exception as e:
        logger.error(f"Smart assistant error: {e}")
        return {
            "message": request.message,
            "intent": {"name": "error", "confidence": 0.0, "action": "error", "entities": {}},
            "response": f"I encountered an error processing your request: {str(e)}",
            "actions": [],
            "data": None,
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id or "default_user"
        }

@app.get("/functions")
async def get_openwebui_functions():
    """Get available OpenWebUI functions"""
    functions = [
        {
            "name": "search_knowledge",
            "description": "Search the RAG knowledge base for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5}
            }
        },
        {
            "name": "get_tasks",
            "description": "Get personal tasks and schedule",
            "parameters": {
                "completed": {"type": "boolean", "default": False}
            }
        },
        {
            "name": "create_task",
            "description": "Create a new personal task",
            "parameters": {
                "title": {"type": "string", "description": "Task title"},
                "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD)"}
            }
        },
        {
            "name": "unified_search",
            "description": "Search both knowledge base and personal data",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "include_rag": {"type": "boolean", "default": True},
                "include_personal": {"type": "boolean", "default": True}
            }
        }
    ]

    return {"functions": functions, "total": len(functions)}

@app.post("/chat")
async def chat_with_assistant(request: AssistantRequest):
    """Chat interface for natural conversation with the assistant"""
    try:
        # Use the smart assistant for chat
        assistant = get_smart_assistant(RAG_SERVICE_URL, PERSONAL_SERVICE_URL)
        result = await assistant.process_message(
            request.message,
            context=request.context,
            health_checker=health_checker
        )

        # Return a chat-optimized response
        return {
            "response": result.response_text,
            "intent": result.intent.name,
            "confidence": result.intent.confidence,
            "actions_taken": len(result.actions),
            "timestamp": result.timestamp
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "response": "I'm having trouble understanding that. Could you try rephrasing your question?",
            "intent": "error",
            "confidence": 0.0,
            "actions_taken": 0,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/capabilities")
async def get_capabilities():
    """Get assistant capabilities and example commands"""
    return {
        "capabilities": {
            "knowledge_search": {
                "description": "Search and retrieve information from the knowledge base",
                "examples": [
                    "Search for artificial intelligence",
                    "What is machine learning?",
                    "Find information about Docker",
                    "Tell me about microservices"
                ]
            },
            "task_management": {
                "description": "Create, manage, and track personal tasks",
                "examples": [
                    "Create a task to review the AI paper",
                    "Show my pending tasks",
                    "Add a high priority task due tomorrow",
                    "What tasks are due today?"
                ]
            },
            "productivity_tracking": {
                "description": "View productivity statistics and progress",
                "examples": [
                    "Show my productivity stats",
                    "How many tasks did I complete this week?",
                    "What's my focus score?"
                ]
            },
            "schedule_management": {
                "description": "View and manage your schedule and calendar",
                "examples": [
                    "What's my schedule today?",
                    "Show me tomorrow's agenda",
                    "Any meetings scheduled?"
                ]
            },
            "integration_services": {
                "description": "Sync with external services like Todoist",
                "examples": [
                    "Sync with Todoist",
                    "Import tasks from external services"
                ]
            },
            "natural_conversation": {
                "description": "Engage in helpful conversation and provide assistance",
                "examples": [
                    "Help me get organized",
                    "What can you do?",
                    "How can I be more productive?"
                ]
            }
        },
        "features": [
            "Advanced intent recognition using pattern matching and LLM assistance",
            "Entity extraction from natural language",
            "Context-aware responses",
            "Multi-service integration",
            "Productivity analytics",
            "Task automation",
            "Knowledge base search with vector similarity"
        ],
        "supported_integrations": [
            "Todoist",
            "OpenWebUI",
            "Custom APIs"
        ]
    }

# Original proxy endpoints
@app.api_route("/rag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_rag(path: str, request: Request):
    """Proxy requests to RAG module"""
    return await proxy_request(RAG_SERVICE_URL, path, request)

@app.api_route("/personal/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_personal(path: str, request: Request):
    """Proxy requests to Personal module"""
    return await proxy_request(PERSONAL_SERVICE_URL, path, request)

async def proxy_request(service_url: str, path: str, request: Request):
    """Generic proxy function for forwarding requests to backend services"""
    try:
        session = await health_checker.get_session()
        url = f"{service_url}/{path}"
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        body = await request.body()

        async with session.request(
            method=request.method,
            url=url,
            headers=headers,
            data=body,
            params=request.query_params
        ) as response:
            content = await response.read()
            return JSONResponse(
                content=await response.json() if response.content_type == "application/json" else {"data": content.decode()},
                status_code=response.status,
                headers=dict(response.headers)
            )

    except aiohttp.ClientError as e:
        logger.error(f"Error proxying request to {service_url}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "gateway_enhanced:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )