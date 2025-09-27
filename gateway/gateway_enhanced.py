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
async def smart_assistant(request: AssistantRequest):
    """Smart assistant endpoint with intent-based routing"""
    message = request.message.lower()

    # Simple intent detection
    intent = "general"
    if any(word in message for word in ["search", "find", "lookup", "know"]):
        intent = "search"
    elif any(word in message for word in ["task", "todo", "schedule", "reminder"]):
        intent = "personal"
    elif any(word in message for word in ["explain", "what is", "define"]):
        intent = "knowledge"

    response = {
        "intent": intent,
        "message": request.message,
        "response": "",
        "data": None,
        "timestamp": datetime.now().isoformat()
    }

    try:
        if intent == "search" or intent == "knowledge":
            # Route to RAG
            rag_response = await health_checker.make_request(
                f"{RAG_SERVICE_URL}/search",
                "POST",
                {"query": request.message, "max_results": 3}
            )
            if "error" not in rag_response:
                response["data"] = rag_response
                response["response"] = f"Found {len(rag_response.get('results', []))} relevant results"

        elif intent == "personal":
            # Route to Personal module
            personal_response = await health_checker.make_request(f"{PERSONAL_SERVICE_URL}/tasks")
            if "error" not in personal_response:
                response["data"] = personal_response
                response["response"] = f"You have {len(personal_response.get('tasks', []))} tasks"

        else:
            response["response"] = "I can help you search for information or manage your tasks. What would you like to do?"

    except Exception as e:
        response["response"] = f"Error processing request: {str(e)}"

    return response

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