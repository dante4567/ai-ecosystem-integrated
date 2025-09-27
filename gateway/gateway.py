#!/usr/bin/env python3

import os
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Ecosystem Gateway",
    description="Integration gateway for RAG and Personal modules",
    version="1.0.0"
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
                    return {"status": "healthy", "response_time": data.get("response_time", "unknown")}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except asyncio.TimeoutError:
            return {"status": "unhealthy", "error": "timeout"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self):
        if self.session:
            await self.session.close()


health_checker = ServiceHealthChecker()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Ecosystem Gateway")
    logger.info(f"RAG Service URL: {RAG_SERVICE_URL}")
    logger.info(f"Personal Service URL: {PERSONAL_SERVICE_URL}")


@app.on_event("shutdown")
async def shutdown_event():
    await health_checker.close()
    logger.info("Shutting down AI Ecosystem Gateway")


@app.get("/health")
async def health_check():
    """Health check endpoint for the gateway"""
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
        "version": "1.0.0",
        "dependencies": {
            "rag_module": rag_health,
            "personal_module": personal_health
        }
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "AI Ecosystem Gateway",
        "version": "1.0.0",
        "description": "Integration gateway for RAG and Personal modules",
        "endpoints": {
            "health": "/health",
            "rag": "/rag/*",
            "personal": "/personal/*"
        }
    }


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

        # Prepare request data
        url = f"{service_url}/{path}"
        headers = dict(request.headers)

        # Remove hop-by-hop headers
        headers.pop("host", None)
        headers.pop("content-length", None)

        # Get request body
        body = await request.body()

        # Forward the request
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
        "gateway:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )