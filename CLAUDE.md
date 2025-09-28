# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker Operations
```bash
# Start all services (primary development workflow)
docker-compose up --build

# Start services in detached mode
docker-compose up -d --build

# Stop all services
docker-compose down

# View logs for all services
docker-compose logs

# View logs for specific service
docker-compose logs [chromadb|rag-service|personal-service|gateway]

# Rebuild specific service
docker-compose up --build [service-name]
```

### Testing and Integration
```bash
# Run comprehensive integration tests
python integration/test_integration.py

# CLI tool for interactive testing
python integration/cli.py status                  # Check all services
python integration/cli.py ask "What is Docker?"   # Test assistant
python integration/cli.py today --include-docs    # Daily briefing
python integration/cli.py search "machine learning" # Test unified search

# Direct API testing
curl http://localhost:8003/health                 # Gateway health
curl http://localhost:8000/api/v1/heartbeat       # ChromaDB
curl http://localhost:8001/health                 # RAG service
curl http://localhost:8002/health                 # Personal service

# Test unified search
curl -X POST http://localhost:8003/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "include_personal": true, "include_context": true}'

# Test smart assistant
curl -X POST http://localhost:8003/api/assistant \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a task to review AI papers due tomorrow"}'
```

## Architecture Overview

This is a microservices-based AI ecosystem with four main components plus sophisticated orchestration:

### Service Architecture
- **ChromaDB** (Port 8000): Vector database for semantic search
- **RAG Service** (Port 8001): Knowledge base with vector search capabilities
- **Personal Service** (Port 8002): Task management with Todoist integration
- **Gateway** (Port 8003): Unified orchestration with intelligent routing
- **Integration Layer** (`integration/`): CLI tools, testing, and OpenWebUI functions

### Key Integration Points
- Services communicate via Docker network (`ai-ecosystem`) with health checks
- Gateway provides unified API with intent recognition and context enhancement
- ChromaDB enables semantic vector search across knowledge base
- Bidirectional Todoist synchronization for task management
- OpenWebUI-compatible functions for chat interface integration
- Comprehensive CLI tool for development and testing

### Service URLs (External Access)
- ChromaDB: `http://localhost:8000`
- RAG Service: `http://localhost:8001`
- Personal Service: `http://localhost:8002`
- Gateway: `http://localhost:8003`

### Internal Docker Network URLs
- ChromaDB: `http://chromadb:8000`
- RAG Service: `http://rag-service:8001`
- Personal Service: `http://personal-service:8002`

## Technology Stack

### Core Framework
- **FastAPI** for all services with Uvicorn ASGI server
- **ChromaDB** for vector database and semantic search
- **Pydantic** for request/response validation with comprehensive models
- **aiohttp** for async HTTP client communication
- **sentence-transformers** for vector embeddings

### Testing & Integration
- **Integration test suite** (`integration/test_integration.py`)
- **CLI tool** (`integration/cli.py`) for interactive development
- **Health checks** with automatic service dependency management
- **OpenWebUI functions** ready for chat interface integration

## Development Workflow

### Adding New Features
1. Identify which service(s) need modification
2. Update the appropriate service in its subdirectory (`rag-module/`, `personal-module/`, `integration/`)
3. For cross-service features, implement in the gateway (`integration/gateway.py`)
4. Add tests to `integration/test_integration.py`
5. Test using CLI tool and integration tests
6. Rebuild and validate using docker-compose commands

### File Structure
```
ai-ecosystem-integrated/
├── docker-compose.yml          # 4-service orchestration with health checks
├── .env                        # API keys and service configuration
├── rag-module/                 # RAG service with ChromaDB integration
│   ├── app.py                  # FastAPI app with vector search
│   ├── models.py               # Document and search models
│   ├── vector_search.py        # ChromaDB client and embedding logic
│   └── requirements.txt
├── personal-module/            # Personal service with Todoist sync
│   ├── app.py                  # FastAPI app with task management
│   ├── models.py               # Task and user preference models
│   ├── todoist_integration.py  # Bidirectional Todoist sync
│   └── requirements.txt
├── integration/                # Gateway and orchestration layer
│   ├── gateway.py              # Main unified API gateway
│   ├── cli.py                  # Command-line interface tool
│   ├── test_integration.py     # Comprehensive test suite
│   ├── openwebui_functions.py  # Chat interface functions
│   ├── orchestrator.py         # Service coordination logic
│   └── requirements.txt
└── shared-config/              # Shared utilities
    ├── llm_config.py           # LLM provider configurations
    └── database.py             # Database connection utilities
```

### Environment Configuration
Critical environment variables in `.env`:
- **LLM APIs**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`
- **External Services**: `TODOIST_API_KEY` for task synchronization
- **Service URLs**: `RAG_SERVICE_URL`, `PERSONAL_SERVICE_URL`, `CHROMA_URL` (Docker internal)

### OpenWebUI Integration
The gateway exposes comprehensive OpenWebUI-compatible functions at `/api/functions`:
- `unified_search()` - Cross-service semantic and keyword search
- `smart_assistant()` - Intent-based responses with action execution
- `research_and_schedule()` - Find relevant docs and create review tasks
- `daily_briefing()` - Calendar overview with relevant document suggestions

### CLI Development Tool
Use `python integration/cli.py` for interactive development:
- `status [--detailed]` - Service health and capabilities
- `ask "question"` - Test smart assistant with any query
- `search "query"` - Test unified search across all services
- `today [--include-docs]` - Daily briefing and task overview

### Testing Strategy
- **Integration tests**: `python integration/test_integration.py` validates all services
- **Health monitoring**: Built-in health checks with dependency management
- **Service isolation**: Each service can be tested independently
- **End-to-end validation**: CLI tool provides manual testing interface