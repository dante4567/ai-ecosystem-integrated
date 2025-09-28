# AI Ecosystem Integration

Complete integration layer that orchestrates RAG and Personal services with CLI and OpenWebUI support.

## Quick Start

### 1. Start All Services
```bash
# From the root directory
docker-compose up --build
```

This starts:
- **ChromaDB** on port 8000
- **RAG Service** on port 8001
- **Personal Service** on port 8002
- **Gateway** on port 8003

### 2. Test the Integration

#### Using the CLI
```bash
# Install CLI (optional - can run directly)
cd integration
pip install -e .

# Check service health
python cli.py status

# Ask questions
python cli.py ask "What is machine learning?"
python cli.py ask "Create a task to review AI papers"

# Search across all services
python cli.py search "docker containers"

# Get today's overview
python cli.py today --include-docs
```

#### Using the API
```bash
# Unified search across both services
curl -X POST http://localhost:8003/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "max_results": 5, "include_personal": true}'

# Smart assistant
curl -X POST http://localhost:8003/api/assistant \
  -H "Content-Type: application/json" \
  -d '{"message": "Show my tasks for today"}'

# Research and schedule
curl -X POST http://localhost:8003/api/smart-schedule \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "duration_minutes": 45}'
```

## Components

### Gateway (`gateway.py`)
**Port: 8003**

Unified API orchestrating all services:
- `/api/search` - Cross-service search with context enrichment
- `/api/assistant` - Intelligent routing based on intent
- `/api/smart-schedule` - Research docs then create review tasks
- `/api/services` - Service health and capabilities
- `/api/functions` - OpenWebUI function definitions

### OpenWebUI Functions (`openwebui_functions.py`)

Ready-to-use functions for OpenWebUI integration:

1. **`unified_search()`** - Search knowledge + personal data
2. **`smart_assistant()`** - Intent-based assistant
3. **`research_and_schedule()`** - Find docs → schedule reviews
4. **`daily_briefing()`** - Calendar + relevant documents

### CLI Tool (`cli.py`)

Command-line interface with five main commands:

```bash
ai-cli ask "What is Docker?"           # Query with unified response
ai-cli status --detailed               # Health check all services
ai-cli today --include-docs           # Today's schedule + docs
ai-cli search "machine learning"       # Cross-service search
```

## Service Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  ChromaDB   │    │ RAG Service │    │ Personal    │
│  Port 8000  │    │ Port 8001   │    │ Port 8002   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                   ┌──────▼──────┐
                   │   Gateway   │
                   │  Port 8003  │
                   └─────────────┘
```

## Integration Examples

### OpenWebUI Setup
```bash
# Add function definitions to OpenWebUI
curl http://localhost:8003/api/functions

# Use functions in chat:
# "unified_search('machine learning')"
# "daily_briefing()"
# "research_and_schedule('AI papers', 48, 60)"
```

### Custom Integration
```python
import aiohttp

async def query_ecosystem(message):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8003/api/assistant",
            json={"message": message, "include_actions": True}
        ) as response:
            return await response.json()

# Usage
result = await query_ecosystem("What tasks do I have today?")
print(result["response"])
```

## Environment Variables

Key variables for service URLs:
```bash
RAG_SERVICE_URL=http://rag-service:8001
PERSONAL_SERVICE_URL=http://personal-service:8002
GATEWAY_URL=http://gateway:8003
CHROMA_URL=http://chromadb:8000
```

## Features

✅ **Unified Search** - Single query across knowledge + personal data
✅ **Smart Routing** - Intent detection routes to appropriate services
✅ **Context Enrichment** - Knowledge results enhanced with personal context
✅ **Auto-scheduling** - Research docs → create review tasks automatically
✅ **Daily Briefings** - Calendar + relevant documents in one view
✅ **Health Monitoring** - Comprehensive service health checks
✅ **CLI Interface** - Command-line access to all functionality
✅ **OpenWebUI Ready** - Functions for chat interface integration

## Troubleshooting

### Service Health
```bash
python cli.py status --detailed
```

### Individual Service Testing
```bash
# Test RAG directly
curl http://localhost:8001/health

# Test Personal directly
curl http://localhost:8002/health

# Test ChromaDB
curl http://localhost:8000/api/v1/heartbeat
```

### Logs
```bash
docker-compose logs gateway
docker-compose logs rag-service
docker-compose logs personal-service
```

The integration layer provides a complete solution for unified AI ecosystem access!