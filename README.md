# AI Ecosystem Integration

Complete microservices-based AI ecosystem with ChromaDB vector search, RAG knowledge base, personal task management, and unified intelligent gateway.

## 🚀 Quick Start

### 1. Start All Services
```bash
docker-compose up --build
```

**Services started:**
- **ChromaDB** (port 8000) - Vector database
- **RAG Service** (port 8001) - Knowledge base with semantic search
- **Personal Service** (port 8002) - Task management with Todoist sync
- **Gateway** (port 8003) - Unified AI assistant and API orchestration

### 2. Test Integration
```bash
# Run integration tests
python integration/test_integration.py

# Try the CLI
python integration/cli.py status
python integration/cli.py ask "What is machine learning?"
python integration/cli.py today --include-docs
```

### 3. Use the APIs
```bash
# Unified search across all services
curl -X POST http://localhost:8003/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Docker containers", "include_personal": true}'

# Smart assistant with intent recognition
curl -X POST http://localhost:8003/api/assistant \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a task to review AI papers due tomorrow"}'
```

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  ChromaDB   │    │ RAG Service │    │ Personal    │    │   Gateway   │
│ Vector DB   │    │ Knowledge   │    │ Tasks &     │    │ Orchestrate │
│ Port 8000   │    │ Port 8001   │    │ Port 8002   │    │ Port 8003   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       └──────────────────┼──────────────────┼──────────────────┘
                          │                  │
                      Vector Search     Task Management
                   Semantic Similarity  Todoist Integration
```

## 🎯 Key Features

### 🔍 **Unified Search**
- **Vector Similarity**: ChromaDB-powered semantic search
- **Cross-Service**: Search knowledge base + personal tasks
- **Context Enrichment**: Results enhanced with personal connections
- **Hybrid Fallback**: Vector search → text search → simple matching

### 🤖 **Smart Assistant**
- **Intent Recognition**: Pattern matching + LLM assistance
- **Entity Extraction**: Dates, priorities, actions from natural language
- **Context Awareness**: Remembers conversation and user preferences
- **Action Execution**: Can create tasks, search knowledge, get stats

### 📋 **Advanced Task Management**
- **Full CRUD**: Create, read, update, delete tasks with rich metadata
- **Todoist Sync**: Bi-directional synchronization
- **Priority & Tags**: Organization with multiple priority levels
- **Schedule Integration**: Calendar view with due dates
- **Productivity Analytics**: Completion rates, focus scores, trends

### 📚 **Enhanced RAG Module**
- **Vector Search**: Sentence transformers + ChromaDB
- **Document Management**: Upload, update, delete, batch operations
- **Semantic Retrieval**: Finds conceptually similar content
- **Metadata Support**: Rich document metadata and categorization

## 🛠️ Components

### Gateway (`integration/gateway.py`)
**Port 8003** - Unified API orchestrating all services

**Key Endpoints:**
- `POST /api/search` - Cross-service search with context
- `POST /api/assistant` - Intelligent routing and responses
- `POST /api/smart-schedule` - Research docs → create review tasks
- `GET /api/services` - Service health and capabilities
- `GET /api/functions` - OpenWebUI function definitions

### CLI Tool (`integration/cli.py`)
**Command-line interface** with comprehensive functionality

**Commands:**
```bash
ai-cli ask "What is Docker?"                    # Ask anything
ai-cli status --detailed                        # Check all services
ai-cli today --include-docs                     # Today's overview
ai-cli search "machine learning"                # Search everything
```

### OpenWebUI Functions (`integration/openwebui_functions.py`)
**Ready-to-use functions** for chat interface integration

**Functions:**
1. `unified_search()` - Search knowledge + personal data
2. `smart_assistant()` - Intent-based responses
3. `research_and_schedule()` - Find docs → schedule reviews
4. `daily_briefing()` - Calendar + relevant documents

## 📊 API Examples

### Unified Search
```bash
curl -X POST http://localhost:8003/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "max_results": 10,
    "include_personal": true,
    "include_context": true
  }'
```

### Smart Assistant
```bash
curl -X POST http://localhost:8003/api/assistant \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a high priority task to review ML papers due next Friday",
    "include_actions": true
  }'
```

### Research & Schedule
```bash
curl -X POST http://localhost:8003/api/smart-schedule \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning fundamentals",
    "schedule_hours_ahead": 24,
    "duration_minutes": 45,
    "priority": "high"
  }'
```

## 🔧 Development

### Environment Variables
```bash
# API Keys (in .env file)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
TODOIST_API_KEY=your_todoist_key

# Service URLs (for Docker)
RAG_SERVICE_URL=http://rag-service:8001
PERSONAL_SERVICE_URL=http://personal-service:8002
CHROMA_URL=http://chromadb:8000
```

### Service Testing
```bash
# Individual service health
curl http://localhost:8000/api/v1/heartbeat  # ChromaDB
curl http://localhost:8001/health            # RAG Service
curl http://localhost:8002/health            # Personal Service
curl http://localhost:8003/health            # Gateway

# View logs
docker-compose logs [service-name]
```

### Adding Documents
```bash
# Add document to knowledge base
curl -X POST http://localhost:8001/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Machine Learning Basics",
    "content": "Machine learning is a subset of AI...",
    "metadata": {"category": "ai", "difficulty": "beginner"}
  }'
```

### Creating Tasks
```bash
# Create task with Todoist sync
curl -X POST http://localhost:8002/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Review AI research papers",
    "priority": "high",
    "due_date": "2024-01-15T10:00:00Z",
    "tags": ["research", "ai"],
    "sync_to_todoist": true
  }'
```

## 🎨 Frontend Integration

### OpenWebUI Setup
```bash
# Get function definitions
curl http://localhost:8003/api/functions

# Use in OpenWebUI chat:
# unified_search("machine learning")
# daily_briefing()
# research_and_schedule("AI papers", 48, 60)
```

### Custom Frontend
```javascript
// React/Next.js example
const response = await fetch('http://localhost:8003/api/assistant', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: userInput,
    include_actions: true
  })
});

const data = await response.json();
console.log(data.response); // Assistant response
console.log(data.actions_taken); // Actions performed
```

## 🚨 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check ports aren't in use
docker-compose down
docker system prune -f
docker-compose up --build
```

**ChromaDB connection issues:**
```bash
# Reset ChromaDB data
docker-compose down
docker volume rm ai-ecosystem-integrated_chroma_data
docker-compose up --build
```

**API key issues:**
```bash
# Check .env file exists and has valid keys
cat .env | grep -E "(OPENAI|ANTHROPIC|TODOIST)_API_KEY"
```

### Health Monitoring
```bash
# Comprehensive health check
python integration/cli.py status --detailed

# Integration test suite
python integration/test_integration.py
```

## 📈 Performance Features

- **Async/Await**: Non-blocking I/O throughout
- **Connection Pooling**: Efficient HTTP client management
- **Caching**: ChromaDB persistence and query optimization
- **Health Checks**: Automatic service monitoring
- **Graceful Degradation**: Fallback search methods
- **Background Tasks**: Non-blocking task processing

## 🔐 Security Features

- **Input Validation**: Pydantic models with constraints
- **Error Handling**: Comprehensive exception management
- **CORS Protection**: Configurable cross-origin policies
- **Health Endpoints**: Monitoring without sensitive data
- **Environment Secrets**: API keys via environment variables

---

**🎉 Ready to use!** Start with `docker-compose up --build` and explore the integrated AI ecosystem.