# AI Ecosystem Integration

Integration layer combining RAG (Retrieval-Augmented Generation) and Personal modules with a unified API gateway.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   RAG Module    │    │ Personal Module │
│   Port: 8011    │    │   Port: 8012    │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────┬─────────────┬─┘
                 │             │
         ┌───────▼─────────────▼───────┐
         │     API Gateway             │
         │     Port: 8013              │
         │  Enhanced with OpenWebUI    │
         └─────────────────────────────┘
```

## Services

### RAG Module (Port 8011)
- Knowledge base search and retrieval
- Document management
- Endpoints: `/search`, `/documents`, `/health`

### Personal Module (Port 8012)
- Task management
- Personal preferences
- Schedule tracking
- Endpoints: `/tasks`, `/preferences`, `/schedule`, `/stats`

### API Gateway (Port 8013)
- Unified entry point
- Service orchestration
- OpenWebUI function support
- Enhanced endpoints:
  - `/search` - Unified search across both services
  - `/assistant` - Smart routing based on intent
  - `/status` - Complete service health overview
  - `/functions` - OpenWebUI function definitions

## Quick Start

1. **Start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Test the integration:**
   ```bash
   # Check overall health
   curl http://localhost:8013/health

   # Test unified search
   curl -X POST http://localhost:8013/search \
     -H "Content-Type: application/json" \
     -d '{"query": "AI", "include_rag": true, "include_personal": true}'

   # Test smart assistant
   curl -X POST http://localhost:8013/assistant \
     -H "Content-Type: application/json" \
     -d '{"message": "What tasks do I have?"}'
   ```

## Configuration

### Environment Variables (.env)
- `OPENAI_API_KEY` - OpenAI API access
- `ANTHROPIC_API_KEY` - Anthropic Claude API
- `GROQ_API_KEY` - Groq API for fast inference
- `GOOGLE_API_KEY` - Google Gemini API
- `TODOIST_API_KEY` - Todoist integration

### Service URLs
- Internal communication uses Docker network
- External access via mapped ports (8011, 8012, 8013)

## API Examples

### Unified Search
```bash
POST /search
{
  "query": "docker containers",
  "include_rag": true,
  "include_personal": true,
  "max_results": 10
}
```

### Smart Assistant
```bash
POST /assistant
{
  "message": "Show me pending tasks for today",
  "context": {"user_id": "user123"}
}
```

### Service Status
```bash
GET /status
# Returns comprehensive health and info for all services
```

## Development

### Project Structure
```
ai-ecosystem-integrated/
├── docker-compose.yml          # Main orchestration
├── .env                        # Shared configuration
├── README.md                   # This file
├── rag-module/                 # RAG service
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── personal-module/            # Personal service
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── gateway/                    # API Gateway
│   ├── gateway_enhanced.py
│   ├── Dockerfile
│   └── requirements.txt
└── shared-config/              # Shared utilities
    └── llm_config.py
```

### Adding New Services
1. Create service directory with Dockerfile
2. Add service to docker-compose.yml
3. Update gateway routing if needed
4. Add health checks and monitoring

## OpenWebUI Integration

The gateway exposes OpenWebUI-compatible functions:

- `search_knowledge` - Search RAG knowledge base
- `get_tasks` - Retrieve personal tasks
- `create_task` - Add new tasks
- `unified_search` - Search across all services

Access function definitions at `/functions`

## Monitoring & Health

- Service health: `GET /health` (individual services)
- Overall status: `GET /status` (gateway)
- Logs: `docker-compose logs [service-name]`

## Production Considerations

- Add authentication/authorization
- Implement rate limiting
- Add persistent storage volumes
- Configure proper logging
- Set up monitoring/alerting
- Use environment-specific .env files