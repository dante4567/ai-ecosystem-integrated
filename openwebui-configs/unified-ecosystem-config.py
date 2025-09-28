"""
OpenWebUI Functions for Complete AI Ecosystem
Combines RAG and Organizer services with intelligent routing through the gateway
"""

import json
import requests
from typing import List, Dict, Any

# Configuration
RAG_SERVICE_URL = "http://localhost:8001"
ORGANIZER_SERVICE_URL = "http://localhost:8002"
GATEWAY_URL = "http://localhost:8003"

def smart_assistant(message: str) -> str:
    """
    Ask the smart assistant - it automatically routes to the right service.

    Args:
        message: Natural language message or question

    Returns:
        AI response with intelligent routing
    """
    try:
        # Try gateway first (best experience)
        response = requests.post(
            f"{GATEWAY_URL}/api/assistant",
            json={"message": message, "include_actions": True},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            result = f"🤖 **Smart Assistant Response:**\n\n{data.get('response', 'No response')}\n\n"

            # Show intent and confidence
            intent = data.get('intent', 'unknown')
            confidence = data.get('confidence', 0)
            result += f"🎯 **Intent:** {intent} (confidence: {confidence:.2f})\n\n"

            # Show actions taken
            actions = data.get('actions_taken', [])
            if actions:
                result += "🔧 **Actions Taken:**\n"
                for action in actions:
                    action_type = action.get('type', 'action')
                    details = action.get('details', '')
                    result += f"• {action_type}: {details}\n"
                result += "\n"

            # Show data sources
            sources = data.get('data_sources', [])
            if sources:
                result += f"📊 **Sources:** {', '.join(sources)}\n"

            return result
        else:
            # Fallback to direct service routing
            return _fallback_routing(message)

    except Exception as e:
        # Fallback to direct service routing
        return _fallback_routing(message)

def _fallback_routing(message: str) -> str:
    """Fallback routing when gateway is unavailable"""
    message_lower = message.lower()

    # Document-related keywords
    doc_keywords = ["search", "document", "find", "knowledge", "pdf", "file", "what does", "explain"]

    # Organizer-related keywords
    org_keywords = ["task", "todo", "contact", "event", "calendar", "remind", "schedule", "meeting", "create"]

    if any(keyword in message_lower for keyword in doc_keywords):
        return chat_with_documents(message)
    elif any(keyword in message_lower for keyword in org_keywords):
        return ask_personal_assistant(message)
    else:
        # Default to personal assistant
        return ask_personal_assistant(message)

def unified_search(query: str, max_results: int = 8) -> str:
    """
    Search across both documents and personal data.

    Args:
        query: Search query
        max_results: Maximum total results

    Returns:
        Combined search results
    """
    try:
        # Try gateway unified search first
        response = requests.post(
            f"{GATEWAY_URL}/api/search",
            json={
                "query": query,
                "max_results": max_results,
                "include_personal": True,
                "include_context": True
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            result = f"🔍 **Unified Search Results for '{query}':**\n\n"

            # RAG results
            rag_results = data.get('rag_results', [])
            if rag_results:
                result += "📄 **From Documents:**\n"
                for i, doc in enumerate(rag_results[:4], 1):
                    content = doc.get('content', 'No content')[:150]
                    title = doc.get('metadata', {}).get('title', 'Untitled')
                    score = doc.get('relevance_score', 0)
                    result += f"{i}. **{title}** (Score: {score:.3f})\n"
                    result += f"   {content}...\n\n"

            # Personal context
            personal_context = data.get('personal_context', [])
            if personal_context:
                result += "👤 **From Personal Data:**\n"
                for i, item in enumerate(personal_context[:4], 1):
                    title = item.get('title', 'Unknown')
                    item_type = item.get('type', 'item')
                    result += f"{i}. {item_type.title()}: {title}\n"

                result += "\n"

            # Processing info
            processing_time = data.get('processing_time', 0)
            result += f"⏱️ Processing time: {processing_time:.3f}s"

            return result
        else:
            # Fallback to direct RAG search
            return search_documents(query, max_results)

    except Exception as e:
        # Fallback to direct RAG search
        return search_documents(query, max_results)

def chat_with_documents(question: str, model: str = "groq/llama-3.1-8b-instant") -> str:
    """
    Chat with your documents using RAG.

    Args:
        question: Question about your documents
        model: LLM model to use

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

            result = f"📄 **Document Chat** (via {model_used}):\n\n{answer}\n\n"

            if sources:
                result += f"📚 **Sources:** {len(sources)} documents used\n"
                for i, source in enumerate(sources[:3], 1):
                    title = source.get("title", "Unknown")
                    result += f"{i}. {title}\n"

            return result
        else:
            return f"❌ Document chat failed: {response.status_code}"

    except Exception as e:
        return f"❌ Document chat error: {str(e)}"

def ask_personal_assistant(message: str) -> str:
    """
    Ask your personal assistant for help with tasks, events, and contacts.

    Args:
        message: Natural language message

    Returns:
        Personal assistant response
    """
    try:
        response = requests.post(
            f"{ORGANIZER_SERVICE_URL}/chat",
            json={"message": message},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            assistant_response = data.get("response", "No response")

            result = f"🗂️ **Personal Assistant:**\n\n{assistant_response}\n\n"

            # Show actions taken
            actions = data.get("actions_taken", [])
            if actions:
                result += "🎯 **Actions Taken:**\n"
                for action in actions:
                    action_type = action.get("type", "action")
                    details = action.get("details", "")
                    result += f"• {action_type}: {details}\n"

            return result
        else:
            return f"❌ Personal assistant request failed: {response.status_code}"

    except Exception as e:
        return f"❌ Personal assistant error: {str(e)}"

def search_documents(query: str, max_results: int = 5) -> str:
    """
    Search only documents in the RAG knowledge base.

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        Document search results
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
                return f"📄 No documents found for: {query}"

            formatted_results = f"📄 **Document Search for '{query}':**\n\n"

            for i, result in enumerate(results, 1):
                content = result.get("content", "No content")[:150]
                score = result.get("relevance_score", 0)
                title = result.get("metadata", {}).get("title", "Untitled")

                formatted_results += f"**{i}. {title}** (Score: {score:.3f})\n"
                formatted_results += f"{content}...\n\n"

            return formatted_results
        else:
            return f"❌ Document search failed: {response.status_code}"

    except Exception as e:
        return f"❌ Document search error: {str(e)}"

def get_my_overview() -> str:
    """
    Get a complete overview of your day: tasks, events, and recent documents.

    Returns:
        Daily overview combining all services
    """
    result = "📊 **Your Complete Overview:**\n\n"

    # Get today's events
    try:
        events_response = requests.get(
            f"{ORGANIZER_SERVICE_URL}/api/events",
            params={"days": 1},
            timeout=10
        )

        if events_response.status_code == 200:
            events_data = events_response.json()
            events = events_data.get("events", [])

            result += "📅 **Today's Events:**\n"
            if events:
                for event in events[:3]:
                    title = event.get("title", "Untitled")
                    start_time = event.get("start_time", "")
                    result += f"• {title} at {start_time}\n"
            else:
                result += "• No events scheduled\n"
        else:
            result += "📅 Events: Unable to fetch\n"

    except:
        result += "📅 Events: Service unavailable\n"

    result += "\n"

    # Get pending tasks
    try:
        tasks_response = requests.get(
            f"{ORGANIZER_SERVICE_URL}/api/todos",
            params={"limit": 5},
            timeout=10
        )

        if tasks_response.status_code == 200:
            tasks_data = tasks_response.json()
            tasks = tasks_data.get("todos", [])

            result += "✅ **Priority Tasks:**\n"
            if tasks:
                for task in tasks[:5]:
                    title = task.get("title", "Untitled")
                    priority = task.get("priority", "medium")
                    emoji = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                    result += f"{emoji} {title}\n"
            else:
                result += "• All caught up!\n"
        else:
            result += "✅ Tasks: Unable to fetch\n"

    except:
        result += "✅ Tasks: Service unavailable\n"

    result += "\n"

    # Get service status
    try:
        rag_health = requests.get(f"{RAG_SERVICE_URL}/health", timeout=5)
        organizer_health = requests.get(f"{ORGANIZER_SERVICE_URL}/health", timeout=5)

        result += "🏥 **Services:**\n"
        result += f"📄 RAG Service: {'✅' if rag_health.status_code == 200 else '❌'}\n"
        result += f"🗂️ Organizer: {'✅' if organizer_health.status_code == 200 else '❌'}\n"

    except:
        result += "🏥 Services: Status unknown\n"

    return result

def schedule_with_research(topic: str, hours_ahead: int = 24, duration_minutes: int = 60) -> str:
    """
    Research a topic in documents and schedule time to review it.

    Args:
        topic: Topic to research and schedule
        hours_ahead: Hours from now to schedule
        duration_minutes: Meeting duration

    Returns:
        Research results and scheduling confirmation
    """
    try:
        # Use gateway's smart scheduling if available
        response = requests.post(
            f"{GATEWAY_URL}/api/smart-schedule",
            json={
                "query": topic,
                "schedule_hours_ahead": hours_ahead,
                "duration_minutes": duration_minutes,
                "priority": "medium"
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            result = f"🔬 **Research & Schedule for '{topic}':**\n\n"

            # Research results
            documents_found = data.get('documents_found', 0)
            if documents_found > 0:
                result += f"📚 Found {documents_found} relevant documents\n"

                # Show key findings
                key_points = data.get('key_points', [])
                if key_points:
                    result += "\n🔍 **Key Findings:**\n"
                    for i, point in enumerate(key_points[:3], 1):
                        result += f"{i}. {point}\n"
            else:
                result += "📚 No relevant documents found\n"

            # Scheduling result
            scheduled = data.get('scheduled', False)
            if scheduled:
                result += f"\n📅 **Scheduled:** Review session in {hours_ahead} hours ({duration_minutes} min)\n"
                event_id = data.get('event_id', '')
                if event_id:
                    result += f"🆔 Event ID: {event_id}\n"
            else:
                result += f"\n❌ **Scheduling failed:** {data.get('error', 'Unknown error')}\n"

            return result
        else:
            # Fallback: manual research + scheduling
            return _manual_research_and_schedule(topic, hours_ahead, duration_minutes)

    except Exception as e:
        # Fallback: manual research + scheduling
        return _manual_research_and_schedule(topic, hours_ahead, duration_minutes)

def _manual_research_and_schedule(topic: str, hours_ahead: int, duration_minutes: int) -> str:
    """Manual fallback for research and scheduling"""
    result = f"🔬 **Research & Schedule for '{topic}' (Manual Mode):**\n\n"

    # Search documents
    search_result = search_documents(topic, 3)
    result += search_result + "\n\n"

    # Try to create calendar event
    try:
        from datetime import datetime, timedelta
        schedule_time = datetime.now() + timedelta(hours=hours_ahead)
        schedule_str = schedule_time.strftime("%Y-%m-%d %H:%M")

        assistant_message = f"Schedule '{topic} review session' on {schedule_str} for {duration_minutes} minutes"
        schedule_result = ask_personal_assistant(assistant_message)
        result += schedule_result

    except Exception as e:
        result += f"❌ Manual scheduling failed: {str(e)}"

    return result

def get_ecosystem_status() -> str:
    """
    Get complete AI ecosystem status and statistics.

    Returns:
        Full ecosystem health report
    """
    result = "🌐 **AI Ecosystem Status:**\n\n"

    services = {
        "RAG Service": RAG_SERVICE_URL,
        "Organizer Service": ORGANIZER_SERVICE_URL,
        "Gateway": GATEWAY_URL
    }

    all_healthy = True

    for service_name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                result += f"✅ **{service_name}**: Healthy\n"

                # Get additional info
                data = response.json()
                if service_name == "RAG Service":
                    chromadb = data.get("chromadb", "unknown")
                    ocr = data.get("ocr_available", False)
                    result += f"   🗄️ ChromaDB: {chromadb}, 📷 OCR: {'✅' if ocr else '❌'}\n"

                elif service_name == "Organizer Service":
                    database = data.get("database", "unknown")
                    email = data.get("email_enabled", False)
                    result += f"   🗄️ Database: {database}, 📧 Email: {'✅' if email else '❌'}\n"

            else:
                result += f"❌ **{service_name}**: Unhealthy (HTTP {response.status_code})\n"
                all_healthy = False

        except Exception as e:
            result += f"❌ **{service_name}**: Unreachable\n"
            all_healthy = False

    result += f"\n🎯 **Overall Status**: {'🟢 All Systems Operational' if all_healthy else '🟡 Some Issues Detected'}\n"

    # Get usage stats if available
    try:
        gateway_response = requests.get(f"{GATEWAY_URL}/api/services", timeout=5)
        if gateway_response.status_code == 200:
            gateway_data = gateway_response.json()
            result += f"\n📊 **Gateway Info**: {len(gateway_data.get('services', []))} services integrated\n"
    except:
        pass

    return result

# OpenWebUI Function Registration
def get_openwebui_functions():
    """Return OpenWebUI function definitions for unified ecosystem"""
    return {
        "smart_assistant": {
            "function": smart_assistant,
            "description": "Ask the smart AI assistant (auto-routes to best service)",
            "parameters": {
                "message": {"type": "str", "description": "Natural language message or question"}
            }
        },
        "unified_search": {
            "function": unified_search,
            "description": "Search across documents and personal data",
            "parameters": {
                "query": {"type": "str", "description": "Search query"},
                "max_results": {"type": "int", "description": "Max results (default: 8)"}
            }
        },
        "chat_with_documents": {
            "function": chat_with_documents,
            "description": "Chat specifically with your documents",
            "parameters": {
                "question": {"type": "str", "description": "Question about documents"},
                "model": {"type": "str", "description": "LLM model (optional)"}
            }
        },
        "ask_personal_assistant": {
            "function": ask_personal_assistant,
            "description": "Ask personal assistant about tasks/events/contacts",
            "parameters": {
                "message": {"type": "str", "description": "Message for personal assistant"}
            }
        },
        "get_my_overview": {
            "function": get_my_overview,
            "description": "Get complete daily overview",
            "parameters": {}
        },
        "schedule_with_research": {
            "function": schedule_with_research,
            "description": "Research topic and schedule review time",
            "parameters": {
                "topic": {"type": "str", "description": "Topic to research"},
                "hours_ahead": {"type": "int", "description": "Hours from now (default: 24)"},
                "duration_minutes": {"type": "int", "description": "Duration in minutes (default: 60)"}
            }
        },
        "search_documents": {
            "function": search_documents,
            "description": "Search only documents (RAG service)",
            "parameters": {
                "query": {"type": "str", "description": "Search query"},
                "max_results": {"type": "int", "description": "Max results (default: 5)"}
            }
        },
        "get_ecosystem_status": {
            "function": get_ecosystem_status,
            "description": "Get complete ecosystem health status",
            "parameters": {}
        }
    }