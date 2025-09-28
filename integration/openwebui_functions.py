"""
OpenWebUI Functions for AI Ecosystem Integration

This module provides OpenWebUI-compatible functions that integrate
RAG knowledge base with personal task management and scheduling.
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re
from dateutil import parser

# Service URLs
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8001")
PERSONAL_SERVICE_URL = os.getenv("PERSONAL_SERVICE_URL", "http://personal-service:8002")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8003")

class ServiceClient:
    def __init__(self):
        self.session = None

    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=50)
            timeout = aiohttp.ClientTimeout(total=30)
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
                    return {"data": await response.text(), "status_code": response.status}
        except Exception as e:
            return {"error": str(e), "status_code": 500}

    async def close(self):
        if self.session:
            await self.session.close()

# Global client instance
client = ServiceClient()

def unified_search(query: str, max_results: int = 10, include_personal: bool = True) -> str:
    """
    Search both RAG and personal data for comprehensive results.

    Args:
        query: Search term to find in documents and tasks
        max_results: Maximum number of results to return (default: 10)
        include_personal: Whether to include personal tasks and schedule (default: True)

    Returns:
        Formatted results combining knowledge and personal information
    """
    async def _search():
        try:
            # Use the gateway's unified search
            search_response = await client.make_request(
                f"{GATEWAY_URL}/api/search",
                "POST",
                {
                    "query": query,
                    "max_results": max_results,
                    "include_personal": include_personal,
                    "include_context": True
                }
            )

            if "error" in search_response:
                return f"❌ Search failed: {search_response['error']}"

            results = []
            unified_results = search_response.get("unified_results", [])

            if not unified_results:
                return f"No results found for '{query}'. Try rephrasing your search or checking for typos."

            results.append(f"🔍 **Search Results for '{query}'** ({len(unified_results)} found)\n")

            for i, item in enumerate(unified_results, 1):
                source_emoji = "📚" if item.get("source") == "knowledge_base" else "📝"
                title = item.get("title", "Unknown")
                relevance = item.get("relevance_score", 0)

                results.append(f"{i}. {source_emoji} **{title}** (Score: {relevance:.2f})")

                if item.get("content"):
                    content = item["content"][:150] + "..." if len(item.get("content", "")) > 150 else item.get("content", "")
                    results.append(f"   {content}")

                # Add personal connections if any
                if item.get("personal_connections"):
                    connections = item["personal_connections"]
                    results.append(f"   🔗 Related: {len(connections)} personal items")

                results.append("")

            # Add processing time
            processing_time = search_response.get("processing_time", 0)
            results.append(f"⏱️ Processed in {processing_time:.2f}s")

            return "\n".join(results)

        except Exception as e:
            return f"❌ Search error: {str(e)}"

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_search())
    finally:
        loop.close()

def smart_assistant(message: str, include_actions: bool = True) -> str:
    """
    Use both RAG and Personal modules based on query intent.

    Args:
        message: Natural language message or question
        include_actions: Whether to perform actions like creating tasks (default: True)

    Returns:
        Assistant response with relevant information and actions taken
    """
    async def _assistant():
        try:
            # Use the gateway's unified assistant
            response = await client.make_request(
                f"{GATEWAY_URL}/api/assistant",
                "POST",
                {
                    "message": message,
                    "include_actions": include_actions,
                    "context": {"timestamp": datetime.now().isoformat()}
                }
            )

            if "error" in response:
                return f"❌ Assistant error: {response['error']}"

            assistant_response = response.get("response", "")
            intent = response.get("intent", "unknown")
            confidence = response.get("confidence", 0)
            actions = response.get("actions_taken", [])
            sources = response.get("data_sources", [])

            result = []

            # Add the main response
            result.append(assistant_response)

            # Add metadata if actions were taken
            if actions:
                result.append(f"\n🎯 **Actions Taken:**")
                for action in actions:
                    action_name = action.get("action", "unknown")
                    if action_name == "knowledge_search":
                        result.append(f"• Searched knowledge base ({action.get('results_count', 0)} results)")
                    elif action_name == "task_created":
                        result.append(f"• Created task (ID: {action.get('task_id', 'unknown')})")
                    elif action_name == "tasks_retrieved":
                        result.append(f"• Retrieved {action.get('count', 0)} tasks")
                    elif action_name == "stats_retrieved":
                        result.append(f"• Retrieved productivity statistics")
                    else:
                        result.append(f"• {action_name}")

            # Add data sources used
            if sources:
                source_names = {"knowledge_base": "📚 Knowledge Base", "personal": "📝 Personal Data"}
                source_list = [source_names.get(src, src) for src in sources]
                result.append(f"\n📊 **Sources:** {', '.join(source_list)}")

            # Add confidence if meaningful
            if confidence > 0.7:
                result.append(f"\n🎯 Intent: {intent} ({confidence:.1%} confidence)")

            return "\n".join(result)

        except Exception as e:
            return f"❌ Assistant error: {str(e)}"

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_assistant())
    finally:
        loop.close()

def research_and_schedule(query: str, schedule_hours_ahead: int = 24, duration_per_doc: int = 30, priority: str = "medium") -> str:
    """
    Find documents matching query then schedule review tasks.

    Args:
        query: Research topic to find documents for
        schedule_hours_ahead: Hours from now to start scheduling (default: 24)
        duration_per_doc: Minutes to allocate per document review (default: 30)
        priority: Task priority level - low, medium, high, urgent (default: medium)

    Returns:
        Formatted schedule with research tasks created
    """
    async def _research_schedule():
        try:
            # Calculate schedule time
            schedule_time = datetime.now() + timedelta(hours=schedule_hours_ahead)

            # Use the gateway's smart schedule function
            response = await client.make_request(
                f"{GATEWAY_URL}/api/smart-schedule",
                "POST",
                {
                    "query": query,
                    "schedule_time": schedule_time.isoformat(),
                    "duration_minutes": duration_per_doc,
                    "priority": priority
                }
            )

            if "error" in response:
                return f"❌ Research scheduling failed: {response['error']}"

            docs_found = response.get("documents_found", 0)
            events_created = response.get("events_created", [])
            total_time = response.get("total_scheduled_time", 0)

            if not events_created:
                return f"📖 Found {docs_found} documents for '{query}', but couldn't create any review tasks."

            result = []
            result.append(f"📚 **Research Schedule Created for '{query}'**\n")
            result.append(f"📊 **Summary:**")
            result.append(f"• Documents found: {docs_found}")
            result.append(f"• Review tasks created: {len(events_created)}")
            result.append(f"• Total time allocated: {total_time} minutes ({total_time/60:.1f} hours)")
            result.append(f"• Priority level: {priority.title()}\n")

            result.append(f"📅 **Schedule:**")
            for i, event in enumerate(events_created, 1):
                title = event.get("title", "Unknown Task")
                scheduled_time = event.get("scheduled_time", "")
                duration = event.get("estimated_duration", duration_per_doc)

                # Format the time nicely
                try:
                    dt = parser.parse(scheduled_time)
                    time_str = dt.strftime("%B %d at %I:%M %p")
                except:
                    time_str = scheduled_time

                result.append(f"{i}. **{title}**")
                result.append(f"   ⏰ {time_str}")
                result.append(f"   ⏱️ Duration: {duration} minutes")
                result.append("")

            result.append(f"✅ All review tasks have been added to your personal schedule.")

            return "\n".join(result)

        except Exception as e:
            return f"❌ Research scheduling error: {str(e)}"

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_research_schedule())
    finally:
        loop.close()

def daily_briefing(date: str = None, include_documents: bool = True, max_tasks: int = 10) -> str:
    """
    Combine calendar events with relevant documents for daily briefing.

    Args:
        date: Date for briefing in YYYY-MM-DD format (default: today)
        include_documents: Whether to include relevant documents (default: True)
        max_tasks: Maximum number of tasks to show (default: 10)

    Returns:
        Formatted daily briefing with calendar and relevant documents
    """
    async def _daily_briefing():
        try:
            target_date = date or datetime.now().strftime("%Y-%m-%d")

            result = []
            result.append(f"📅 **Daily Briefing for {target_date}**\n")

            # Get today's schedule
            schedule_response = await client.make_request(f"{PERSONAL_SERVICE_URL}/schedule")

            if "error" not in schedule_response:
                schedule = schedule_response.get("schedule", {})
                day_info = schedule.get(target_date, {"tasks": [], "events": [], "total_items": 0})

                tasks_today = day_info.get("tasks", [])
                events_today = day_info.get("events", [])

                if tasks_today or events_today:
                    result.append(f"📋 **Today's Schedule** ({len(tasks_today + events_today)} items):")

                    # Show events first
                    for event in events_today:
                        start_time = event.get("start_time", "")
                        title = event.get("title", "Untitled Event")
                        result.append(f"• 📅 {start_time} - {title}")

                    # Show tasks
                    for task in tasks_today[:max_tasks]:
                        title = task.get("title", "Untitled Task")
                        priority = task.get("priority", "medium")
                        priority_emoji = {"urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                        due_time = task.get("due_time", "")
                        time_info = f" at {due_time}" if due_time else ""
                        result.append(f"• {priority_emoji} {title}{time_info}")

                    if len(tasks_today) > max_tasks:
                        result.append(f"• ... and {len(tasks_today) - max_tasks} more tasks")
                else:
                    result.append(f"📋 **Today's Schedule:** No scheduled items")

            # Get general task summary
            tasks_response = await client.make_request(f"{PERSONAL_SERVICE_URL}/tasks")
            if "error" not in tasks_response:
                tasks = tasks_response.get("tasks", [])
                pending_tasks = [t for t in tasks if t.get("status") != "completed"]
                overdue_tasks = []

                # Check for overdue tasks
                today = datetime.now().date()
                for task in pending_tasks:
                    due_date_str = task.get("due_date")
                    if due_date_str:
                        try:
                            due_date = parser.parse(due_date_str).date()
                            if due_date < today:
                                overdue_tasks.append(task)
                        except:
                            pass

                result.append(f"\n📊 **Task Summary:**")
                result.append(f"• Total pending: {len(pending_tasks)}")
                if overdue_tasks:
                    result.append(f"• ⚠️ Overdue: {len(overdue_tasks)}")

                # Show overdue tasks
                if overdue_tasks:
                    result.append(f"\n🚨 **Overdue Tasks:**")
                    for task in overdue_tasks[:3]:
                        title = task.get("title", "Untitled")
                        due_date = task.get("due_date", "")
                        result.append(f"• {title} (Due: {due_date})")

                    if len(overdue_tasks) > 3:
                        result.append(f"• ... and {len(overdue_tasks) - 3} more overdue")

            # Include relevant documents if requested
            if include_documents:
                # Search for documents related to today's tasks
                task_keywords = []
                if 'tasks_today' in locals():
                    for task in tasks_today:
                        title_words = task.get("title", "").split()
                        task_keywords.extend([word for word in title_words if len(word) > 3])

                if task_keywords:
                    # Search for documents related to today's work
                    search_query = " ".join(task_keywords[:5])  # Use first 5 meaningful words

                    doc_response = await client.make_request(
                        f"{RAG_SERVICE_URL}/search",
                        "POST",
                        {"query": search_query, "max_results": 3}
                    )

                    if "error" not in doc_response and doc_response.get("results"):
                        docs = doc_response["results"]
                        result.append(f"\n📚 **Relevant Documents:**")
                        for doc in docs:
                            title = doc.get("title", doc.get("id", "Unknown"))
                            score = doc.get("similarity_score", doc.get("relevance_score", 0))
                            result.append(f"• {title} (Relevance: {score:.2f})")

            # Add productivity motivation
            result.append(f"\n💪 **Today's Focus:** Make progress on your priorities and maintain momentum!")

            return "\n".join(result)

        except Exception as e:
            return f"❌ Daily briefing error: {str(e)}"

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_daily_briefing())
    finally:
        loop.close()

# OpenWebUI Function Definitions for registration
OPENWEBUI_FUNCTIONS = [
    {
        "name": "unified_search",
        "description": "Search both RAG knowledge base and personal data for comprehensive results",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search term to find in documents and tasks"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10
            },
            "include_personal": {
                "type": "boolean",
                "description": "Whether to include personal tasks and schedule",
                "default": True
            }
        },
        "function": unified_search
    },
    {
        "name": "smart_assistant",
        "description": "Intelligent assistant that uses both RAG and Personal modules based on query intent",
        "parameters": {
            "message": {
                "type": "string",
                "description": "Natural language message or question"
            },
            "include_actions": {
                "type": "boolean",
                "description": "Whether to perform actions like creating tasks",
                "default": True
            }
        },
        "function": smart_assistant
    },
    {
        "name": "research_and_schedule",
        "description": "Find documents matching a topic then schedule review tasks automatically",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Research topic to find documents for"
            },
            "schedule_hours_ahead": {
                "type": "integer",
                "description": "Hours from now to start scheduling",
                "default": 24
            },
            "duration_per_doc": {
                "type": "integer",
                "description": "Minutes to allocate per document review",
                "default": 30
            },
            "priority": {
                "type": "string",
                "description": "Task priority level",
                "enum": ["low", "medium", "high", "urgent"],
                "default": "medium"
            }
        },
        "function": research_and_schedule
    },
    {
        "name": "daily_briefing",
        "description": "Generate daily briefing combining calendar events with relevant documents",
        "parameters": {
            "date": {
                "type": "string",
                "description": "Date for briefing in YYYY-MM-DD format (default: today)",
                "default": None
            },
            "include_documents": {
                "type": "boolean",
                "description": "Whether to include relevant documents",
                "default": True
            },
            "max_tasks": {
                "type": "integer",
                "description": "Maximum number of tasks to show",
                "default": 10
            }
        },
        "function": daily_briefing
    }
]