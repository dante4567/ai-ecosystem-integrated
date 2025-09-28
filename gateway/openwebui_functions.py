"""
OpenWebUI Functions for AI Ecosystem Integration

This module provides OpenWebUI-compatible functions that integrate
RAG knowledge base with personal task management and calendar features.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re


class OpenWebUIFunctions:
    def __init__(self, rag_service_url: str, personal_service_url: str):
        self.rag_service_url = rag_service_url
        self.personal_service_url = personal_service_url
        self.session = None

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

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


# Global instance
functions_handler = OpenWebUIFunctions(
    rag_service_url="http://rag-module:8001",
    personal_service_url="http://personal-module:8002"
)


def search_knowledge_and_calendar(query: str, days_ahead: int = 7) -> str:
    """
    Search both knowledge base and calendar/tasks for comprehensive results.

    Args:
        query: Search term to find in documents and tasks
        days_ahead: Number of days to look ahead in calendar (default: 7)

    Returns:
        Formatted results combining knowledge and calendar information
    """
    async def _search():
        # Search knowledge base
        knowledge_result = await functions_handler.make_request(
            f"{functions_handler.rag_service_url}/search",
            "POST",
            {"query": query, "max_results": 5}
        )

        # Get tasks and schedule
        tasks_result = await functions_handler.make_request(
            f"{functions_handler.personal_service_url}/tasks"
        )

        schedule_result = await functions_handler.make_request(
            f"{functions_handler.personal_service_url}/schedule"
        )

        # Format results
        results = []

        # Add knowledge results
        if "results" in knowledge_result:
            results.append("## 📚 Knowledge Base Results:")
            for item in knowledge_result["results"]:
                relevance = item.get("relevance_score", 0)
                results.append(f"**{item.get('id', 'Unknown')}** (Relevance: {relevance:.2f})")
                results.append(f"{item.get('content', '')[:200]}...")
                results.append("")

        # Add calendar/task results
        if "tasks" in tasks_result:
            # Filter tasks that match the query
            matching_tasks = []
            for task in tasks_result["tasks"]:
                if query.lower() in task.get("title", "").lower():
                    matching_tasks.append(task)

            if matching_tasks:
                results.append("## 📅 Matching Tasks:")
                for task in matching_tasks:
                    status = "✅" if task.get("completed") else "⏳"
                    due = task.get("due_date", "No due date")
                    results.append(f"{status} **{task.get('title')}** (Due: {due})")
                results.append("")

        # Add today's schedule
        if "tasks_due" in schedule_result and schedule_result["tasks_due"]:
            results.append("## 📆 Today's Schedule:")
            for task in schedule_result["tasks_due"]:
                results.append(f"⏰ {task.get('title')} (Due today)")
            results.append("")

        if not results:
            return f"No results found for '{query}' in knowledge base or calendar."

        return "\n".join(results)

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_search())
    finally:
        loop.close()


def create_task_from_document_summary(document_id: str, summary: str, priority: str = "medium", days_until_due: int = 3) -> str:
    """
    Create a task based on a document summary with automatic scheduling.

    Args:
        document_id: ID of the document from knowledge base
        summary: Brief summary of what needs to be done
        priority: Task priority (low, medium, high)
        days_until_due: Number of days from now for due date

    Returns:
        Confirmation message with task details
    """
    async def _create_task():
        # Get document details first
        doc_result = await functions_handler.make_request(
            f"{functions_handler.rag_service_url}/documents"
        )

        document_found = False
        doc_content = ""

        if "documents" in doc_result:
            for doc in doc_result["documents"]:
                if doc.get("id") == document_id:
                    document_found = True
                    doc_content = doc.get("preview", "")
                    break

        if not document_found:
            return f"❌ Document '{document_id}' not found in knowledge base."

        # Create due date
        due_date = (datetime.now() + timedelta(days=days_until_due)).strftime("%Y-%m-%d")

        # Create task title from summary and document
        task_title = f"Review: {document_id} - {summary}"

        # Create the task
        task_result = await functions_handler.make_request(
            f"{functions_handler.personal_service_url}/tasks",
            "POST",
            {
                "title": task_title,
                "due_date": due_date,
                "priority": priority
            }
        )

        if "error" in task_result:
            return f"❌ Failed to create task: {task_result['error']}"

        return f"""✅ **Task Created Successfully!**

📋 **Task:** {task_title}
📅 **Due Date:** {due_date}
🔥 **Priority:** {priority.capitalize()}
📄 **Document Preview:** {doc_content[:100]}...

The task has been added to your personal schedule."""

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_create_task())
    finally:
        loop.close()


def schedule_document_review(document_ids: List[str], review_type: str = "detailed", time_per_doc: int = 30) -> str:
    """
    Schedule time blocks for reviewing multiple documents.

    Args:
        document_ids: List of document IDs to review
        review_type: Type of review (quick, detailed, deep)
        time_per_doc: Minutes to allocate per document

    Returns:
        Formatted schedule with review tasks
    """
    async def _schedule_review():
        # Get documents info
        doc_result = await functions_handler.make_request(
            f"{functions_handler.rag_service_url}/documents"
        )

        if "error" in doc_result:
            return f"❌ Failed to fetch documents: {doc_result['error']}"

        available_docs = {}
        if "documents" in doc_result:
            for doc in doc_result["documents"]:
                available_docs[doc.get("id")] = doc

        # Validate document IDs
        missing_docs = [doc_id for doc_id in document_ids if doc_id not in available_docs]
        if missing_docs:
            return f"❌ Documents not found: {', '.join(missing_docs)}"

        # Calculate time blocks based on review type
        time_multipliers = {
            "quick": 1.0,
            "detailed": 1.5,
            "deep": 2.0
        }

        actual_time = int(time_per_doc * time_multipliers.get(review_type, 1.5))

        # Create tasks for each document
        created_tasks = []
        start_date = datetime.now()

        for i, doc_id in enumerate(document_ids):
            due_date = (start_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")

            task_title = f"{review_type.capitalize()} review: {doc_id} ({actual_time}min)"

            task_result = await functions_handler.make_request(
                f"{functions_handler.personal_service_url}/tasks",
                "POST",
                {
                    "title": task_title,
                    "due_date": due_date,
                    "priority": "medium"
                }
            )

            if "error" not in task_result:
                created_tasks.append({
                    "doc_id": doc_id,
                    "due_date": due_date,
                    "time": actual_time,
                    "preview": available_docs[doc_id].get("preview", "")[:100]
                })

        if not created_tasks:
            return "❌ Failed to create any review tasks."

        # Format response
        result = [f"✅ **Document Review Schedule Created** ({review_type} review)\n"]

        total_time = len(created_tasks) * actual_time
        result.append(f"⏱️ **Total Time:** {total_time} minutes ({total_time/60:.1f} hours)")
        result.append(f"📅 **Schedule:** {len(created_tasks)} documents over {len(created_tasks)} days\n")

        for i, task in enumerate(created_tasks, 1):
            result.append(f"**Day {i} - {task['due_date']}:**")
            result.append(f"📖 {task['doc_id']} ({task['time']} min)")
            result.append(f"📄 Preview: {task['preview']}...")
            result.append("")

        return "\n".join(result)

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_schedule_review())
    finally:
        loop.close()


def smart_document_assistant(query: str, action: str = "search") -> str:
    """
    Smart assistant that can search, summarize, and create tasks from documents.

    Args:
        query: Natural language query
        action: Action to perform (search, summarize, schedule, create_task)

    Returns:
        Response based on the requested action
    """
    async def _smart_assistant():
        if action == "search":
            return await _search_documents(query)
        elif action == "summarize":
            return await _summarize_documents(query)
        elif action == "schedule":
            return await _schedule_from_query(query)
        elif action == "create_task":
            return await _create_task_from_query(query)
        else:
            return "❌ Unknown action. Available actions: search, summarize, schedule, create_task"

    async def _search_documents(query):
        result = await functions_handler.make_request(
            f"{functions_handler.rag_service_url}/search",
            "POST",
            {"query": query, "max_results": 3}
        )

        if "results" in result and result["results"]:
            response = [f"🔍 **Search Results for '{query}':**\n"]
            for item in result["results"]:
                response.append(f"📄 **{item.get('id')}** (Score: {item.get('relevance_score', 0):.2f})")
                response.append(f"{item.get('content', '')[:150]}...")
                response.append("")
            return "\n".join(response)
        else:
            return f"No documents found matching '{query}'"

    async def _summarize_documents(query):
        # Get search results
        search_result = await functions_handler.make_request(
            f"{functions_handler.rag_service_url}/search",
            "POST",
            {"query": query, "max_results": 5}
        )

        if "results" in search_result and search_result["results"]:
            summary = [f"📋 **Summary of documents related to '{query}':**\n"]

            for item in search_result["results"]:
                doc_id = item.get('id', 'Unknown')
                content = item.get('content', '')

                # Simple extractive summary (first sentence + key points)
                sentences = content.split('.')
                key_info = sentences[0] if sentences else content[:100]

                summary.append(f"**{doc_id}:** {key_info}...")

            summary.append(f"\n💡 **Suggested Actions:**")
            summary.append(f"- Create review tasks: Use create_task_from_document_summary()")
            summary.append(f"- Schedule reading time: Use schedule_document_review()")

            return "\n".join(summary)
        else:
            return f"No documents found to summarize for '{query}'"

    async def _schedule_from_query(query):
        # Extract document IDs from query or search for relevant docs
        doc_ids = re.findall(r'\b([a-zA-Z_]+)\b', query)

        if not doc_ids:
            # Search for documents first
            search_result = await functions_handler.make_request(
                f"{functions_handler.rag_service_url}/search",
                "POST",
                {"query": query, "max_results": 3}
            )

            if "results" in search_result:
                doc_ids = [item.get('id') for item in search_result["results"]]

        if doc_ids:
            return schedule_document_review(doc_ids[:3], "detailed", 30)
        else:
            return f"❌ No documents found to schedule for '{query}'"

    async def _create_task_from_query(query):
        # Simple parsing to extract document and action
        words = query.lower().split()

        # Look for document keywords
        doc_candidates = []
        for word in words:
            if len(word) > 2 and word.isalpha():
                doc_candidates.append(word)

        if doc_candidates:
            doc_id = doc_candidates[0]  # Take first candidate
            summary = f"Follow up on {query}"
            return create_task_from_document_summary(doc_id, summary, "medium", 2)
        else:
            return f"❌ Could not identify document from query: '{query}'"

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_smart_assistant())
    finally:
        loop.close()


# OpenWebUI Function Definitions
OPENWEBUI_FUNCTIONS = [
    {
        "name": "search_knowledge_and_calendar",
        "description": "Search both knowledge base and calendar/tasks for comprehensive results",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Search term to find in documents and tasks"
            },
            "days_ahead": {
                "type": "integer",
                "description": "Number of days to look ahead in calendar",
                "default": 7
            }
        },
        "function": search_knowledge_and_calendar
    },
    {
        "name": "create_task_from_document_summary",
        "description": "Create a task based on a document summary with automatic scheduling",
        "parameters": {
            "document_id": {
                "type": "string",
                "description": "ID of the document from knowledge base"
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of what needs to be done"
            },
            "priority": {
                "type": "string",
                "description": "Task priority",
                "enum": ["low", "medium", "high"],
                "default": "medium"
            },
            "days_until_due": {
                "type": "integer",
                "description": "Number of days from now for due date",
                "default": 3
            }
        },
        "function": create_task_from_document_summary
    },
    {
        "name": "schedule_document_review",
        "description": "Schedule time blocks for reviewing multiple documents",
        "parameters": {
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of document IDs to review"
            },
            "review_type": {
                "type": "string",
                "description": "Type of review",
                "enum": ["quick", "detailed", "deep"],
                "default": "detailed"
            },
            "time_per_doc": {
                "type": "integer",
                "description": "Minutes to allocate per document",
                "default": 30
            }
        },
        "function": schedule_document_review
    },
    {
        "name": "smart_document_assistant",
        "description": "Smart assistant for document operations with natural language",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Natural language query about documents"
            },
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["search", "summarize", "schedule", "create_task"],
                "default": "search"
            }
        },
        "function": smart_document_assistant
    }
]