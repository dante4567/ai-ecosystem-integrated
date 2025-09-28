"""
OpenWebUI Functions for Organizer Service Only
Configure OpenWebUI to use just the organizer service for personal assistant tasks
"""

import json
import requests
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Configuration
ORGANIZER_SERVICE_URL = "http://localhost:8002"

def ask_assistant(message: str) -> str:
    """
    Ask the personal assistant a question or give it a command.

    Args:
        message: Natural language message for the assistant

    Returns:
        Assistant response
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

            result = f"🤖 **Assistant Response:**\n\n{assistant_response}\n\n"

            # Show actions taken if any
            if "actions_taken" in data and data["actions_taken"]:
                result += "🎯 **Actions Taken:**\n"
                for action in data["actions_taken"]:
                    action_type = action.get("type", "action")
                    details = action.get("details", "")
                    result += f"• {action_type}: {details}\n"

            return result
        else:
            return f"❌ Assistant request failed with status {response.status_code}"

    except Exception as e:
        return f"❌ Assistant error: {str(e)}"

def get_my_tasks(limit: int = 10) -> str:
    """
    Get your pending tasks and todos.

    Args:
        limit: Maximum number of tasks to return

    Returns:
        Formatted list of tasks
    """
    try:
        response = requests.get(
            f"{ORGANIZER_SERVICE_URL}/api/todos",
            params={"limit": limit},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            todos = data.get("todos", [])

            if not todos:
                return "✅ No pending tasks! You're all caught up."

            result = "✅ **Your Pending Tasks:**\n\n"

            for i, todo in enumerate(todos, 1):
                title = todo.get("title", "Untitled")
                priority = todo.get("priority", "medium")
                due_date = todo.get("due_date", "")
                created = todo.get("created_at", "")

                # Priority emoji
                priority_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(priority, "⚪")

                result += f"{priority_emoji} **{i}. {title}**\n"
                result += f"   Priority: {priority.title()}\n"

                if due_date:
                    result += f"   📅 Due: {due_date}\n"

                result += "\n"

            return result
        else:
            return f"❌ Failed to get tasks: {response.status_code}"

    except Exception as e:
        return f"❌ Task retrieval error: {str(e)}"

def get_my_events(days: int = 7) -> str:
    """
    Get your upcoming calendar events.

    Args:
        days: Number of days ahead to look

    Returns:
        Formatted list of events
    """
    try:
        response = requests.get(
            f"{ORGANIZER_SERVICE_URL}/api/events",
            params={"days": days},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            events = data.get("events", [])

            if not events:
                return f"📅 No events scheduled for the next {days} days."

            result = f"📅 **Upcoming Events (Next {days} Days):**\n\n"

            for i, event in enumerate(events, 1):
                title = event.get("title", "Untitled")
                start_time = event.get("start_time", "")
                end_time = event.get("end_time", "")
                location = event.get("location", "")
                description = event.get("description", "")

                result += f"**{i}. {title}**\n"
                result += f"   🕐 Start: {start_time}\n"

                if end_time:
                    result += f"   🕑 End: {end_time}\n"

                if location:
                    result += f"   📍 Location: {location}\n"

                if description:
                    desc_short = description[:100] + "..." if len(description) > 100 else description
                    result += f"   📝 Description: {desc_short}\n"

                result += "\n"

            return result
        else:
            return f"❌ Failed to get events: {response.status_code}"

    except Exception as e:
        return f"❌ Event retrieval error: {str(e)}"

def get_my_contacts(search: str = "", limit: int = 10) -> str:
    """
    Get your contacts, optionally filtered by search term.

    Args:
        search: Optional search term to filter contacts
        limit: Maximum number of contacts to return

    Returns:
        Formatted list of contacts
    """
    try:
        params = {"limit": limit}
        if search:
            params["search"] = search

        response = requests.get(
            f"{ORGANIZER_SERVICE_URL}/api/contacts",
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            contacts = data.get("contacts", [])

            if not contacts:
                search_text = f" matching '{search}'" if search else ""
                return f"👥 No contacts found{search_text}."

            search_text = f" (Search: '{search}')" if search else ""
            result = f"👥 **Your Contacts{search_text}:**\n\n"

            for i, contact in enumerate(contacts, 1):
                name = contact.get("name", "Unknown")
                email = contact.get("email", "")
                phone = contact.get("phone", "")
                company = contact.get("company", "")
                notes = contact.get("notes", "")

                result += f"👤 **{i}. {name}**\n"

                if email:
                    result += f"   ✉️ Email: {email}\n"

                if phone:
                    result += f"   📞 Phone: {phone}\n"

                if company:
                    result += f"   🏢 Company: {company}\n"

                if notes:
                    notes_short = notes[:100] + "..." if len(notes) > 100 else notes
                    result += f"   📝 Notes: {notes_short}\n"

                result += "\n"

            return result
        else:
            return f"❌ Failed to get contacts: {response.status_code}"

    except Exception as e:
        return f"❌ Contact retrieval error: {str(e)}"

def create_task(title: str, priority: str = "medium", due_date: str = "") -> str:
    """
    Create a new task.

    Args:
        title: Task title/description
        priority: Task priority (low, medium, high)
        due_date: Optional due date (YYYY-MM-DD format)

    Returns:
        Task creation result
    """
    try:
        payload = {
            "title": title,
            "priority": priority
        }

        if due_date:
            payload["due_date"] = due_date

        response = requests.post(
            f"{ORGANIZER_SERVICE_URL}/api/todos",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            task_id = data.get("id", "unknown")

            result = f"✅ **Task Created Successfully!**\n\n"
            result += f"📝 Title: {title}\n"
            result += f"🎯 Priority: {priority.title()}\n"

            if due_date:
                result += f"📅 Due: {due_date}\n"

            result += f"🆔 Task ID: {task_id}"

            return result
        else:
            return f"❌ Failed to create task: {response.status_code}"

    except Exception as e:
        return f"❌ Task creation error: {str(e)}"

def create_contact(name: str, email: str = "", phone: str = "", company: str = "") -> str:
    """
    Create a new contact.

    Args:
        name: Contact name
        email: Contact email (optional)
        phone: Contact phone (optional)
        company: Contact company (optional)

    Returns:
        Contact creation result
    """
    try:
        payload = {"name": name}

        if email:
            payload["email"] = email
        if phone:
            payload["phone"] = phone
        if company:
            payload["company"] = company

        response = requests.post(
            f"{ORGANIZER_SERVICE_URL}/api/contacts",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            contact_id = data.get("id", "unknown")

            result = f"👤 **Contact Created Successfully!**\n\n"
            result += f"📝 Name: {name}\n"

            if email:
                result += f"✉️ Email: {email}\n"

            if phone:
                result += f"📞 Phone: {phone}\n"

            if company:
                result += f"🏢 Company: {company}\n"

            result += f"🆔 Contact ID: {contact_id}"

            return result
        else:
            return f"❌ Failed to create contact: {response.status_code}"

    except Exception as e:
        return f"❌ Contact creation error: {str(e)}"

def get_organizer_stats() -> str:
    """
    Get organizer service statistics and health information.

    Returns:
        Service statistics
    """
    try:
        response = requests.get(f"{ORGANIZER_SERVICE_URL}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()

            result = "📊 **Organizer Service Status:**\n\n"
            result += f"✅ Status: {data.get('status', 'unknown')}\n"
            result += f"🐳 Docker: {data.get('docker', False)}\n"
            result += f"🗄️ Database: {data.get('database', 'unknown')}\n"
            result += f"📧 Email: {data.get('email_enabled', False)}\n"
            result += f"📁 File Monitor: {data.get('file_monitor', False)}\n\n"

            # Get counts
            try:
                stats_response = requests.get(f"{ORGANIZER_SERVICE_URL}/api/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    result += "📈 **Counts:**\n"
                    result += f"✅ Tasks: {stats.get('total_todos', 0)}\n"
                    result += f"📅 Events: {stats.get('total_events', 0)}\n"
                    result += f"👥 Contacts: {stats.get('total_contacts', 0)}\n"
            except:
                pass

            return result
        else:
            return f"❌ Health check failed with status {response.status_code}"

    except Exception as e:
        return f"❌ Health check error: {str(e)}"

def todays_overview() -> str:
    """
    Get today's overview with tasks and events.

    Returns:
        Today's overview
    """
    result = "📅 **Today's Overview:**\n\n"

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

            result += "🗓️ **Today's Events:**\n"
            if events:
                for event in events[:3]:
                    title = event.get("title", "Untitled")
                    start_time = event.get("start_time", "")
                    result += f"• {title} at {start_time}\n"
            else:
                result += "• No events scheduled\n"
        else:
            result += "🗓️ Events: Unable to fetch\n"

    except:
        result += "🗓️ Events: Service unavailable\n"

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

            result += "✅ **Pending Tasks:**\n"
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

    return result

# OpenWebUI Function Registration
def get_openwebui_functions():
    """Return OpenWebUI function definitions"""
    return {
        "ask_assistant": {
            "function": ask_assistant,
            "description": "Ask the personal assistant anything",
            "parameters": {
                "message": {"type": "str", "description": "Natural language message"}
            }
        },
        "get_my_tasks": {
            "function": get_my_tasks,
            "description": "Get your pending tasks",
            "parameters": {
                "limit": {"type": "int", "description": "Max tasks to return (default: 10)"}
            }
        },
        "get_my_events": {
            "function": get_my_events,
            "description": "Get upcoming calendar events",
            "parameters": {
                "days": {"type": "int", "description": "Days ahead to look (default: 7)"}
            }
        },
        "get_my_contacts": {
            "function": get_my_contacts,
            "description": "Get your contacts",
            "parameters": {
                "search": {"type": "str", "description": "Optional search term"},
                "limit": {"type": "int", "description": "Max contacts (default: 10)"}
            }
        },
        "create_task": {
            "function": create_task,
            "description": "Create a new task",
            "parameters": {
                "title": {"type": "str", "description": "Task title"},
                "priority": {"type": "str", "description": "Priority: low, medium, high"},
                "due_date": {"type": "str", "description": "Due date (YYYY-MM-DD)"}
            }
        },
        "create_contact": {
            "function": create_contact,
            "description": "Create a new contact",
            "parameters": {
                "name": {"type": "str", "description": "Contact name"},
                "email": {"type": "str", "description": "Email address"},
                "phone": {"type": "str", "description": "Phone number"},
                "company": {"type": "str", "description": "Company name"}
            }
        },
        "get_organizer_stats": {
            "function": get_organizer_stats,
            "description": "Get organizer service statistics",
            "parameters": {}
        },
        "todays_overview": {
            "function": todays_overview,
            "description": "Get today's overview",
            "parameters": {}
        }
    }