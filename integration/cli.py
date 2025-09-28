#!/usr/bin/env python3

"""
AI Ecosystem CLI Tool

Command-line interface for interacting with the AI ecosystem services.
Uses Click for commands and provides easy access to all service functionalities.
"""

import click
import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import sys

# Service URLs
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
PERSONAL_SERVICE_URL = os.getenv("PERSONAL_SERVICE_URL", "http://localhost:8002")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8003")
CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:8000")

class EcosystemClient:
    def __init__(self):
        self.session = None

    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=10)
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

def run_async(func):
    """Decorator to run async functions in Click commands"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

@click.group()
@click.version_option("1.0.0")
def cli():
    """AI Ecosystem CLI - Command-line interface for the AI ecosystem services."""
    pass

@cli.command()
@click.argument('query', required=True)
@click.option('--service', '-s', type=click.Choice(['gateway', 'rag', 'personal', 'all']), default='gateway',
              help='Which service to query (default: gateway for unified response)')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (default: text)')
@click.option('--max-results', '-n', type=int, default=5,
              help='Maximum number of results (default: 5)')
@run_async
async def ask(query, service, format, max_results):
    """Ask a question to the AI ecosystem services."""
    client = EcosystemClient()

    try:
        if service == 'gateway' or service == 'all':
            # Use the unified assistant
            response = await client.make_request(
                f"{GATEWAY_URL}/api/assistant",
                "POST",
                {"message": query, "include_actions": True}
            )

            if format == 'json':
                click.echo(json.dumps(response, indent=2))
            else:
                if "error" in response:
                    click.echo(f"❌ Error: {response['error']}", err=True)
                else:
                    click.echo(f"🤖 Assistant Response:")
                    click.echo(response.get("response", "No response"))

                    actions = response.get("actions_taken", [])
                    if actions:
                        click.echo(f"\n📋 Actions taken: {len(actions)}")
                        for action in actions:
                            click.echo(f"  • {action.get('action', 'unknown')}")

                    sources = response.get("data_sources", [])
                    if sources:
                        click.echo(f"\n📊 Data sources: {', '.join(sources)}")

        elif service == 'rag':
            # Search RAG service directly
            response = await client.make_request(
                f"{RAG_SERVICE_URL}/search",
                "POST",
                {"query": query, "max_results": max_results}
            )

            if format == 'json':
                click.echo(json.dumps(response, indent=2))
            else:
                if "error" in response:
                    click.echo(f"❌ Error: {response['error']}", err=True)
                else:
                    results = response.get("results", [])
                    click.echo(f"📚 RAG Search Results ({len(results)} found):")
                    for i, result in enumerate(results, 1):
                        title = result.get("title", result.get("id", "Unknown"))
                        score = result.get("similarity_score", result.get("relevance_score", 0))
                        content = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
                        click.echo(f"\n{i}. {title} (Score: {score:.2f})")
                        click.echo(f"   {content}")

        elif service == 'personal':
            # Check if query is about tasks
            if any(word in query.lower() for word in ['task', 'todo', 'schedule', 'calendar']):
                if any(word in query.lower() for word in ['create', 'add', 'new']):
                    # Create a task
                    task_title = query
                    for word in ['create', 'add', 'new', 'task', 'todo']:
                        task_title = task_title.replace(word, '').strip()

                    response = await client.make_request(
                        f"{PERSONAL_SERVICE_URL}/tasks",
                        "POST",
                        {"title": task_title, "priority": "medium"}
                    )

                    if format == 'json':
                        click.echo(json.dumps(response, indent=2))
                    else:
                        if "error" in response:
                            click.echo(f"❌ Error: {response['error']}", err=True)
                        else:
                            click.echo(f"✅ Created task: {response.get('title', task_title)}")
                            click.echo(f"   ID: {response.get('id', 'unknown')}")
                else:
                    # Show tasks
                    response = await client.make_request(f"{PERSONAL_SERVICE_URL}/tasks")

                    if format == 'json':
                        click.echo(json.dumps(response, indent=2))
                    else:
                        if "error" in response:
                            click.echo(f"❌ Error: {response['error']}", err=True)
                        else:
                            tasks = response.get("tasks", [])
                            pending = [t for t in tasks if t.get("status") != "completed"]
                            click.echo(f"📝 Personal Tasks ({len(pending)} pending):")
                            for i, task in enumerate(pending[:max_results], 1):
                                title = task.get("title", "Untitled")
                                status = task.get("status", "unknown")
                                due_date = task.get("due_date", "No due date")
                                click.echo(f"{i}. {title}")
                                click.echo(f"   Status: {status}, Due: {due_date}")
            else:
                # Get general stats
                response = await client.make_request(f"{PERSONAL_SERVICE_URL}/stats")

                if format == 'json':
                    click.echo(json.dumps(response, indent=2))
                else:
                    if "error" in response:
                        click.echo(f"❌ Error: {response['error']}", err=True)
                    else:
                        summary = response.get("summary", {})
                        click.echo(f"📊 Personal Statistics:")
                        click.echo(f"   Total tasks: {summary.get('total_tasks', 0)}")
                        click.echo(f"   Completion rate: {summary.get('completion_rate', 0):.1f}%")
                        click.echo(f"   Recent activity: {summary.get('recent_activity', 0)} tasks")

    except Exception as e:
        click.echo(f"❌ Request failed: {str(e)}", err=True)
    finally:
        await client.close()

@cli.command()
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (default: text)')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed health information')
@run_async
async def status(format, detailed):
    """Check health status of all services."""
    client = EcosystemClient()

    services = {
        "Gateway": GATEWAY_URL,
        "RAG Service": RAG_SERVICE_URL,
        "Personal Service": PERSONAL_SERVICE_URL,
        "ChromaDB": CHROMA_URL
    }

    try:
        health_results = {}

        for service_name, service_url in services.items():
            if service_name == "ChromaDB":
                health_url = f"{service_url}/api/v1/heartbeat"
            else:
                health_url = f"{service_url}/health"

            response = await client.make_request(health_url)
            health_results[service_name] = response

        if format == 'json':
            click.echo(json.dumps(health_results, indent=2, default=str))
        else:
            click.echo("🏥 AI Ecosystem Health Status")
            click.echo("=" * 40)

            all_healthy = True

            for service_name, result in health_results.items():
                if "error" in result:
                    status_emoji = "🔴"
                    status_text = "UNHEALTHY"
                    all_healthy = False
                else:
                    status_emoji = "🟢"
                    status_text = "HEALTHY"

                click.echo(f"{status_emoji} {service_name}: {status_text}")

                if detailed:
                    if "error" not in result:
                        if "service" in result:
                            click.echo(f"   Service: {result['service']}")
                        if "version" in result:
                            click.echo(f"   Version: {result['version']}")
                        if "response_time" in result:
                            click.echo(f"   Response time: {result['response_time']}")
                        if "timestamp" in result:
                            click.echo(f"   Last check: {result['timestamp']}")
                    else:
                        click.echo(f"   Error: {result['error']}")
                    click.echo()

            click.echo("=" * 40)
            if all_healthy:
                click.echo("✅ All services are healthy!")
            else:
                click.echo("⚠️  Some services are experiencing issues")

            # Get unified service status if gateway is healthy
            if "Gateway" in health_results and "error" not in health_results["Gateway"]:
                gateway_status = await client.make_request(f"{GATEWAY_URL}/api/services")
                if "error" not in gateway_status and detailed:
                    click.echo("\n🔗 Service Connectivity:")
                    services_info = gateway_status.get("services", {})
                    for svc_name, svc_info in services_info.items():
                        health = svc_info.get("health", {})
                        status_text = health.get("status", "unknown")
                        capabilities = svc_info.get("capabilities", [])
                        click.echo(f"   {svc_name}: {status_text}")
                        if capabilities:
                            click.echo(f"      Capabilities: {', '.join(capabilities)}")

    except Exception as e:
        click.echo(f"❌ Status check failed: {str(e)}", err=True)
    finally:
        await client.close()

@cli.command()
@click.option('--date', '-d', help='Date in YYYY-MM-DD format (default: today)')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (default: text)')
@click.option('--include-docs', is_flag=True, help='Include relevant documents')
@run_async
async def today(date, format, include_docs):
    """Show today's events and relevant documents."""
    client = EcosystemClient()

    target_date = date or datetime.now().strftime("%Y-%m-%d")

    try:
        # Get today's schedule
        schedule_response = await client.make_request(f"{PERSONAL_SERVICE_URL}/schedule")

        # Get tasks
        tasks_response = await client.make_request(f"{PERSONAL_SERVICE_URL}/tasks")

        if format == 'json':
            result = {
                "date": target_date,
                "schedule": schedule_response,
                "tasks": tasks_response
            }

            if include_docs:
                # Search for relevant documents
                search_response = await client.make_request(
                    f"{RAG_SERVICE_URL}/search",
                    "POST",
                    {"query": "daily tasks work", "max_results": 3}
                )
                result["relevant_documents"] = search_response

            click.echo(json.dumps(result, indent=2, default=str))
        else:
            click.echo(f"📅 Today's Overview - {target_date}")
            click.echo("=" * 50)

            # Show schedule
            if "error" not in schedule_response:
                schedule = schedule_response.get("schedule", {})
                day_info = schedule.get(target_date, {"tasks": [], "events": [], "total_items": 0})

                tasks_today = day_info.get("tasks", [])
                events_today = day_info.get("events", [])

                if tasks_today or events_today:
                    click.echo(f"📋 Scheduled Items ({len(tasks_today + events_today)} total):")

                    # Show events
                    for event in events_today:
                        start_time = event.get("start_time", "")
                        title = event.get("title", "Untitled Event")
                        click.echo(f"  📅 {start_time} - {title}")

                    # Show tasks
                    for task in tasks_today:
                        title = task.get("title", "Untitled Task")
                        priority = task.get("priority", "medium")
                        priority_emoji = {"urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                        due_time = task.get("due_time", "")
                        time_info = f" at {due_time}" if due_time else ""
                        click.echo(f"  {priority_emoji} {title}{time_info}")
                else:
                    click.echo("📋 No scheduled items for today")
            else:
                click.echo(f"❌ Could not retrieve schedule: {schedule_response.get('error', 'Unknown error')}")

            # Show task summary
            if "error" not in tasks_response:
                tasks = tasks_response.get("tasks", [])
                pending_tasks = [t for t in tasks if t.get("status") != "completed"]

                # Check for overdue
                today = datetime.now().date()
                overdue_tasks = []
                for task in pending_tasks:
                    due_date_str = task.get("due_date")
                    if due_date_str:
                        try:
                            from dateutil import parser
                            due_date = parser.parse(due_date_str).date()
                            if due_date < today:
                                overdue_tasks.append(task)
                        except:
                            pass

                click.echo(f"\n📊 Task Summary:")
                click.echo(f"  • Total pending: {len(pending_tasks)}")
                if overdue_tasks:
                    click.echo(f"  • ⚠️ Overdue: {len(overdue_tasks)}")
                    click.echo(f"\n🚨 Overdue Tasks:")
                    for task in overdue_tasks[:3]:
                        title = task.get("title", "Untitled")
                        due_date = task.get("due_date", "")
                        click.echo(f"    • {title} (Due: {due_date})")

            # Include relevant documents if requested
            if include_docs:
                click.echo(f"\n📚 Relevant Documents:")
                try:
                    search_response = await client.make_request(
                        f"{RAG_SERVICE_URL}/search",
                        "POST",
                        {"query": "daily work productivity", "max_results": 3}
                    )

                    if "error" not in search_response and search_response.get("results"):
                        for doc in search_response["results"]:
                            title = doc.get("title", doc.get("id", "Unknown"))
                            score = doc.get("similarity_score", doc.get("relevance_score", 0))
                            click.echo(f"  📄 {title} (Relevance: {score:.2f})")
                    else:
                        click.echo("  No relevant documents found")
                except Exception as e:
                    click.echo(f"  ❌ Could not retrieve documents: {str(e)}")

            click.echo("\n💪 Have a productive day!")

    except Exception as e:
        click.echo(f"❌ Failed to get today's overview: {str(e)}", err=True)
    finally:
        await client.close()

@cli.command()
@click.argument('query')
@click.option('--max-results', '-n', type=int, default=5, help='Maximum results to return')
@run_async
async def search(query, max_results):
    """Search across all services for comprehensive results."""
    client = EcosystemClient()

    try:
        # Use the gateway's unified search
        response = await client.make_request(
            f"{GATEWAY_URL}/api/search",
            "POST",
            {
                "query": query,
                "max_results": max_results,
                "include_personal": True,
                "include_context": True
            }
        )

        if "error" in response:
            click.echo(f"❌ Search failed: {response['error']}", err=True)
            return

        unified_results = response.get("unified_results", [])
        processing_time = response.get("processing_time", 0)

        click.echo(f"🔍 Unified Search Results for '{query}'")
        click.echo(f"Found {len(unified_results)} results in {processing_time:.2f}s")
        click.echo("=" * 60)

        for i, result in enumerate(unified_results, 1):
            source = result.get("source", "unknown")
            source_emoji = "📚" if source == "knowledge_base" else "📝"
            title = result.get("title", "Unknown")
            relevance = result.get("relevance_score", 0)

            click.echo(f"{i}. {source_emoji} {title} (Score: {relevance:.2f})")

            if result.get("content"):
                content = result["content"][:150] + "..." if len(result.get("content", "")) > 150 else result.get("content", "")
                click.echo(f"   {content}")

            # Show personal connections
            connections = result.get("personal_connections", [])
            if connections:
                click.echo(f"   🔗 Related: {len(connections)} personal items")

            click.echo()

    except Exception as e:
        click.echo(f"❌ Search error: {str(e)}", err=True)
    finally:
        await client.close()

if __name__ == "__main__":
    cli()