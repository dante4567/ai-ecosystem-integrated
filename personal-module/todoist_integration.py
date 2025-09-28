#!/usr/bin/env python3

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from todoist_api_python.api import TodoistAPI
from todoist_api_python.models import Task as TodoistTask, Project
import pytz
from dateutil import parser

logger = logging.getLogger(__name__)

class TodoistIntegration:
    def __init__(self):
        self.api_token = os.getenv("TODOIST_API_KEY")
        self.api = None
        self.enabled = bool(self.api_token and self.api_token.strip())
        self.default_project_id = None
        self.sync_enabled = os.getenv("TODOIST_SYNC_ENABLED", "true").lower() == "true"

    async def initialize(self):
        """Initialize Todoist API connection"""
        if not self.enabled:
            logger.warning("Todoist integration disabled - no API key provided")
            return False

        try:
            self.api = TodoistAPI(self.api_token)

            # Test connection and get default project
            projects = await asyncio.get_event_loop().run_in_executor(
                None, self.api.get_projects
            )

            if projects:
                # Find or create AI Ecosystem project
                ai_project = None
                for project in projects:
                    if project.name == "AI Ecosystem":
                        ai_project = project
                        break

                if not ai_project:
                    ai_project = await asyncio.get_event_loop().run_in_executor(
                        None, self.api.add_project, {"name": "AI Ecosystem", "color": "blue"}
                    )

                self.default_project_id = ai_project.id

            logger.info("Todoist integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Todoist integration: {e}")
            self.enabled = False
            return False

    async def create_task(self, task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a task in Todoist"""
        if not self.enabled:
            return None

        try:
            # Prepare task data for Todoist
            todoist_task_data = {
                "content": task_data.get("title", ""),
                "project_id": self.default_project_id,
            }

            # Add description if provided
            if task_data.get("description"):
                todoist_task_data["description"] = task_data["description"]

            # Add due date if provided
            if task_data.get("due_date"):
                due_date = task_data["due_date"]
                if isinstance(due_date, str):
                    due_date = parser.parse(due_date)
                todoist_task_data["due_date"] = due_date.strftime("%Y-%m-%d")

            # Add priority (Todoist uses 1-4, we use low/medium/high/urgent)
            priority_mapping = {
                "low": 1,
                "medium": 2,
                "high": 3,
                "urgent": 4
            }
            if task_data.get("priority"):
                todoist_task_data["priority"] = priority_mapping.get(
                    task_data["priority"].lower(), 2
                )

            # Add labels/tags
            if task_data.get("tags"):
                todoist_task_data["labels"] = task_data["tags"]

            # Create task in Todoist
            task = await asyncio.get_event_loop().run_in_executor(
                None, self.api.add_task, todoist_task_data
            )

            return {
                "todoist_id": task.id,
                "url": task.url,
                "created_at": task.created_at,
                "project_id": task.project_id
            }

        except Exception as e:
            logger.error(f"Failed to create Todoist task: {e}")
            return None

    async def update_task(self, todoist_id: str, task_data: Dict[str, Any]) -> bool:
        """Update a task in Todoist"""
        if not self.enabled:
            return False

        try:
            update_data = {}

            if "title" in task_data:
                update_data["content"] = task_data["title"]

            if "description" in task_data:
                update_data["description"] = task_data["description"]

            if "due_date" in task_data and task_data["due_date"]:
                due_date = task_data["due_date"]
                if isinstance(due_date, str):
                    due_date = parser.parse(due_date)
                update_data["due_date"] = due_date.strftime("%Y-%m-%d")

            if "priority" in task_data:
                priority_mapping = {
                    "low": 1,
                    "medium": 2,
                    "high": 3,
                    "urgent": 4
                }
                update_data["priority"] = priority_mapping.get(
                    task_data["priority"].lower(), 2
                )

            if update_data:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.api.update_task, todoist_id, **update_data
                )

            return True

        except Exception as e:
            logger.error(f"Failed to update Todoist task {todoist_id}: {e}")
            return False

    async def complete_task(self, todoist_id: str) -> bool:
        """Mark a task as completed in Todoist"""
        if not self.enabled:
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.api.close_task, todoist_id
            )
            return True

        except Exception as e:
            logger.error(f"Failed to complete Todoist task {todoist_id}: {e}")
            return False

    async def delete_task(self, todoist_id: str) -> bool:
        """Delete a task from Todoist"""
        if not self.enabled:
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.api.delete_task, todoist_id
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete Todoist task {todoist_id}: {e}")
            return False

    async def sync_tasks_from_todoist(self) -> List[Dict[str, Any]]:
        """Fetch tasks from Todoist to sync with local database"""
        if not self.enabled or not self.sync_enabled:
            return []

        try:
            tasks = await asyncio.get_event_loop().run_in_executor(
                None, self.api.get_tasks, {"project_id": self.default_project_id}
            )

            synced_tasks = []
            for task in tasks:
                # Convert Todoist task to our format
                local_task = {
                    "todoist_id": task.id,
                    "title": task.content,
                    "description": getattr(task, "description", ""),
                    "status": "completed" if task.is_completed else "pending",
                    "priority": self._convert_todoist_priority(task.priority),
                    "due_date": self._parse_todoist_due_date(task.due),
                    "created_at": task.created_at,
                    "completed_at": task.completed_at if task.is_completed else None,
                    "tags": getattr(task, "labels", []),
                    "metadata": {
                        "todoist_url": task.url,
                        "project_id": task.project_id,
                        "section_id": getattr(task, "section_id", None)
                    }
                }

                synced_tasks.append(local_task)

            logger.info(f"Synced {len(synced_tasks)} tasks from Todoist")
            return synced_tasks

        except Exception as e:
            logger.error(f"Failed to sync tasks from Todoist: {e}")
            return []

    async def get_projects(self) -> List[Dict[str, Any]]:
        """Get all Todoist projects"""
        if not self.enabled:
            return []

        try:
            projects = await asyncio.get_event_loop().run_in_executor(
                None, self.api.get_projects
            )

            return [
                {
                    "id": project.id,
                    "name": project.name,
                    "color": project.color,
                    "url": project.url
                }
                for project in projects
            ]

        except Exception as e:
            logger.error(f"Failed to get Todoist projects: {e}")
            return []

    def _convert_todoist_priority(self, priority: int) -> str:
        """Convert Todoist priority (1-4) to our priority system"""
        priority_map = {
            1: "low",
            2: "medium",
            3: "high",
            4: "urgent"
        }
        return priority_map.get(priority, "medium")

    def _parse_todoist_due_date(self, due_info) -> Optional[datetime]:
        """Parse Todoist due date information"""
        if not due_info:
            return None

        try:
            if hasattr(due_info, "date"):
                return parser.parse(due_info.date)
            elif hasattr(due_info, "datetime"):
                return parser.parse(due_info.datetime)
            else:
                return None
        except Exception:
            return None

    async def webhook_handler(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Todoist webhook events"""
        if not self.enabled:
            return {"status": "disabled"}

        try:
            event_name = event_data.get("event_name", "")
            event_data_payload = event_data.get("event_data", {})

            if event_name == "item:added":
                return await self._handle_task_added(event_data_payload)
            elif event_name == "item:updated":
                return await self._handle_task_updated(event_data_payload)
            elif event_name == "item:completed":
                return await self._handle_task_completed(event_data_payload)
            elif event_name == "item:deleted":
                return await self._handle_task_deleted(event_data_payload)

            return {"status": "ignored", "event": event_name}

        except Exception as e:
            logger.error(f"Webhook handler error: {e}")
            return {"status": "error", "error": str(e)}

    async def _handle_task_added(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task added event from Todoist"""
        # This would integrate with the personal module's task creation
        return {"status": "task_added", "todoist_id": task_data.get("id")}

    async def _handle_task_updated(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task updated event from Todoist"""
        return {"status": "task_updated", "todoist_id": task_data.get("id")}

    async def _handle_task_completed(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task completed event from Todoist"""
        return {"status": "task_completed", "todoist_id": task_data.get("id")}

    async def _handle_task_deleted(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task deleted event from Todoist"""
        return {"status": "task_deleted", "todoist_id": task_data.get("id")}

    def is_enabled(self) -> bool:
        """Check if Todoist integration is enabled"""
        return self.enabled

    async def test_connection(self) -> Dict[str, Any]:
        """Test the Todoist API connection"""
        if not self.enabled:
            return {"status": "disabled", "message": "No API key provided"}

        try:
            projects = await asyncio.get_event_loop().run_in_executor(
                None, self.api.get_projects
            )

            return {
                "status": "connected",
                "project_count": len(projects),
                "default_project_id": self.default_project_id
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Global Todoist integration instance
todoist = TodoistIntegration()