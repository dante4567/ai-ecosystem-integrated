#!/usr/bin/env python3

import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personal Module",
    description="Personal assistant and task management service",
    version="1.0.0"
)

PORT = int(os.getenv("PORT", 8002))

# Sample personal data
TASKS = [
    {"id": 1, "title": "Review AI integration", "completed": False, "due_date": "2025-09-28"},
    {"id": 2, "title": "Update documentation", "completed": False, "due_date": "2025-09-29"},
    {"id": 3, "title": "Test Docker setup", "completed": True, "due_date": "2025-09-27"}
]

PREFERENCES = {
    "timezone": "UTC",
    "notification_style": "email",
    "default_llm": "openai",
    "max_context_length": 4000
}

class TaskRequest(BaseModel):
    title: str
    due_date: str = None
    priority: str = "medium"

class TaskResponse(BaseModel):
    id: int
    title: str
    completed: bool
    due_date: str
    created_at: str

class PreferenceUpdate(BaseModel):
    key: str
    value: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "Personal Module",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "active_tasks": len([t for t in TASKS if not t["completed"]])
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Personal Module API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "preferences": "/preferences",
            "schedule": "/schedule"
        }
    }

@app.get("/tasks")
async def get_tasks():
    """Get all tasks"""
    return {
        "tasks": TASKS,
        "summary": {
            "total": len(TASKS),
            "completed": len([t for t in TASKS if t["completed"]]),
            "pending": len([t for t in TASKS if not t["completed"]])
        }
    }

@app.post("/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    """Create a new task"""
    new_id = max([t["id"] for t in TASKS], default=0) + 1
    new_task = {
        "id": new_id,
        "title": task.title,
        "completed": False,
        "due_date": task.due_date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "priority": task.priority
    }

    TASKS.append(new_task)

    return TaskResponse(
        id=new_task["id"],
        title=new_task["title"],
        completed=new_task["completed"],
        due_date=new_task["due_date"],
        created_at=datetime.now().isoformat()
    )

@app.put("/tasks/{task_id}/complete")
async def complete_task(task_id: int):
    """Mark a task as completed"""
    for task in TASKS:
        if task["id"] == task_id:
            task["completed"] = True
            return {
                "message": f"Task {task_id} marked as completed",
                "task": task,
                "timestamp": datetime.now().isoformat()
            }

    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/preferences")
async def get_preferences():
    """Get user preferences"""
    return {
        "preferences": PREFERENCES,
        "last_updated": datetime.now().isoformat()
    }

@app.put("/preferences")
async def update_preference(update: PreferenceUpdate):
    """Update a user preference"""
    PREFERENCES[update.key] = update.value
    return {
        "message": f"Preference {update.key} updated",
        "preferences": PREFERENCES,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/schedule")
async def get_schedule():
    """Get today's schedule"""
    today = datetime.now().strftime("%Y-%m-%d")
    today_tasks = [t for t in TASKS if t.get("due_date") == today and not t["completed"]]

    return {
        "date": today,
        "tasks_due": today_tasks,
        "summary": f"{len(today_tasks)} tasks due today"
    }

@app.get("/stats")
async def get_stats():
    """Get personal productivity stats"""
    completed_today = len([t for t in TASKS if t["completed"] and t.get("due_date") == datetime.now().strftime("%Y-%m-%d")])

    return {
        "productivity_stats": {
            "tasks_completed_today": completed_today,
            "total_tasks": len(TASKS),
            "completion_rate": len([t for t in TASKS if t["completed"]]) / len(TASKS) * 100 if TASKS else 0
        },
        "preferences": PREFERENCES,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )