#!/usr/bin/env python3

import os
import json
import sys
sys.path.append('../shared-config')

from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import pytz
from dateutil import parser

from database import get_db, db_config
from models import Task, TaskStatus, TaskPriority, UserPreference, Schedule, Productivity
from todoist_integration import todoist
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Personal Module",
    description="Advanced personal assistant with task management, scheduling, and Todoist integration",
    version="2.0.0"
)

PORT = int(os.getenv("PORT", 8002))

# Default user for single-user mode
DEFAULT_USER_ID = "default_user"

# Sample initial tasks and preferences
INITIAL_TASKS = [
    {
        "title": "Review AI ecosystem integration",
        "description": "Evaluate the microservices architecture and integration points",
        "status": TaskStatus.PENDING,
        "priority": TaskPriority.HIGH,
        "due_date": datetime.now() + timedelta(days=1),
        "tags": ["ai", "integration", "review"]
    },
    {
        "title": "Update documentation",
        "description": "Document the new features and API endpoints",
        "status": TaskStatus.PENDING,
        "priority": TaskPriority.MEDIUM,
        "due_date": datetime.now() + timedelta(days=2),
        "tags": ["documentation", "api"]
    },
    {
        "title": "Test Docker setup",
        "description": "Verify all containers work correctly with new dependencies",
        "status": TaskStatus.COMPLETED,
        "priority": TaskPriority.HIGH,
        "due_date": datetime.now() - timedelta(days=1),
        "completed_at": datetime.now() - timedelta(hours=2),
        "tags": ["docker", "testing"]
    }
]

INITIAL_PREFERENCES = {
    "timezone": "UTC",
    "notification_style": "email",
    "default_llm": "openai",
    "max_context_length": 4000,
    "work_hours_start": "09:00",
    "work_hours_end": "17:00",
    "todoist_sync": "true",
    "theme": "light"
}

class TaskRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sync_to_todoist: bool = True

class TaskResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    status: TaskStatus
    priority: TaskPriority
    due_date: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    todoist_id: Optional[str]
    user_id: str

class TaskUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    due_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class PreferenceRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1)
    value_type: str = "string"

class PreferenceResponse(BaseModel):
    id: str
    key: str
    value: str
    value_type: str
    created_at: datetime
    updated_at: datetime

class ScheduleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    is_all_day: bool = False
    recurring_pattern: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ScheduleResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    is_all_day: bool
    recurring_pattern: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    user_id: str

class ProductivityStatsResponse(BaseModel):
    date: datetime
    tasks_completed: int
    tasks_created: int
    total_work_time: int
    focus_score: int
    completion_rate: float
    stats_data: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Enhanced Personal Module v2.0")

    # Initialize database
    if await db_config.initialize():
        await db_config.create_tables()
        logger.info("Database initialized")

    # Initialize Todoist integration
    if await todoist.initialize():
        logger.info("Todoist integration initialized")

    # Seed initial data if needed
    async with db_config.get_session() as session:
        await seed_initial_data(session)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await db_config.close()
    logger.info("Enhanced Personal Module shutdown complete")

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Enhanced health check endpoint"""
    start_time = datetime.now()

    # Check database connection
    db_healthy = False
    active_tasks = 0
    try:
        result = await db.execute(
            select(func.count(Task.id)).where(
                and_(Task.user_id == DEFAULT_USER_ID, Task.status != TaskStatus.COMPLETED)
            )
        )
        active_tasks = result.scalar() or 0
        db_healthy = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    # Check Todoist integration
    todoist_status = await todoist.test_connection()

    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()

    overall_status = "healthy" if db_healthy else "degraded"

    return {
        "service": "Enhanced Personal Module",
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "response_time": f"{response_time:.3f}s",
        "components": {
            "database": {
                "status": "healthy" if db_healthy else "unhealthy",
                "active_tasks": active_tasks
            },
            "todoist_integration": todoist_status
        }
    }

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "Enhanced Personal Module API",
        "version": "2.0.0",
        "description": "Advanced personal assistant with task management, scheduling, and Todoist integration",
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "tasks_create": "/tasks (POST)",
            "tasks_by_id": "/tasks/{task_id}",
            "preferences": "/preferences",
            "schedule": "/schedule",
            "productivity": "/productivity",
            "sync": "/sync",
            "stats": "/stats"
        },
        "features": [
            "PostgreSQL task persistence",
            "Todoist bi-directional sync",
            "Advanced filtering and search",
            "Productivity analytics",
            "Schedule management",
            "User preferences"
        ]
    }

async def seed_initial_data(db: AsyncSession = None):
    """Seed initial tasks and preferences if database is empty"""
    try:
        if not db:
            async with db_config.get_session() as session:
                db = session

        # Check if we already have data
        task_count = await db.execute(
            select(func.count(Task.id)).where(Task.user_id == DEFAULT_USER_ID)
        )
        if task_count.scalar() > 0:
            logger.info("Database already has tasks, skipping seed")
            return

        logger.info("Seeding initial personal data...")

        # Add initial tasks
        for task_data in INITIAL_TASKS:
            task = Task(
                title=task_data["title"],
                description=task_data["description"],
                status=task_data["status"],
                priority=task_data["priority"],
                due_date=task_data["due_date"],
                completed_at=task_data.get("completed_at"),
                user_id=DEFAULT_USER_ID,
                tags=task_data["tags"],
                metadata={"source": "initial_seed"}
            )
            db.add(task)

        # Add initial preferences
        for key, value in INITIAL_PREFERENCES.items():
            pref = UserPreference(
                user_id=DEFAULT_USER_ID,
                preference_key=key,
                preference_value=str(value),
                value_type="string"
            )
            db.add(pref)

        await db.commit()
        logger.info("Seeded initial personal data")

    except Exception as e:
        logger.error(f"Failed to seed initial data: {e}")

@app.get("/tasks")
async def get_tasks(
    status: Optional[TaskStatus] = None,
    priority: Optional[TaskPriority] = None,
    due_before: Optional[datetime] = None,
    due_after: Optional[datetime] = None,
    tags: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get tasks with advanced filtering and pagination"""
    try:
        # Build query with filters
        query = select(Task).where(Task.user_id == DEFAULT_USER_ID)

        if status:
            query = query.where(Task.status == status)
        if priority:
            query = query.where(Task.priority == priority)
        if due_before:
            query = query.where(Task.due_date <= due_before)
        if due_after:
            query = query.where(Task.due_date >= due_after)
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            query = query.where(Task.tags.op("&&")(tag_list))

        # Get total count
        total_query = select(func.count(Task.id)).where(Task.user_id == DEFAULT_USER_ID)
        if status:
            total_query = total_query.where(Task.status == status)
        total_result = await db.execute(total_query)
        total_count = total_result.scalar() or 0

        # Apply pagination and ordering
        query = query.order_by(Task.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        tasks = result.scalars().all()

        # Convert to response format
        task_responses = [
            TaskResponse(
                id=str(task.id),
                title=task.title,
                description=task.description,
                status=task.status,
                priority=task.priority,
                due_date=task.due_date,
                completed_at=task.completed_at,
                created_at=task.created_at,
                updated_at=task.updated_at,
                tags=task.tags or [],
                metadata=task.metadata or {},
                todoist_id=task.todoist_id,
                user_id=task.user_id
            )
            for task in tasks
        ]

        # Calculate summary statistics
        summary_stats = await get_task_summary(db)

        return {
            "tasks": task_responses,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "summary": summary_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tasks")

async def get_task_summary(db: AsyncSession) -> Dict[str, Any]:
    """Get task summary statistics"""
    try:
        # Count tasks by status
        status_counts = await db.execute(
            select(Task.status, func.count(Task.id))
            .where(Task.user_id == DEFAULT_USER_ID)
            .group_by(Task.status)
        )

        summary = {"total": 0, "by_status": {}, "by_priority": {}}

        for status, count in status_counts:
            summary["by_status"][status.value] = count
            summary["total"] += count

        # Count tasks by priority
        priority_counts = await db.execute(
            select(Task.priority, func.count(Task.id))
            .where(Task.user_id == DEFAULT_USER_ID)
            .group_by(Task.priority)
        )

        for priority, count in priority_counts:
            summary["by_priority"][priority.value] = count

        # Tasks due today
        today = datetime.now().date()
        due_today = await db.execute(
            select(func.count(Task.id))
            .where(
                and_(
                    Task.user_id == DEFAULT_USER_ID,
                    func.date(Task.due_date) == today,
                    Task.status != TaskStatus.COMPLETED
                )
            )
        )
        summary["due_today"] = due_today.scalar() or 0

        # Overdue tasks
        overdue = await db.execute(
            select(func.count(Task.id))
            .where(
                and_(
                    Task.user_id == DEFAULT_USER_ID,
                    Task.due_date < datetime.now(),
                    Task.status != TaskStatus.COMPLETED
                )
            )
        )
        summary["overdue"] = overdue.scalar() or 0

        return summary

    except Exception as e:
        logger.error(f"Error getting task summary: {e}")
        return {"total": 0, "by_status": {}, "by_priority": {}, "due_today": 0, "overdue": 0}

@app.post("/tasks", response_model=TaskResponse)
async def create_task(
    task: TaskRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a new task with optional Todoist sync"""
    try:
        # Create database record
        new_task = Task(
            title=task.title,
            description=task.description,
            status=task.status,
            priority=task.priority,
            due_date=task.due_date,
            user_id=DEFAULT_USER_ID,
            tags=task.tags,
            metadata=task.metadata
        )
        db.add(new_task)
        await db.flush()  # Get the ID

        # Sync to Todoist if enabled
        if task.sync_to_todoist and todoist.is_enabled():
            background_tasks.add_task(
                sync_task_to_todoist,
                str(new_task.id),
                {
                    "title": task.title,
                    "description": task.description,
                    "due_date": task.due_date,
                    "priority": task.priority.value,
                    "tags": task.tags
                }
            )

        await db.commit()
        await db.refresh(new_task)

        logger.info(f"Created task: {new_task.title} (ID: {new_task.id})")

        return TaskResponse(
            id=str(new_task.id),
            title=new_task.title,
            description=new_task.description,
            status=new_task.status,
            priority=new_task.priority,
            due_date=new_task.due_date,
            completed_at=new_task.completed_at,
            created_at=new_task.created_at,
            updated_at=new_task.updated_at,
            tags=new_task.tags or [],
            metadata=new_task.metadata or {},
            todoist_id=new_task.todoist_id,
            user_id=new_task.user_id
        )

    except Exception as e:
        logger.error(f"Error creating task: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create task")

@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific task by ID"""
    try:
        task_uuid = uuid.UUID(task_id)
        query = select(Task).where(
            and_(Task.id == task_uuid, Task.user_id == DEFAULT_USER_ID)
        )
        result = await db.execute(query)
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return TaskResponse(
            id=str(task.id),
            title=task.title,
            description=task.description,
            status=task.status,
            priority=task.priority,
            due_date=task.due_date,
            completed_at=task.completed_at,
            created_at=task.created_at,
            updated_at=task.updated_at,
            tags=task.tags or [],
            metadata=task.metadata or {},
            todoist_id=task.todoist_id,
            user_id=task.user_id
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    except Exception as e:
        logger.error(f"Error retrieving task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task")

@app.put("/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    update: TaskUpdateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Update an existing task"""
    try:
        task_uuid = uuid.UUID(task_id)
        query = select(Task).where(
            and_(Task.id == task_uuid, Task.user_id == DEFAULT_USER_ID)
        )
        result = await db.execute(query)
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Update fields
        if update.title is not None:
            task.title = update.title
        if update.description is not None:
            task.description = update.description
        if update.status is not None:
            # Handle completion logic
            if update.status == TaskStatus.COMPLETED and task.status != TaskStatus.COMPLETED:
                task.completed_at = datetime.utcnow()
            elif update.status != TaskStatus.COMPLETED:
                task.completed_at = None
            task.status = update.status
        if update.priority is not None:
            task.priority = update.priority
        if update.due_date is not None:
            task.due_date = update.due_date
        if update.tags is not None:
            task.tags = update.tags
        if update.metadata is not None:
            task.metadata = update.metadata

        task.updated_at = datetime.utcnow()

        # Sync to Todoist if task has todoist_id
        if task.todoist_id and todoist.is_enabled():
            background_tasks.add_task(
                update_todoist_task,
                task.todoist_id,
                {
                    "title": task.title,
                    "description": task.description,
                    "due_date": task.due_date,
                    "priority": task.priority.value,
                    "completed": task.status == TaskStatus.COMPLETED
                }
            )

        await db.commit()
        await db.refresh(task)

        logger.info(f"Updated task: {task.title} (ID: {task.id})")

        return TaskResponse(
            id=str(task.id),
            title=task.title,
            description=task.description,
            status=task.status,
            priority=task.priority,
            due_date=task.due_date,
            completed_at=task.completed_at,
            created_at=task.created_at,
            updated_at=task.updated_at,
            tags=task.tags or [],
            metadata=task.metadata or {},
            todoist_id=task.todoist_id,
            user_id=task.user_id
        )

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    except Exception as e:
        logger.error(f"Error updating task {task_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update task")

@app.put("/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Mark a task as completed"""
    try:
        task_uuid = uuid.UUID(task_id)
        query = select(Task).where(
            and_(Task.id == task_uuid, Task.user_id == DEFAULT_USER_ID)
        )
        result = await db.execute(query)
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()

        # Sync to Todoist
        if task.todoist_id and todoist.is_enabled():
            background_tasks.add_task(complete_todoist_task, task.todoist_id)

        await db.commit()
        await db.refresh(task)

        logger.info(f"Completed task: {task.title} (ID: {task.id})")

        return {
            "message": f"Task {task_id} marked as completed",
            "task_id": task_id,
            "completed_at": task.completed_at.isoformat(),
            "timestamp": datetime.now().isoformat()
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    except Exception as e:
        logger.error(f"Error completing task {task_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to complete task")

@app.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Delete a task"""
    try:
        task_uuid = uuid.UUID(task_id)
        query = select(Task).where(
            and_(Task.id == task_uuid, Task.user_id == DEFAULT_USER_ID)
        )
        result = await db.execute(query)
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Delete from Todoist if synced
        if task.todoist_id and todoist.is_enabled():
            background_tasks.add_task(delete_todoist_task, task.todoist_id)

        await db.delete(task)
        await db.commit()

        logger.info(f"Deleted task: {task.title} (ID: {task.id})")

        return {
            "message": f"Task {task_id} deleted successfully",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete task")

@app.get("/preferences")
async def get_preferences(db: AsyncSession = Depends(get_db)):
    """Get user preferences"""
    try:
        query = select(UserPreference).where(UserPreference.user_id == DEFAULT_USER_ID)
        result = await db.execute(query)
        preferences = result.scalars().all()

        pref_dict = {}
        for pref in preferences:
            # Convert value based on type
            value = pref.preference_value
            if pref.value_type == "boolean":
                value = value.lower() == "true"
            elif pref.value_type == "number":
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    pass
            elif pref.value_type == "json":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

            pref_dict[pref.preference_key] = value

        return {
            "preferences": pref_dict,
            "last_updated": datetime.now().isoformat(),
            "user_id": DEFAULT_USER_ID
        }

    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve preferences")

@app.put("/preferences")
async def update_preference(
    update: PreferenceRequest,
    db: AsyncSession = Depends(get_db)
):
    """Update a user preference"""
    try:
        # Check if preference exists
        query = select(UserPreference).where(
            and_(
                UserPreference.user_id == DEFAULT_USER_ID,
                UserPreference.preference_key == update.key
            )
        )
        result = await db.execute(query)
        preference = result.scalar_one_or_none()

        if preference:
            # Update existing preference
            preference.preference_value = update.value
            preference.value_type = update.value_type
            preference.updated_at = datetime.utcnow()
        else:
            # Create new preference
            preference = UserPreference(
                user_id=DEFAULT_USER_ID,
                preference_key=update.key,
                preference_value=update.value,
                value_type=update.value_type
            )
            db.add(preference)

        await db.commit()
        await db.refresh(preference)

        logger.info(f"Updated preference: {update.key} = {update.value}")

        return {
            "message": f"Preference {update.key} updated",
            "preference": {
                "key": preference.preference_key,
                "value": preference.preference_value,
                "type": preference.value_type
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating preference {update.key}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update preference")

@app.get("/schedule")
async def get_schedule(
    date: Optional[datetime] = None,
    days_ahead: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """Get schedule for a specific date or range"""
    try:
        if not date:
            date = datetime.now()

        start_date = date.date()
        end_date = start_date + timedelta(days=days_ahead)

        # Get tasks due in the date range
        tasks_query = select(Task).where(
            and_(
                Task.user_id == DEFAULT_USER_ID,
                func.date(Task.due_date) >= start_date,
                func.date(Task.due_date) < end_date,
                Task.status != TaskStatus.COMPLETED
            )
        ).order_by(Task.due_date, Task.priority.desc())

        result = await db.execute(tasks_query)
        tasks = result.scalars().all()

        # Get scheduled events
        schedule_query = select(Schedule).where(
            and_(
                Schedule.user_id == DEFAULT_USER_ID,
                func.date(Schedule.start_time) >= start_date,
                func.date(Schedule.start_time) < end_date
            )
        ).order_by(Schedule.start_time)

        schedule_result = await db.execute(schedule_query)
        events = schedule_result.scalars().all()

        # Group by date
        schedule_by_date = {}
        current_date = start_date
        while current_date < end_date:
            date_str = current_date.isoformat()

            # Tasks for this date
            date_tasks = [
                {
                    "id": str(task.id),
                    "title": task.title,
                    "priority": task.priority.value,
                    "due_time": task.due_date.time().isoformat() if task.due_date else None,
                    "type": "task"
                }
                for task in tasks
                if task.due_date and task.due_date.date() == current_date
            ]

            # Events for this date
            date_events = [
                {
                    "id": str(event.id),
                    "title": event.title,
                    "start_time": event.start_time.time().isoformat(),
                    "end_time": event.end_time.time().isoformat() if event.end_time else None,
                    "is_all_day": event.is_all_day,
                    "type": "event"
                }
                for event in events
                if event.start_time.date() == current_date
            ]

            schedule_by_date[date_str] = {
                "tasks": date_tasks,
                "events": date_events,
                "total_items": len(date_tasks) + len(date_events)
            }

            current_date += timedelta(days=1)

        return {
            "schedule": schedule_by_date,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days_ahead
            },
            "summary": {
                "total_tasks": len(tasks),
                "total_events": len(events)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve schedule")

@app.get("/productivity")
async def get_productivity_stats(
    date: Optional[datetime] = None,
    days_back: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Get productivity statistics"""
    try:
        if not date:
            date = datetime.now()

        end_date = date.date()
        start_date = end_date - timedelta(days=days_back)

        # Get productivity stats for date range
        stats_query = select(Productivity).where(
            and_(
                Productivity.user_id == DEFAULT_USER_ID,
                func.date(Productivity.date) >= start_date,
                func.date(Productivity.date) <= end_date
            )
        ).order_by(Productivity.date)

        result = await db.execute(stats_query)
        stats = result.scalars().all()

        # Calculate aggregated stats
        total_completed = sum(stat.tasks_completed for stat in stats)
        total_created = sum(stat.tasks_created for stat in stats)
        avg_focus_score = sum(stat.focus_score for stat in stats) / len(stats) if stats else 0

        # Get current task counts
        task_summary = await get_task_summary(db)

        return {
            "productivity_stats": {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days_back
                },
                "totals": {
                    "tasks_completed": total_completed,
                    "tasks_created": total_created,
                    "avg_focus_score": round(avg_focus_score, 1)
                },
                "daily_stats": [
                    {
                        "date": stat.date.isoformat(),
                        "tasks_completed": stat.tasks_completed,
                        "tasks_created": stat.tasks_created,
                        "work_time_minutes": stat.total_work_time,
                        "focus_score": stat.focus_score
                    }
                    for stat in stats
                ]
            },
            "current_status": task_summary,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting productivity stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve productivity stats")

@app.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get comprehensive personal assistant statistics"""
    try:
        # Task statistics
        task_summary = await get_task_summary(db)

        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_tasks = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.user_id == DEFAULT_USER_ID,
                    Task.created_at >= week_ago
                )
            )
        )
        recent_task_count = recent_tasks.scalar() or 0

        # Completion rate
        total_tasks = task_summary.get("total", 0)
        completed_tasks = task_summary.get("by_status", {}).get("completed", 0)
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Get preferences count
        pref_count = await db.execute(
            select(func.count(UserPreference.id)).where(
                UserPreference.user_id == DEFAULT_USER_ID
            )
        )
        preference_count = pref_count.scalar() or 0

        # Todoist integration status
        todoist_status = await todoist.test_connection()

        return {
            "summary": {
                "total_tasks": total_tasks,
                "completion_rate": round(completion_rate, 1),
                "recent_activity": recent_task_count,
                "preferences_set": preference_count
            },
            "task_breakdown": task_summary,
            "integrations": {
                "todoist": todoist_status
            },
            "user_id": DEFAULT_USER_ID,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.post("/sync")
async def sync_with_todoist(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Sync tasks with Todoist"""
    if not todoist.is_enabled():
        raise HTTPException(status_code=400, detail="Todoist integration not enabled")

    try:
        # Start sync in background
        background_tasks.add_task(perform_todoist_sync, db)

        return {
            "message": "Todoist sync started",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start sync")

# Background task functions
async def sync_task_to_todoist(task_id: str, task_data: Dict[str, Any]):
    """Background task to sync task to Todoist"""
    try:
        result = await todoist.create_task(task_data)
        if result:
            # Update local task with Todoist ID
            async with db_config.get_session() as db:
                task_uuid = uuid.UUID(task_id)
                query = select(Task).where(Task.id == task_uuid)
                result_db = await db.execute(query)
                task = result_db.scalar_one_or_none()
                if task:
                    task.todoist_id = result["todoist_id"]
                    task.metadata = task.metadata or {}
                    task.metadata["todoist_url"] = result.get("url")
                    await db.commit()
                    logger.info(f"Synced task {task_id} to Todoist: {result['todoist_id']}")
    except Exception as e:
        logger.error(f"Failed to sync task {task_id} to Todoist: {e}")

async def update_todoist_task(todoist_id: str, task_data: Dict[str, Any]):
    """Background task to update Todoist task"""
    try:
        if task_data.get("completed"):
            await todoist.complete_task(todoist_id)
        else:
            await todoist.update_task(todoist_id, task_data)
        logger.info(f"Updated Todoist task: {todoist_id}")
    except Exception as e:
        logger.error(f"Failed to update Todoist task {todoist_id}: {e}")

async def complete_todoist_task(todoist_id: str):
    """Background task to complete Todoist task"""
    try:
        await todoist.complete_task(todoist_id)
        logger.info(f"Completed Todoist task: {todoist_id}")
    except Exception as e:
        logger.error(f"Failed to complete Todoist task {todoist_id}: {e}")

async def delete_todoist_task(todoist_id: str):
    """Background task to delete Todoist task"""
    try:
        await todoist.delete_task(todoist_id)
        logger.info(f"Deleted Todoist task: {todoist_id}")
    except Exception as e:
        logger.error(f"Failed to delete Todoist task {todoist_id}: {e}")

async def perform_todoist_sync(db: AsyncSession):
    """Background task to perform full Todoist sync"""
    try:
        logger.info("Starting Todoist sync")

        # Get tasks from Todoist
        todoist_tasks = await todoist.sync_tasks_from_todoist()

        synced_count = 0
        for task_data in todoist_tasks:
            # Check if task already exists locally
            existing_query = select(Task).where(
                and_(
                    Task.user_id == DEFAULT_USER_ID,
                    Task.todoist_id == task_data["todoist_id"]
                )
            )
            result = await db.execute(existing_query)
            existing_task = result.scalar_one_or_none()

            if existing_task:
                # Update existing task
                existing_task.title = task_data["title"]
                existing_task.description = task_data["description"]
                existing_task.status = TaskStatus(task_data["status"])
                existing_task.priority = TaskPriority(task_data["priority"])
                existing_task.due_date = task_data["due_date"]
                existing_task.completed_at = task_data["completed_at"]
                existing_task.updated_at = datetime.utcnow()
            else:
                # Create new task
                new_task = Task(
                    title=task_data["title"],
                    description=task_data["description"],
                    status=TaskStatus(task_data["status"]),
                    priority=TaskPriority(task_data["priority"]),
                    due_date=task_data["due_date"],
                    completed_at=task_data["completed_at"],
                    user_id=DEFAULT_USER_ID,
                    todoist_id=task_data["todoist_id"],
                    tags=task_data["tags"],
                    metadata=task_data["metadata"]
                )
                db.add(new_task)

            synced_count += 1

        await db.commit()
        logger.info(f"Todoist sync completed: {synced_count} tasks synced")

    except Exception as e:
        logger.error(f"Todoist sync failed: {e}")
        await db.rollback()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level="info"
    )