#!/usr/bin/env python3

"""
AI Ecosystem Orchestrator

Advanced orchestration layer that creates intelligent connections between
calendar events, tasks, and knowledge base documents with contextual linking.
"""

import asyncio
import aiohttp
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from dateutil import parser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
PERSONAL_SERVICE_URL = os.getenv("PERSONAL_SERVICE_URL", "http://localhost:8002")

@dataclass
class DocumentContext:
    id: str
    title: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any]

@dataclass
class TaskContext:
    id: str
    title: str
    description: str
    status: str
    priority: str
    due_date: Optional[datetime]
    tags: List[str]

@dataclass
class EventContext:
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    metadata: Dict[str, Any]

class ContextualOrchestrator:
    def __init__(self):
        self.session = None
        self.similarity_threshold = 0.3  # Minimum relevance for context linking

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
            logger.error(f"Request failed to {url}: {e}")
            return {"error": str(e), "status_code": 500}

    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'our', 'their', 'myself', 'yourself', 'himself', 'herself',
            'itself', 'ourselves', 'yourselves', 'themselves'
        }

        # Extract words, convert to lowercase, filter by length and stop words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = [word for word in words
                   if len(word) >= min_length and word not in stop_words]

        # Return unique keywords, sorted by frequency
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

    async def find_related_documents(self, query: str, max_results: int = 5) -> List[DocumentContext]:
        """Find documents related to a query"""
        try:
            # Search RAG service with the query
            search_response = await self.make_request(
                f"{RAG_SERVICE_URL}/search",
                "POST",
                {"query": query, "max_results": max_results, "use_vector_search": True}
            )

            if "error" in search_response or "results" not in search_response:
                return []

            documents = []
            for result in search_response["results"]:
                doc = DocumentContext(
                    id=result.get("id", ""),
                    title=result.get("title", result.get("id", "Unknown")),
                    content=result.get("content", ""),
                    relevance_score=result.get("similarity_score", result.get("relevance_score", 0)),
                    metadata=result.get("metadata", {})
                )

                # Only include documents above threshold
                if doc.relevance_score >= self.similarity_threshold:
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error finding related documents: {e}")
            return []

    async def find_related_tasks(self, keywords: List[str]) -> List[TaskContext]:
        """Find tasks related to keywords"""
        try:
            # Get all tasks from personal service
            tasks_response = await self.make_request(f"{PERSONAL_SERVICE_URL}/tasks")

            if "error" in tasks_response or "tasks" not in tasks_response:
                return []

            related_tasks = []
            for task_data in tasks_response["tasks"]:
                task_title = task_data.get("title", "").lower()
                task_desc = task_data.get("description", "").lower()
                task_tags = [tag.lower() for tag in task_data.get("tags", [])]

                # Calculate relevance based on keyword matches
                relevance = 0
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in task_title:
                        relevance += 3  # Title matches are more important
                    if keyword_lower in task_desc:
                        relevance += 2  # Description matches
                    if keyword_lower in task_tags:
                        relevance += 4  # Tag matches are very important

                # Include task if it has any relevance
                if relevance > 0:
                    task = TaskContext(
                        id=task_data.get("id", ""),
                        title=task_data.get("title", ""),
                        description=task_data.get("description", ""),
                        status=task_data.get("status", ""),
                        priority=task_data.get("priority", ""),
                        due_date=parser.parse(task_data["due_date"]) if task_data.get("due_date") else None,
                        tags=task_data.get("tags", [])
                    )
                    related_tasks.append(task)

            return related_tasks

        except Exception as e:
            logger.error(f"Error finding related tasks: {e}")
            return []

    async def find_upcoming_events(self, keywords: List[str], days_ahead: int = 30) -> List[EventContext]:
        """Find upcoming events related to keywords"""
        try:
            # Get schedule from personal service
            schedule_response = await self.make_request(f"{PERSONAL_SERVICE_URL}/schedule")

            if "error" in schedule_response or "schedule" not in schedule_response:
                return []

            related_events = []
            schedule = schedule_response["schedule"]

            # Check events in the next 30 days
            current_date = datetime.now().date()
            for i in range(days_ahead):
                check_date = current_date + timedelta(days=i)
                date_str = check_date.isoformat()

                if date_str in schedule:
                    day_info = schedule[date_str]

                    # Check tasks scheduled for this day
                    for task in day_info.get("tasks", []):
                        task_title = task.get("title", "").lower()

                        relevance = 0
                        for keyword in keywords:
                            if keyword.lower() in task_title:
                                relevance += 1

                        if relevance > 0:
                            event = EventContext(
                                id=task.get("id", ""),
                                title=task.get("title", ""),
                                description="Scheduled task",
                                start_time=datetime.combine(check_date, datetime.min.time()),
                                end_time=None,
                                metadata={"type": "task", "relevance": relevance}
                            )
                            related_events.append(event)

                    # Check events scheduled for this day
                    for event in day_info.get("events", []):
                        event_title = event.get("title", "").lower()

                        relevance = 0
                        for keyword in keywords:
                            if keyword.lower() in event_title:
                                relevance += 1

                        if relevance > 0:
                            start_time = datetime.combine(check_date, datetime.min.time())
                            if event.get("start_time"):
                                try:
                                    start_time = datetime.combine(
                                        check_date,
                                        parser.parse(event["start_time"]).time()
                                    )
                                except:
                                    pass

                            end_time = None
                            if event.get("end_time"):
                                try:
                                    end_time = datetime.combine(
                                        check_date,
                                        parser.parse(event["end_time"]).time()
                                    )
                                except:
                                    pass

                            related_event = EventContext(
                                id=event.get("id", ""),
                                title=event.get("title", ""),
                                description="Calendar event",
                                start_time=start_time,
                                end_time=end_time,
                                metadata={"type": "event", "relevance": relevance}
                            )
                            related_events.append(related_event)

            return related_events

        except Exception as e:
            logger.error(f"Error finding upcoming events: {e}")
            return []

    async def create_contextual_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create calendar event with RAG-sourced context and document attachments

        Args:
            event_data: Event information including title, description, start_time, etc.

        Returns:
            Enhanced event data with related documents and context
        """
        try:
            title = event_data.get("title", "")
            description = event_data.get("description", "")

            # Extract keywords from event title and description
            search_text = f"{title} {description}"
            keywords = self.extract_keywords(search_text)

            # Find related documents using the event content
            related_docs = await self.find_related_documents(search_text, max_results=5)

            # Find related tasks
            related_tasks = await self.find_related_tasks(keywords)

            # Enhance event metadata with context
            enhanced_metadata = event_data.get("metadata", {})

            # Add document context
            if related_docs:
                enhanced_metadata["related_documents"] = [
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "relevance_score": doc.relevance_score,
                        "preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    }
                    for doc in related_docs
                ]

                # Create contextual description if none provided
                if not description and related_docs:
                    description = f"Event with {len(related_docs)} related documents"
                    description += f"\n\nKey documents:\n"
                    for doc in related_docs[:3]:
                        description += f"• {doc.title} (Relevance: {doc.relevance_score:.2f})\n"

            # Add related tasks context
            if related_tasks:
                enhanced_metadata["related_tasks"] = [
                    {
                        "id": task.id,
                        "title": task.title,
                        "status": task.status,
                        "priority": task.priority
                    }
                    for task in related_tasks[:5]
                ]

            # Add search keywords used
            enhanced_metadata["context_keywords"] = keywords
            enhanced_metadata["context_generated_at"] = datetime.now().isoformat()

            # Create enhanced event data
            enhanced_event = {
                **event_data,
                "description": description,
                "metadata": enhanced_metadata
            }

            logger.info(f"Enhanced event '{title}' with {len(related_docs)} documents and {len(related_tasks)} tasks")

            return {
                "event_data": enhanced_event,
                "context": {
                    "related_documents": len(related_docs),
                    "related_tasks": len(related_tasks),
                    "keywords_used": keywords
                },
                "suggestions": self._generate_event_suggestions(related_docs, related_tasks)
            }

        except Exception as e:
            logger.error(f"Error creating contextual event: {e}")
            return {"event_data": event_data, "error": str(e)}

    async def create_contextual_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create task with knowledge base links and supporting documents

        Args:
            task_data: Task information including title, description, priority, etc.

        Returns:
            Enhanced task data with linked documents and suggestions
        """
        try:
            title = task_data.get("title", "")
            description = task_data.get("description", "")

            # Extract keywords from task content
            search_text = f"{title} {description}"
            keywords = self.extract_keywords(search_text)

            # Find supporting documents
            supporting_docs = await self.find_related_documents(search_text, max_results=5)

            # Find related upcoming events
            related_events = await self.find_upcoming_events(keywords, days_ahead=14)

            # Enhance task metadata with knowledge links
            enhanced_metadata = task_data.get("metadata", {})

            # Add document links
            if supporting_docs:
                enhanced_metadata["supporting_documents"] = [
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "relevance_score": doc.relevance_score,
                        "key_content": doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
                    }
                    for doc in supporting_docs
                ]

                # Enhance description with document references
                if len(supporting_docs) > 0:
                    if not description:
                        description = f"Task with {len(supporting_docs)} supporting documents"
                    else:
                        description += f"\n\nReferences ({len(supporting_docs)} documents):"

                    for doc in supporting_docs[:3]:
                        description += f"\n• {doc.title}"

            # Add related events context
            if related_events:
                enhanced_metadata["related_events"] = [
                    {
                        "id": event.id,
                        "title": event.title,
                        "start_time": event.start_time.isoformat(),
                        "type": event.metadata.get("type", "unknown")
                    }
                    for event in related_events[:3]
                ]

            # Add intelligent tags based on document content
            enhanced_tags = list(task_data.get("tags", []))
            doc_tags = self._extract_tags_from_documents(supporting_docs)
            enhanced_tags.extend([tag for tag in doc_tags if tag not in enhanced_tags])

            # Add context metadata
            enhanced_metadata["context_keywords"] = keywords
            enhanced_metadata["context_generated_at"] = datetime.now().isoformat()

            # Create enhanced task data
            enhanced_task = {
                **task_data,
                "description": description,
                "tags": enhanced_tags[:10],  # Limit to 10 tags
                "metadata": enhanced_metadata
            }

            logger.info(f"Enhanced task '{title}' with {len(supporting_docs)} documents and {len(related_events)} events")

            return {
                "task_data": enhanced_task,
                "context": {
                    "supporting_documents": len(supporting_docs),
                    "related_events": len(related_events),
                    "keywords_used": keywords,
                    "suggested_tags": doc_tags
                },
                "suggestions": self._generate_task_suggestions(supporting_docs, related_events)
            }

        except Exception as e:
            logger.error(f"Error creating contextual task: {e}")
            return {"task_data": task_data, "error": str(e)}

    async def search_with_context(self, query: str, max_docs: int = 10) -> Dict[str, Any]:
        """
        Search documents with upcoming events and task suggestions

        Args:
            query: Search query
            max_docs: Maximum number of documents to return

        Returns:
            Search results with contextual information and suggestions
        """
        try:
            # Extract keywords from query
            keywords = self.extract_keywords(query)

            # Search documents
            documents = await self.find_related_documents(query, max_results=max_docs)

            # Find related tasks and events
            related_tasks = await self.find_related_tasks(keywords)
            related_events = await self.find_upcoming_events(keywords, days_ahead=7)

            # Generate task suggestions based on document content
            task_suggestions = []
            if documents:
                task_suggestions = self._generate_task_suggestions_from_docs(documents, query)

            # Generate event suggestions
            event_suggestions = []
            if documents:
                event_suggestions = self._generate_event_suggestions_from_docs(documents, query)

            return {
                "query": query,
                "documents": [
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "content": doc.content,
                        "relevance_score": doc.relevance_score,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ],
                "context": {
                    "related_tasks": [
                        {
                            "id": task.id,
                            "title": task.title,
                            "status": task.status,
                            "priority": task.priority,
                            "due_date": task.due_date.isoformat() if task.due_date else None
                        }
                        for task in related_tasks[:5]
                    ],
                    "upcoming_events": [
                        {
                            "id": event.id,
                            "title": event.title,
                            "start_time": event.start_time.isoformat(),
                            "type": event.metadata.get("type", "unknown")
                        }
                        for event in related_events[:5]
                    ]
                },
                "suggestions": {
                    "tasks": task_suggestions,
                    "events": event_suggestions,
                    "keywords_used": keywords
                },
                "stats": {
                    "documents_found": len(documents),
                    "related_tasks": len(related_tasks),
                    "upcoming_events": len(related_events)
                }
            }

        except Exception as e:
            logger.error(f"Error in contextual search: {e}")
            return {"query": query, "error": str(e)}

    def _extract_tags_from_documents(self, documents: List[DocumentContext]) -> List[str]:
        """Extract relevant tags from document content and metadata"""
        tags = set()

        for doc in documents:
            # Add tags from document metadata
            doc_tags = doc.metadata.get("tags", [])
            if isinstance(doc_tags, list):
                tags.update(doc_tags)

            # Extract category-based tags
            category = doc.metadata.get("category", "")
            if category:
                tags.add(category)

            # Extract tags from title keywords
            title_keywords = self.extract_keywords(doc.title, min_length=4)
            tags.update(title_keywords[:3])  # Add top 3 title keywords

        return list(tags)[:8]  # Return up to 8 tags

    def _generate_event_suggestions(self, docs: List[DocumentContext], tasks: List[TaskContext]) -> List[Dict[str, Any]]:
        """Generate event suggestions based on documents and tasks"""
        suggestions = []

        # Suggest review sessions for high-relevance documents
        high_relevance_docs = [doc for doc in docs if doc.relevance_score > 0.7]
        if high_relevance_docs:
            suggestions.append({
                "type": "review_session",
                "title": f"Review {len(high_relevance_docs)} related documents",
                "description": f"Suggested review session for: {', '.join([doc.title for doc in high_relevance_docs[:3]])}",
                "suggested_duration": min(30 * len(high_relevance_docs), 120),  # 30 min per doc, max 2 hours
                "priority": "medium"
            })

        # Suggest planning sessions for related tasks
        pending_tasks = [task for task in tasks if task.status != "completed"]
        if pending_tasks:
            suggestions.append({
                "type": "planning_session",
                "title": f"Plan {len(pending_tasks)} related tasks",
                "description": f"Planning session for tasks: {', '.join([task.title for task in pending_tasks[:3]])}",
                "suggested_duration": 45,
                "priority": "medium"
            })

        return suggestions

    def _generate_task_suggestions(self, docs: List[DocumentContext], events: List[EventContext]) -> List[Dict[str, Any]]:
        """Generate task suggestions based on documents and events"""
        suggestions = []

        # Suggest documentation tasks for technical documents
        tech_docs = [doc for doc in docs if any(keyword in doc.content.lower()
                    for keyword in ['api', 'code', 'implementation', 'technical', 'system'])]
        if tech_docs:
            suggestions.append({
                "type": "documentation",
                "title": f"Document findings from {len(tech_docs)} technical resources",
                "description": f"Create summary documentation based on: {', '.join([doc.title for doc in tech_docs[:2]])}",
                "priority": "medium",
                "estimated_time": "60 minutes"
            })

        # Suggest follow-up tasks for research documents
        research_docs = [doc for doc in docs if any(keyword in doc.content.lower()
                        for keyword in ['research', 'study', 'analysis', 'findings', 'conclusion'])]
        if research_docs:
            suggestions.append({
                "type": "follow_up",
                "title": f"Follow up on research from {len(research_docs)} documents",
                "description": f"Create action items based on research: {', '.join([doc.title for doc in research_docs[:2]])}",
                "priority": "high",
                "estimated_time": "45 minutes"
            })

        return suggestions

    def _generate_task_suggestions_from_docs(self, docs: List[DocumentContext], query: str) -> List[Dict[str, Any]]:
        """Generate task suggestions specifically from document content"""
        suggestions = []

        if docs:
            suggestions.append({
                "title": f"Review and summarize findings on '{query}'",
                "description": f"Create summary of {len(docs)} documents found for '{query}'",
                "priority": "medium",
                "tags": ["research", "review", "summary"],
                "estimated_duration": 30 + (len(docs) * 15)  # Base 30 min + 15 min per doc
            })

            if len(docs) > 3:
                suggestions.append({
                    "title": f"Create presentation on '{query}' research",
                    "description": f"Develop presentation based on {len(docs)} research documents",
                    "priority": "medium",
                    "tags": ["presentation", "research", "communication"],
                    "estimated_duration": 90
                })

        return suggestions

    def _generate_event_suggestions_from_docs(self, docs: List[DocumentContext], query: str) -> List[Dict[str, Any]]:
        """Generate event suggestions from document content"""
        suggestions = []

        if docs:
            suggestions.append({
                "title": f"Study session: {query}",
                "description": f"Dedicated study time for {len(docs)} documents on '{query}'",
                "suggested_duration": min(45 + (len(docs) * 20), 180),  # 45 min base + 20 min per doc, max 3 hours
                "type": "study_session"
            })

            if len(docs) >= 2:
                suggestions.append({
                    "title": f"Discussion/brainstorm: {query}",
                    "description": f"Team discussion based on research findings from {len(docs)} documents",
                    "suggested_duration": 60,
                    "type": "meeting"
                })

        return suggestions

    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

# Global orchestrator instance
orchestrator = ContextualOrchestrator()

# Convenience functions for easy import
async def create_smart_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create event with intelligent document context"""
    return await orchestrator.create_contextual_event(event_data)

async def create_smart_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create task with knowledge base links"""
    return await orchestrator.create_contextual_task(task_data)

async def contextual_search(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Search with related events and task suggestions"""
    return await orchestrator.search_with_context(query, max_results)

async def find_document_context(query: str) -> List[DocumentContext]:
    """Find related documents for a query"""
    return await orchestrator.find_related_documents(query)

async def close_orchestrator():
    """Close orchestrator resources"""
    await orchestrator.close()

if __name__ == "__main__":
    # Example usage
    async def demo():
        print("🎼 AI Ecosystem Orchestrator Demo")

        # Demo contextual search
        print("\n🔍 Contextual Search Demo:")
        search_result = await contextual_search("machine learning algorithms")
        print(f"Found {search_result.get('stats', {}).get('documents_found', 0)} documents")
        print(f"Related tasks: {len(search_result.get('context', {}).get('related_tasks', []))}")
        print(f"Upcoming events: {len(search_result.get('context', {}).get('upcoming_events', []))}")

        # Demo smart task creation
        print("\n📝 Smart Task Creation Demo:")
        task_result = await create_smart_task({
            "title": "Research neural networks for project",
            "description": "Deep dive into neural network architectures",
            "priority": "high",
            "tags": ["research", "ai"]
        })

        context = task_result.get("context", {})
        print(f"Supporting documents: {context.get('supporting_documents', 0)}")
        print(f"Related events: {context.get('related_events', 0)}")
        print(f"Suggested tags: {', '.join(context.get('suggested_tags', []))}")

        await close_orchestrator()

    # Run demo
    asyncio.run(demo())