#!/usr/bin/env python3

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import openai
import anthropic
from dateutil import parser

logger = logging.getLogger(__name__)

@dataclass
class Intent:
    name: str
    confidence: float
    entities: Dict[str, Any]
    action: str
    parameters: Dict[str, Any]

@dataclass
class AssistantResponse:
    intent: Intent
    response_text: str
    actions: List[Dict[str, Any]]
    data: Optional[Dict[str, Any]]
    processing_time: float
    timestamp: str

class SmartAssistant:
    def __init__(self, rag_service_url: str, personal_service_url: str):
        self.rag_service_url = rag_service_url
        self.personal_service_url = personal_service_url

        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_llm_clients()

        # Intent patterns and keywords
        self.intent_patterns = self._load_intent_patterns()
        self.entity_extractors = self._load_entity_extractors()

    def _initialize_llm_clients(self):
        """Initialize LLM clients for advanced intent recognition"""
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)

            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}")

    def _load_intent_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load intent recognition patterns"""
        return {
            "search_knowledge": {
                "keywords": ["search", "find", "lookup", "know", "about", "explain", "what is", "tell me"],
                "patterns": [
                    r"(?:search|find|lookup|what is|tell me about|explain)\s+(.+)",
                    r"(?:do you know|can you find)\s+(.+)",
                    r"(?:information|details)\s+(?:about|on)\s+(.+)"
                ],
                "confidence_base": 0.8
            },
            "task_management": {
                "keywords": ["task", "todo", "reminder", "schedule", "deadline", "complete", "done"],
                "patterns": [
                    r"(?:create|add|make)\s+(?:a\s+)?(?:task|todo|reminder)\s+(.+)",
                    r"(?:mark|set)\s+(.+)\s+(?:as\s+)?(?:complete|done)",
                    r"(?:what|show|list)\s+(?:are\s+)?(?:my\s+)?(?:tasks|todos)",
                    r"(?:schedule|plan)\s+(.+)",
                    r"(?:deadline|due)\s+(.+)"
                ],
                "confidence_base": 0.9
            },
            "productivity": {
                "keywords": ["productivity", "stats", "progress", "completed", "performance"],
                "patterns": [
                    r"(?:show|get|display)\s+(?:my\s+)?(?:productivity|stats|progress)",
                    r"(?:how many|how much)\s+(.+)\s+(?:completed|done)",
                    r"(?:performance|progress)\s+(?:report|summary)"
                ],
                "confidence_base": 0.7
            },
            "calendar_schedule": {
                "keywords": ["calendar", "schedule", "appointment", "meeting", "event", "today", "tomorrow"],
                "patterns": [
                    r"(?:what.s|show|display)\s+(?:my\s+)?(?:schedule|calendar|agenda)(?:\s+for\s+(.+))?",
                    r"(?:any|what)\s+(?:meetings|appointments|events)(?:\s+(.+))?",
                    r"(?:free|busy|available)\s+(?:time|slots)(?:\s+(.+))?"
                ],
                "confidence_base": 0.8
            },
            "document_management": {
                "keywords": ["document", "file", "doc", "pdf", "note", "upload", "download"],
                "patterns": [
                    r"(?:upload|add|create)\s+(?:a\s+)?(?:document|file|note)\s+(.+)",
                    r"(?:find|search)\s+(?:document|file)\s+(.+)",
                    r"(?:delete|remove)\s+(?:document|file)\s+(.+)"
                ],
                "confidence_base": 0.8
            },
            "integration": {
                "keywords": ["todoist", "sync", "integration", "connect", "import", "export"],
                "patterns": [
                    r"(?:sync|connect|integrate)\s+(?:with\s+)?(.+)",
                    r"(?:import|export)\s+(?:from|to)\s+(.+)",
                    r"(?:todoist|external)\s+(.+)"
                ],
                "confidence_base": 0.7
            },
            "help": {
                "keywords": ["help", "how", "can you", "what can", "assist", "support"],
                "patterns": [
                    r"(?:help|assist|support)(?:\s+me)?(?:\s+with\s+(.+))?",
                    r"(?:how do|can you|what can)\s+(.+)",
                    r"(?:what are|show)\s+(?:your\s+)?(?:capabilities|features|functions)"
                ],
                "confidence_base": 0.6
            }
        }

    def _load_entity_extractors(self) -> Dict[str, Any]:
        """Load entity extraction patterns"""
        return {
            "dates": {
                "patterns": [
                    r"\b(today|tomorrow|yesterday)\b",
                    r"\b(\d{1,2}\/\d{1,2}\/\d{2,4})\b",
                    r"\b(\d{4}-\d{2}-\d{2})\b",
                    r"\b(next\s+\w+|last\s+\w+)\b",
                    r"\b(\w+day)\b"
                ]
            },
            "times": {
                "patterns": [
                    r"\b(\d{1,2}:\d{2}(?:\s*[ap]m)?)\b",
                    r"\b(\d{1,2}\s*[ap]m)\b",
                    r"\b(morning|afternoon|evening|night)\b"
                ]
            },
            "priorities": {
                "patterns": [
                    r"\b(urgent|high|medium|low)\s+priority\b",
                    r"\b(important|critical|asap)\b"
                ]
            },
            "numbers": {
                "patterns": [
                    r"\b(\d+)\b"
                ]
            }
        }

    async def process_message(self, message: str, context: Optional[Dict] = None, health_checker=None) -> AssistantResponse:
        """Process user message and determine intent and response"""
        start_time = datetime.now()

        try:
            # Normalize message
            normalized_message = message.lower().strip()

            # Extract entities
            entities = self._extract_entities(normalized_message)

            # Determine intent
            intent = await self._determine_intent(normalized_message, entities, context)

            # Execute action based on intent
            response_text, actions, data = await self._execute_intent(intent, message, entities, health_checker)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return AssistantResponse(
                intent=intent,
                response_text=response_text,
                actions=actions,
                data=data,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return AssistantResponse(
                intent=Intent("error", 0.0, {}, "error", {}),
                response_text=f"I encountered an error processing your request: {str(e)}",
                actions=[],
                data=None,
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """Extract entities from message using regex patterns"""
        entities = {}

        for entity_type, config in self.entity_extractors.items():
            matches = []
            for pattern in config["patterns"]:
                found = re.findall(pattern, message, re.IGNORECASE)
                matches.extend(found)

            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates

        return entities

    async def _determine_intent(self, message: str, entities: Dict, context: Optional[Dict]) -> Intent:
        """Determine user intent using pattern matching and LLM assistance"""

        # Pattern-based intent detection
        best_intent = None
        best_confidence = 0.0

        for intent_name, config in self.intent_patterns.items():
            confidence = self._calculate_pattern_confidence(message, config)

            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent_name

        # Enhance with LLM-based intent classification if available
        if self.openai_client and best_confidence < 0.8:
            llm_intent = await self._llm_intent_classification(message)
            if llm_intent and llm_intent["confidence"] > best_confidence:
                best_intent = llm_intent["intent"]
                best_confidence = llm_intent["confidence"]

        # Determine action and parameters
        action, parameters = self._determine_action_parameters(best_intent, message, entities)

        return Intent(
            name=best_intent or "general",
            confidence=best_confidence,
            entities=entities,
            action=action,
            parameters=parameters
        )

    def _calculate_pattern_confidence(self, message: str, config: Dict) -> float:
        """Calculate confidence score for intent pattern matching"""
        confidence = 0.0

        # Check keywords
        keyword_matches = sum(1 for keyword in config["keywords"] if keyword in message)
        keyword_score = min(keyword_matches / len(config["keywords"]), 1.0) * 0.6

        # Check regex patterns
        pattern_score = 0.0
        for pattern in config.get("patterns", []):
            if re.search(pattern, message, re.IGNORECASE):
                pattern_score = 0.4
                break

        total_confidence = keyword_score + pattern_score

        # Apply base confidence
        if total_confidence > 0:
            total_confidence = max(total_confidence, config["confidence_base"])

        return min(total_confidence, 1.0)

    async def _llm_intent_classification(self, message: str) -> Optional[Dict]:
        """Use LLM for advanced intent classification"""
        try:
            prompt = f"""
Classify the following user message into one of these intents:
- search_knowledge: User wants to search for information or learn about something
- task_management: User wants to create, update, complete, or view tasks
- productivity: User wants to see productivity stats or progress
- calendar_schedule: User wants to view or manage their schedule
- document_management: User wants to manage documents or files
- integration: User wants to sync with external services
- help: User needs help or wants to know capabilities
- general: General conversation or unclear intent

Message: "{message}"

Respond with JSON format:
{{"intent": "intent_name", "confidence": 0.85, "reasoning": "brief explanation"}}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            import json
            result = json.loads(result_text)
            return result

        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            return None

    def _determine_action_parameters(self, intent: str, message: str, entities: Dict) -> Tuple[str, Dict]:
        """Determine specific action and parameters based on intent"""

        action_mapping = {
            "search_knowledge": {
                "default_action": "search_documents",
                "parameters": {"query": self._extract_search_query(message)}
            },
            "task_management": {
                "default_action": self._determine_task_action(message),
                "parameters": self._extract_task_parameters(message, entities)
            },
            "productivity": {
                "default_action": "get_productivity_stats",
                "parameters": {"period": entities.get("dates", ["week"])[0] if entities.get("dates") else "week"}
            },
            "calendar_schedule": {
                "default_action": "get_schedule",
                "parameters": {"date": entities.get("dates", ["today"])[0] if entities.get("dates") else "today"}
            },
            "document_management": {
                "default_action": self._determine_document_action(message),
                "parameters": {"query": self._extract_search_query(message)}
            },
            "integration": {
                "default_action": "sync_external",
                "parameters": {"service": "todoist"}
            },
            "help": {
                "default_action": "show_help",
                "parameters": {"topic": self._extract_help_topic(message)}
            },
            "general": {
                "default_action": "general_response",
                "parameters": {"message": message}
            }
        }

        mapping = action_mapping.get(intent, action_mapping["general"])
        return mapping["default_action"], mapping["parameters"]

    def _extract_search_query(self, message: str) -> str:
        """Extract search query from message"""
        # Remove common command words
        query = re.sub(r'\b(search|find|lookup|what is|tell me about|explain|show|get)\b', '', message, flags=re.IGNORECASE)
        return query.strip()

    def _determine_task_action(self, message: str) -> str:
        """Determine specific task action"""
        if any(word in message.lower() for word in ["create", "add", "make", "new"]):
            return "create_task"
        elif any(word in message.lower() for word in ["complete", "done", "finish", "mark"]):
            return "complete_task"
        elif any(word in message.lower() for word in ["update", "edit", "change", "modify"]):
            return "update_task"
        elif any(word in message.lower() for word in ["delete", "remove", "cancel"]):
            return "delete_task"
        else:
            return "list_tasks"

    def _determine_document_action(self, message: str) -> str:
        """Determine specific document action"""
        if any(word in message.lower() for word in ["upload", "add", "create"]):
            return "upload_document"
        elif any(word in message.lower() for word in ["delete", "remove"]):
            return "delete_document"
        else:
            return "search_documents"

    def _extract_task_parameters(self, message: str, entities: Dict) -> Dict[str, Any]:
        """Extract task-specific parameters"""
        params = {}

        # Extract task title/description
        if "create" in message.lower() or "add" in message.lower():
            title_match = re.search(r'(?:create|add|make)\s+(?:a\s+)?(?:task|todo|reminder)\s+(.+)', message, re.IGNORECASE)
            if title_match:
                params["title"] = title_match.group(1).strip()

        # Extract priority
        if entities.get("priorities"):
            priority_text = entities["priorities"][0].lower()
            if "urgent" in priority_text or "critical" in priority_text or "asap" in priority_text:
                params["priority"] = "urgent"
            elif "high" in priority_text or "important" in priority_text:
                params["priority"] = "high"
            elif "low" in priority_text:
                params["priority"] = "low"
            else:
                params["priority"] = "medium"

        # Extract due date
        if entities.get("dates"):
            params["due_date"] = entities["dates"][0]

        return params

    def _extract_help_topic(self, message: str) -> str:
        """Extract help topic from message"""
        topic_match = re.search(r'help\s+(?:me\s+)?(?:with\s+)?(.+)', message, re.IGNORECASE)
        if topic_match:
            return topic_match.group(1).strip()
        return "general"

    async def _execute_intent(self, intent: Intent, original_message: str, entities: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Execute the determined intent and return response"""

        try:
            if intent.action == "search_documents":
                return await self._handle_search(intent.parameters, health_checker)
            elif intent.action == "create_task":
                return await self._handle_create_task(intent.parameters, health_checker)
            elif intent.action == "list_tasks":
                return await self._handle_list_tasks(intent.parameters, health_checker)
            elif intent.action == "complete_task":
                return await self._handle_complete_task(intent.parameters, health_checker)
            elif intent.action == "get_productivity_stats":
                return await self._handle_productivity_stats(intent.parameters, health_checker)
            elif intent.action == "get_schedule":
                return await self._handle_schedule(intent.parameters, health_checker)
            elif intent.action == "sync_external":
                return await self._handle_sync(intent.parameters, health_checker)
            elif intent.action == "show_help":
                return self._handle_help(intent.parameters)
            else:
                return self._handle_general_response(original_message, intent)

        except Exception as e:
            logger.error(f"Error executing intent {intent.action}: {e}")
            return f"I encountered an error while {intent.action}: {str(e)}", [], None

    async def _handle_search(self, parameters: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle knowledge search requests"""
        query = parameters.get("query", "")
        if not query:
            return "I need a search query. What would you like me to look for?", [], None

        try:
            search_result = await health_checker.make_request(
                f"{self.rag_service_url}/search",
                "POST",
                {"query": query, "max_results": 3, "use_vector_search": True}
            )

            if "error" in search_result:
                return f"Search failed: {search_result['error']}", [], None

            results = search_result.get("results", [])
            if not results:
                return f"I couldn't find any information about '{query}'. Try rephrasing your search.", [], search_result

            response_parts = [f"I found {len(results)} results for '{query}':"]

            for i, result in enumerate(results, 1):
                title = result.get("title", result.get("id", "Unknown"))
                content = result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
                score = result.get("similarity_score", result.get("relevance_score", 0))

                response_parts.append(f"\n{i}. **{title}** (Score: {score:.2f})")
                response_parts.append(f"   {content}")

            actions = [{"type": "search_completed", "query": query, "result_count": len(results)}]

            return "".join(response_parts), actions, search_result

        except Exception as e:
            return f"Search error: {str(e)}", [], None

    async def _handle_create_task(self, parameters: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle task creation requests"""
        title = parameters.get("title")
        if not title:
            return "I need a task title. What task would you like me to create?", [], None

        try:
            task_data = {
                "title": title,
                "priority": parameters.get("priority", "medium"),
                "sync_to_todoist": True
            }

            # Handle due date
            due_date_str = parameters.get("due_date")
            if due_date_str:
                if due_date_str.lower() == "today":
                    task_data["due_date"] = datetime.now().isoformat()
                elif due_date_str.lower() == "tomorrow":
                    task_data["due_date"] = (datetime.now() + timedelta(days=1)).isoformat()
                else:
                    try:
                        parsed_date = parser.parse(due_date_str)
                        task_data["due_date"] = parsed_date.isoformat()
                    except:
                        pass  # Skip invalid dates

            result = await health_checker.make_request(
                f"{self.personal_service_url}/tasks",
                "POST",
                task_data
            )

            if "error" in result:
                return f"Failed to create task: {result['error']}", [], None

            task_id = result.get("id")
            due_info = f" (due {parameters.get('due_date', 'no due date')})" if parameters.get("due_date") else ""

            actions = [{"type": "task_created", "task_id": task_id, "title": title}]

            return f"✅ Created task: '{title}'{due_info}", actions, result

        except Exception as e:
            return f"Task creation error: {str(e)}", [], None

    async def _handle_list_tasks(self, parameters: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle task listing requests"""
        try:
            result = await health_checker.make_request(f"{self.personal_service_url}/tasks")

            if "error" in result:
                return f"Failed to get tasks: {result['error']}", [], None

            tasks = result.get("tasks", [])
            summary = result.get("summary", {})

            if not tasks:
                return "You have no tasks. Would you like me to create one?", [], result

            # Filter and format tasks
            pending_tasks = [t for t in tasks if t.get("status") != "completed"]

            response_parts = [f"📋 You have {len(pending_tasks)} pending tasks:"]

            for i, task in enumerate(pending_tasks[:5], 1):  # Show max 5 tasks
                title = task.get("title", "Untitled")
                priority = task.get("priority", "medium")
                due_date = task.get("due_date")

                priority_emoji = {"urgent": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                due_info = ""

                if due_date:
                    try:
                        due_dt = parser.parse(due_date) if isinstance(due_date, str) else due_date
                        if due_dt.date() == datetime.now().date():
                            due_info = " (due today)"
                        elif due_dt.date() == (datetime.now() + timedelta(days=1)).date():
                            due_info = " (due tomorrow)"
                        elif due_dt < datetime.now():
                            due_info = " (overdue)"
                    except:
                        pass

                response_parts.append(f"\n{i}. {priority_emoji} {title}{due_info}")

            if len(pending_tasks) > 5:
                response_parts.append(f"\n... and {len(pending_tasks) - 5} more tasks")

            summary_text = f"\n\n📊 Summary: {summary.get('by_status', {}).get('completed', 0)} completed, {summary.get('due_today', 0)} due today"
            response_parts.append(summary_text)

            actions = [{"type": "tasks_listed", "count": len(pending_tasks)}]

            return "".join(response_parts), actions, result

        except Exception as e:
            return f"Task listing error: {str(e)}", [], None

    async def _handle_complete_task(self, parameters: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle task completion requests"""
        # This is a simplified version - in practice, you'd need task selection logic
        return "To complete a task, please specify which task or use the task management interface.", [], None

    async def _handle_productivity_stats(self, parameters: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle productivity statistics requests"""
        try:
            result = await health_checker.make_request(f"{self.personal_service_url}/productivity")

            if "error" in result:
                return f"Failed to get productivity stats: {result['error']}", [], None

            stats = result.get("productivity_stats", {})
            totals = stats.get("totals", {})

            response_parts = [
                "📈 **Productivity Summary**",
                f"\n✅ Tasks completed: {totals.get('tasks_completed', 0)}",
                f"\n📝 Tasks created: {totals.get('tasks_created', 0)}",
                f"\n🎯 Focus score: {totals.get('avg_focus_score', 0)}/100"
            ]

            current_status = result.get("current_status", {})
            if current_status:
                response_parts.append(f"\n\n📊 **Current Status:**")
                response_parts.append(f"\n• Total tasks: {current_status.get('total', 0)}")
                response_parts.append(f"\n• Due today: {current_status.get('due_today', 0)}")
                response_parts.append(f"\n• Overdue: {current_status.get('overdue', 0)}")

            actions = [{"type": "productivity_viewed", "period": parameters.get("period", "week")}]

            return "".join(response_parts), actions, result

        except Exception as e:
            return f"Productivity stats error: {str(e)}", [], None

    async def _handle_schedule(self, parameters: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle schedule requests"""
        try:
            result = await health_checker.make_request(f"{self.personal_service_url}/schedule")

            if "error" in result:
                return f"Failed to get schedule: {result['error']}", [], None

            schedule = result.get("schedule", {})

            if not schedule:
                return "Your schedule is clear! No tasks or events scheduled.", [], result

            response_parts = ["📅 **Your Schedule:**"]

            for date_str, day_info in schedule.items():
                date_obj = parser.parse(date_str).date()
                day_name = date_obj.strftime("%A")

                if date_obj == datetime.now().date():
                    day_label = "Today"
                elif date_obj == (datetime.now() + timedelta(days=1)).date():
                    day_label = "Tomorrow"
                else:
                    day_label = day_name

                total_items = day_info.get("total_items", 0)
                if total_items > 0:
                    response_parts.append(f"\n\n**{day_label} ({date_str}):**")

                    tasks = day_info.get("tasks", [])
                    events = day_info.get("events", [])

                    for task in tasks:
                        due_time = task.get("due_time", "")
                        time_info = f" at {due_time}" if due_time else ""
                        response_parts.append(f"\n📝 {task.get('title')}{time_info}")

                    for event in events:
                        start_time = event.get("start_time", "")
                        response_parts.append(f"\n📅 {event.get('title')} at {start_time}")

            if not any(day_info.get("total_items", 0) > 0 for day_info in schedule.values()):
                response_parts.append("\nNo scheduled items found.")

            actions = [{"type": "schedule_viewed", "date": parameters.get("date", "today")}]

            return "".join(response_parts), actions, result

        except Exception as e:
            return f"Schedule error: {str(e)}", [], None

    async def _handle_sync(self, parameters: Dict, health_checker) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle sync requests"""
        service = parameters.get("service", "todoist")

        try:
            result = await health_checker.make_request(
                f"{self.personal_service_url}/sync",
                "POST"
            )

            if "error" in result:
                return f"Sync failed: {result['error']}", [], None

            actions = [{"type": "sync_initiated", "service": service}]

            return f"🔄 Started syncing with {service.title()}. This may take a moment.", actions, result

        except Exception as e:
            return f"Sync error: {str(e)}", [], None

    def _handle_help(self, parameters: Dict) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle help requests"""
        topic = parameters.get("topic", "general")

        help_content = {
            "general": """
🤖 **AI Assistant Help**

I can help you with:
• 🔍 **Search**: Find information in your knowledge base
• 📝 **Tasks**: Create, manage, and track tasks
• 📊 **Productivity**: View stats and progress
• 📅 **Schedule**: Check your calendar and agenda
• 🔄 **Sync**: Integrate with Todoist and other services

Try saying things like:
- "Search for AI information"
- "Create a task to review documents"
- "Show my productivity stats"
- "What's my schedule today?"
- "Sync with Todoist"
""",
            "search": """
🔍 **Search Help**

I can search your knowledge base using advanced vector similarity.

Examples:
- "Search for machine learning"
- "Find information about Docker"
- "What is RAG?"
- "Tell me about microservices"
""",
            "tasks": """
📝 **Task Management Help**

I can help you manage tasks with Todoist integration.

Examples:
- "Create a task to review AI paper"
- "Add a high priority task due tomorrow"
- "Show my pending tasks"
- "What's due today?"
"""
        }

        content = help_content.get(topic.lower(), help_content["general"])
        actions = [{"type": "help_shown", "topic": topic}]

        return content, actions, None

    def _handle_general_response(self, message: str, intent: Intent) -> Tuple[str, List[Dict], Optional[Dict]]:
        """Handle general conversation"""
        responses = [
            "I'm here to help! I can search for information, manage your tasks, show productivity stats, and more.",
            "How can I assist you today? Try asking me to search for something or show your tasks.",
            "I'm your AI assistant. I can help with knowledge search, task management, productivity tracking, and scheduling.",
            "Feel free to ask me to search for information, create tasks, or check your schedule!",
            "I'm ready to help! You can ask me about your tasks, search for information, or get productivity insights."
        ]

        import random
        response = random.choice(responses)

        if intent.confidence < 0.3:
            response += f"\n\nI'm not sure exactly what you meant by '{message}'. Could you be more specific?"

        actions = [{"type": "general_response", "original_message": message}]

        return response, actions, None

# Global smart assistant instance
smart_assistant = None

def get_smart_assistant(rag_service_url: str, personal_service_url: str) -> SmartAssistant:
    """Get or create smart assistant instance"""
    global smart_assistant
    if not smart_assistant:
        smart_assistant = SmartAssistant(rag_service_url, personal_service_url)
    return smart_assistant