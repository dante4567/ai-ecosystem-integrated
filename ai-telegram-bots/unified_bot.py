#!/usr/bin/env python3
"""
Unified AI Ecosystem Telegram Bot
Combines RAG service and Organizer service with intelligent routing
"""

import os
import asyncio
import aiohttp
import logging
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Document
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
ORGANIZER_SERVICE_URL = os.getenv("ORGANIZER_SERVICE_URL", "http://localhost:8002")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8003")

if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is required!")
    logger.info("Get a bot token from @BotFather on Telegram")
    logger.info("Then set: export TELEGRAM_BOT_TOKEN='your_token_here'")
    exit(1)

class UnifiedBot:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def check_services_health(self) -> dict:
        """Check health of all services"""
        results = {}
        session = await self.get_session()

        services = {
            "rag": RAG_SERVICE_URL,
            "organizer": ORGANIZER_SERVICE_URL,
            "gateway": GATEWAY_URL
        }

        for name, url in services.items():
            try:
                async with session.get(f"{url}/health", timeout=5) as response:
                    results[name] = response.status == 200
            except Exception:
                results[name] = False

        return results

    async def intelligent_query(self, message: str) -> dict:
        """Use gateway for intelligent routing or fallback to direct services"""
        session = await self.get_session()

        # Try gateway first (unified processing)
        try:
            payload = {"message": message, "include_actions": True}
            async with session.post(
                f"{GATEWAY_URL}/api/assistant",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    result["source"] = "gateway"
                    return result
        except Exception as e:
            logger.info(f"Gateway not available, using direct routing: {e}")

        # Fallback: Determine intent and route directly
        message_lower = message.lower()

        # Check for document-related keywords
        doc_keywords = ["search", "document", "find", "query", "knowledge", "pdf", "file"]
        organizer_keywords = ["task", "todo", "contact", "event", "calendar", "remind", "schedule", "meeting"]

        if any(keyword in message_lower for keyword in doc_keywords):
            return await self._search_documents(message)
        elif any(keyword in message_lower for keyword in organizer_keywords):
            return await self._process_organizer_request(message)
        else:
            # Default to organizer for general queries
            return await self._process_organizer_request(message)

    async def _search_documents(self, query: str) -> dict:
        """Search documents using RAG service"""
        try:
            session = await self.get_session()
            payload = {"question": query, "llm_model": "groq/llama-3.1-8b-instant"}
            async with session.post(
                f"{RAG_SERVICE_URL}/chat",
                json=payload,
                timeout=60
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    result["source"] = "rag"
                    return result
                else:
                    return {"error": f"RAG search failed: {response.status}", "source": "rag"}
        except Exception as e:
            return {"error": str(e), "source": "rag"}

    async def _process_organizer_request(self, message: str) -> dict:
        """Process organizer requests"""
        try:
            session = await self.get_session()
            payload = {"message": message}
            async with session.post(
                f"{ORGANIZER_SERVICE_URL}/chat",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    result["source"] = "organizer"
                    return result
                else:
                    return {"error": f"Organizer processing failed: {response.status}", "source": "organizer"}
        except Exception as e:
            return {"error": str(e), "source": "organizer"}

    async def unified_search(self, query: str) -> dict:
        """Search across both RAG and organizer services"""
        try:
            session = await self.get_session()
            payload = {
                "query": query,
                "include_personal": True,
                "include_context": True,
                "max_results": 5
            }
            async with session.post(
                f"{GATEWAY_URL}/api/search",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Unified search failed: {response.status}"}
        except Exception as e:
            logger.info(f"Gateway search not available: {e}")
            # Fallback to direct RAG search
            return await self._search_documents(query)

    async def upload_document(self, file_content: bytes, filename: str) -> dict:
        """Upload document to RAG service"""
        try:
            session = await self.get_session()
            data = aiohttp.FormData()
            data.add_field('file', file_content, filename=filename)

            async with session.post(
                f"{RAG_SERVICE_URL}/ingest/file",
                data=data,
                timeout=120
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Upload failed: {response.status}"}
        except Exception as e:
            return {"error": str(e)}

# Initialize bot
unified_bot = UnifiedBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = """
🤖 *AI Ecosystem Bot*

I combine your personal organizer and document search in one place!

*What I can do:*
📄 Search documents and knowledge base
📅 Manage calendar events and meetings
✅ Handle tasks and todos
👥 Organize contacts
🤖 Answer questions using AI

*How to use me:*
• Upload documents - I'll process them automatically
• Ask questions about your documents
• Natural language: "Schedule meeting tomorrow at 3pm"
• Search everything: /search your query here

*Quick commands:*
/status - Check all services
/search <query> - Search everything
/help - Show all commands

Just talk to me naturally! 🚀
"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check all services status"""
    await update.message.reply_text("🔍 Checking all services...")

    health = await unified_bot.check_services_health()

    status_text = "🏥 *Service Status:*\n\n"

    services = {
        "rag": "📄 RAG Service (Documents)",
        "organizer": "🗂️ Organizer Service (Tasks/Calendar)",
        "gateway": "🌉 Gateway (Unified Access)"
    }

    for service, description in services.items():
        if health.get(service, False):
            status_text += f"✅ {description}\n"
        else:
            status_text += f"❌ {description}\n"

    # Overall status
    all_working = all(health.values())
    some_working = any(health.values())

    if all_working:
        status_text += "\n🎉 All services operational!"
    elif some_working:
        status_text += "\n⚠️ Some services available, functionality may be limited"
    else:
        status_text += "\n🚨 No services responding - check deployments"

    await update.message.reply_text(status_text, parse_mode='Markdown')

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Unified search across all services"""
    if not context.args:
        await update.message.reply_text("Please provide a search query: /search your question here")
        return

    query = " ".join(context.args)
    await update.message.reply_text(f"🔍 Searching across all services for: *{query}*", parse_mode='Markdown')

    result = await unified_bot.unified_search(query)

    if "error" in result:
        await update.message.reply_text(f"❌ Search failed: {result['error']}")
        return

    response = "🔍 *Search Results:*\n\n"

    # RAG results
    if "rag_results" in result and result["rag_results"]:
        response += "📄 *From Documents:*\n"
        for i, doc in enumerate(result["rag_results"][:2], 1):
            content = doc.get("content", "No content")[:150]
            response += f"{i}. {content}...\n\n"

    # Personal context
    if "personal_context" in result and result["personal_context"]:
        response += "👤 *From Personal Data:*\n"
        for item in result["personal_context"][:2]:
            title = item.get("title", "Unknown")
            type_name = item.get("type", "item")
            response += f"• {type_name}: {title}\n"
        response += "\n"

    if len(response) == len("🔍 *Search Results:*\n\n"):
        response += "No results found across services."

    # Split long messages
    if len(response) > 4000:
        response = response[:4000] + "...\n\n_Results truncated_"

    await update.message.reply_text(response, parse_mode='Markdown')

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads"""
    document: Document = update.message.document

    if document.file_size > 10 * 1024 * 1024:  # 10MB limit
        await update.message.reply_text("❌ File too large. Maximum size is 10MB.")
        return

    await update.message.reply_text(f"📤 Processing {document.file_name}...")

    try:
        file = await context.bot.get_file(document.file_id)
        file_content = await file.download_as_bytearray()

        result = await unified_bot.upload_document(bytes(file_content), document.file_name)

        if "error" in result:
            await update.message.reply_text(f"❌ Upload failed: {result['error']}")
        else:
            await update.message.reply_text(
                f"✅ Successfully processed {document.file_name}!\n"
                f"🔍 You can now search for content from this document."
            )

    except Exception as e:
        await update.message.reply_text(f"❌ Upload error: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages with intelligent routing"""
    message = update.message.text

    if len(message) < 3:
        await update.message.reply_text("Please send a more detailed message.")
        return

    await update.message.reply_text("🤔 Processing your request...")

    result = await unified_bot.intelligent_query(message)

    if "error" in result:
        await update.message.reply_text(f"❌ Processing failed: {result['error']}")
        return

    # Format response based on source and content
    source = result.get("source", "unknown")
    source_emoji = {
        "gateway": "🌉",
        "rag": "📄",
        "organizer": "🗂️"
    }.get(source, "🤖")

    response = ""

    # Handle different response formats
    if "answer" in result:  # RAG response
        response = f"{source_emoji} *Answer:*\n{result['answer']}"
        if "sources" in result and result["sources"]:
            response += f"\n\n📚 Sources: {len(result['sources'])} documents"

    elif "response" in result:  # Organizer or gateway response
        response = f"{source_emoji} {result['response']}"

        if "actions_taken" in result and result["actions_taken"]:
            response += "\n\n🎯 *Actions:*\n"
            for action in result["actions_taken"]:
                action_type = action.get("type", "action")
                details = action.get("details", "")
                response += f"• {action_type}: {details}\n"

    else:
        response = f"{source_emoji} Processed your message but no specific response available."

    # Split long messages
    if len(response) > 4000:
        response = response[:4000] + "...\n\n_Response truncated_"

    await update.message.reply_text(response, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = """
🤖 *AI Ecosystem Bot Help*

*Commands:*
/start - Welcome and overview
/status - Check all service health
/search <query> - Search everything
/help - Show this help

*Natural Language Examples:*
📄 *Document queries:*
• "Search for information about machine learning"
• "What did the report say about Q3 revenue?"

🗂️ *Organizer tasks:*
• "Schedule meeting with John tomorrow at 3pm"
• "Add task: Buy groceries, high priority"
• "What do I have today?"

📤 *File uploads:*
• Send any document and I'll process it
• Supported: PDF, DOCX, TXT, MD, and more

🔍 *Smart routing:*
I automatically determine whether to search documents, manage tasks, or use both services based on your message!

*Service URLs:*
• RAG: {rag_url}
• Organizer: {organizer_url}
• Gateway: {gateway_url}
""".format(
        rag_url=RAG_SERVICE_URL,
        organizer_url=ORGANIZER_SERVICE_URL,
        gateway_url=GATEWAY_URL
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main():
    """Main function"""
    print("🤖 Starting Unified AI Ecosystem Bot...")
    print(f"📄 RAG Service: {RAG_SERVICE_URL}")
    print(f"🗂️ Organizer Service: {ORGANIZER_SERVICE_URL}")
    print(f"🌉 Gateway: {GATEWAY_URL}")

    # Create application
    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start bot
    print("✅ Unified Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        asyncio.run(unified_bot.close_session())