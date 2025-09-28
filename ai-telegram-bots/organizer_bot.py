#!/usr/bin/env python3
"""
Organizer Service Telegram Bot
Personal assistant bot for managing tasks, contacts, and calendar events
"""

import os
import asyncio
import aiohttp
import logging
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ORGANIZER_SERVICE_URL = os.getenv("ORGANIZER_SERVICE_URL", "http://localhost:8002")

if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is required!")
    logger.info("Get a bot token from @BotFather on Telegram")
    logger.info("Then set: export TELEGRAM_BOT_TOKEN='your_token_here'")
    exit(1)

class OrganizerBot:
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

    async def check_health(self) -> bool:
        """Check if Organizer service is healthy"""
        try:
            session = await self.get_session()
            async with session.get(f"{ORGANIZER_SERVICE_URL}/health", timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Organizer health check failed: {e}")
            return False

    async def process_natural_language(self, message: str) -> dict:
        """Process natural language with the assistant"""
        try:
            session = await self.get_session()
            payload = {"message": message}
            async with session.post(
                f"{ORGANIZER_SERVICE_URL}/chat",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Processing failed with status {response.status}"}
        except Exception as e:
            logger.error(f"Natural language processing failed: {e}")
            return {"error": str(e)}

    async def get_contacts(self, limit: int = 10) -> dict:
        """Get contacts from organizer service"""
        try:
            session = await self.get_session()
            async with session.get(f"{ORGANIZER_SERVICE_URL}/api/contacts?limit={limit}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to get contacts: {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def get_todos(self, limit: int = 10) -> dict:
        """Get todos from organizer service"""
        try:
            session = await self.get_session()
            async with session.get(f"{ORGANIZER_SERVICE_URL}/api/todos?limit={limit}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to get todos: {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def get_events(self, days: int = 7) -> dict:
        """Get upcoming events"""
        try:
            session = await self.get_session()
            async with session.get(f"{ORGANIZER_SERVICE_URL}/api/events?days={days}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to get events: {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def create_contact(self, name: str, email: str = "", phone: str = "") -> dict:
        """Create a new contact"""
        try:
            session = await self.get_session()
            payload = {"name": name, "email": email, "phone": phone}
            async with session.post(f"{ORGANIZER_SERVICE_URL}/api/contacts", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to create contact: {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def create_todo(self, title: str, priority: str = "medium") -> dict:
        """Create a new todo"""
        try:
            session = await self.get_session()
            payload = {"title": title, "priority": priority}
            async with session.post(f"{ORGANIZER_SERVICE_URL}/api/todos", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to create todo: {response.status}"}
        except Exception as e:
            return {"error": str(e)}

# Initialize bot
organizer_bot = OrganizerBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = """
🗂️ *Personal Organizer Bot*

I can help you manage:
📅 Events and calendar (/events)
✅ Tasks and todos (/todos)
👥 Contacts (/contacts)
🤖 Natural language commands (just talk to me!)

*Quick commands:*
/status - Check service health
/today - Today's overview
/add - Quick add menu

Just send me messages like:
• "Schedule meeting with John tomorrow at 3pm"
• "Add task: Buy groceries"
• "Add contact: Sarah, sarah@email.com"
"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check organizer service status"""
    is_healthy = await organizer_bot.check_health()

    if is_healthy:
        status_text = "✅ Organizer Service is healthy and ready!"
    else:
        status_text = "❌ Organizer Service is not responding. Please check if it's running on port 8002."

    await update.message.reply_text(status_text)

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's overview"""
    await update.message.reply_text("📊 Loading today's overview...")

    # Get events, todos, and contacts
    events = await organizer_bot.get_events(1)
    todos = await organizer_bot.get_todos(5)

    response = "📅 *Today's Overview:*\n\n"

    # Events
    if "error" not in events and events.get("events"):
        response += "🗓️ *Events Today:*\n"
        for event in events["events"][:3]:
            title = event.get("title", "Untitled")
            time = event.get("start_time", "")
            response += f"• {title} at {time}\n"
        response += "\n"
    else:
        response += "🗓️ No events today\n\n"

    # Todos
    if "error" not in todos and todos.get("todos"):
        response += "✅ *Pending Tasks:*\n"
        for todo in todos["todos"][:5]:
            title = todo.get("title", "Untitled")
            priority = todo.get("priority", "medium")
            emoji = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
            response += f"{emoji} {title}\n"
    else:
        response += "✅ No pending tasks\n"

    await update.message.reply_text(response, parse_mode='Markdown')

async def events_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show upcoming events"""
    result = await organizer_bot.get_events(7)

    if "error" in result:
        await update.message.reply_text(f"❌ Failed to get events: {result['error']}")
        return

    if not result.get("events"):
        await update.message.reply_text("📅 No upcoming events in the next 7 days.")
        return

    response = "📅 *Upcoming Events:*\n\n"
    for event in result["events"][:10]:
        title = event.get("title", "Untitled")
        start_time = event.get("start_time", "")
        location = event.get("location", "")
        response += f"• *{title}*\n  📅 {start_time}\n"
        if location:
            response += f"  📍 {location}\n"
        response += "\n"

    await update.message.reply_text(response, parse_mode='Markdown')

async def todos_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show pending todos"""
    result = await organizer_bot.get_todos(10)

    if "error" in result:
        await update.message.reply_text(f"❌ Failed to get todos: {result['error']}")
        return

    if not result.get("todos"):
        await update.message.reply_text("✅ No pending tasks!")
        return

    response = "✅ *Pending Tasks:*\n\n"
    for todo in result["todos"]:
        title = todo.get("title", "Untitled")
        priority = todo.get("priority", "medium")
        due_date = todo.get("due_date", "")

        # Priority emoji
        emoji = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"

        response += f"{emoji} *{title}*\n"
        if due_date:
            response += f"  📅 Due: {due_date}\n"
        response += "\n"

    await update.message.reply_text(response, parse_mode='Markdown')

async def contacts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show contacts"""
    result = await organizer_bot.get_contacts(10)

    if "error" in result:
        await update.message.reply_text(f"❌ Failed to get contacts: {result['error']}")
        return

    if not result.get("contacts"):
        await update.message.reply_text("👥 No contacts found.")
        return

    response = "👥 *Your Contacts:*\n\n"
    for contact in result["contacts"]:
        name = contact.get("name", "Unknown")
        email = contact.get("email", "")
        phone = contact.get("phone", "")

        response += f"👤 *{name}*\n"
        if email:
            response += f"  ✉️ {email}\n"
        if phone:
            response += f"  📞 {phone}\n"
        response += "\n"

    await update.message.reply_text(response, parse_mode='Markdown')

async def add_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show quick add menu"""
    keyboard = [
        [InlineKeyboardButton("📅 Add Event", callback_data="add_event")],
        [InlineKeyboardButton("✅ Add Task", callback_data="add_task")],
        [InlineKeyboardButton("👤 Add Contact", callback_data="add_contact")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🔧 *Quick Add Menu*\nWhat would you like to add?",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callbacks"""
    query = update.callback_query
    await query.answer()

    if query.data == "add_event":
        await query.edit_message_text(
            "📅 To add an event, send me a message like:\n"
            "_\"Schedule meeting with John tomorrow at 3pm\"_\n"
            "_\"Weekly standup every Monday at 9am\"_",
            parse_mode='Markdown'
        )
    elif query.data == "add_task":
        await query.edit_message_text(
            "✅ To add a task, send me a message like:\n"
            "_\"Add task: Buy groceries\"_\n"
            "_\"High priority: Finish report by Friday\"_",
            parse_mode='Markdown'
        )
    elif query.data == "add_contact":
        await query.edit_message_text(
            "👤 To add a contact, send me a message like:\n"
            "_\"Add contact: John Doe, john@example.com, +1234567890\"_\n"
            "_\"Save Sarah's info: sarah@email.com\"_",
            parse_mode='Markdown'
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle natural language messages"""
    message = update.message.text

    if len(message) < 3:
        await update.message.reply_text("Please send a more detailed message.")
        return

    await update.message.reply_text("🤔 Processing your request...")

    result = await organizer_bot.process_natural_language(message)

    if "error" in result:
        await update.message.reply_text(f"❌ Processing failed: {result['error']}")
        return

    # Format response based on result
    if "response" in result:
        response = f"✅ {result['response']}"

        if "actions_taken" in result and result["actions_taken"]:
            response += "\n\n🎯 *Actions taken:*\n"
            for action in result["actions_taken"]:
                action_type = action.get("type", "unknown")
                details = action.get("details", "")
                response += f"• {action_type}: {details}\n"
    else:
        response = "I understood your message but couldn't take any specific action."

    # Split long messages
    if len(response) > 4000:
        response = response[:4000] + "...\n\n_Response truncated_"

    await update.message.reply_text(response, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = """
🗂️ *Organizer Bot Commands:*

/start - Welcome message
/status - Check service health
/today - Today's overview
/events - Upcoming events
/todos - Pending tasks
/contacts - Your contacts
/add - Quick add menu
/help - Show this help

*Natural Language Examples:*
• "Schedule meeting with John tomorrow at 3pm"
• "Add task: Buy groceries, high priority"
• "Add contact: Sarah, sarah@example.com"
• "What do I have today?"
• "Show my high priority tasks"

Just talk to me naturally! 🤖
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main():
    """Main function"""
    print("🗂️ Starting Organizer Telegram Bot...")
    print(f"📡 Connecting to Organizer service at: {ORGANIZER_SERVICE_URL}")

    # Create application
    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("today", today))
    application.add_handler(CommandHandler("events", events_command))
    application.add_handler(CommandHandler("todos", todos_command))
    application.add_handler(CommandHandler("contacts", contacts_command))
    application.add_handler(CommandHandler("add", add_menu))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start bot
    print("✅ Organizer Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        asyncio.run(organizer_bot.close_session())