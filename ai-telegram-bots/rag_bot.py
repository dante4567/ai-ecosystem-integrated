#!/usr/bin/env python3
"""
RAG Service Telegram Bot
Simple bot that connects directly to the RAG service for document search and chat
"""

import os
import asyncio
import aiohttp
import logging
from typing import Optional
from telegram import Update, Document
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import json

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")

if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is required!")
    logger.info("Get a bot token from @BotFather on Telegram")
    logger.info("Then set: export TELEGRAM_BOT_TOKEN='your_token_here'")
    exit(1)

class RAGBot:
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

    async def check_rag_health(self) -> bool:
        """Check if RAG service is healthy"""
        try:
            session = await self.get_session()
            async with session.get(f"{RAG_SERVICE_URL}/health", timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"RAG health check failed: {e}")
            return False

    async def search_documents(self, query: str, max_results: int = 5) -> dict:
        """Search documents using RAG service"""
        try:
            session = await self.get_session()
            payload = {
                "text": query,
                "top_k": max_results
            }
            async with session.post(
                f"{RAG_SERVICE_URL}/search",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Search failed with status {response.status}"}
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {"error": str(e)}

    async def chat_with_rag(self, question: str, model: str = "groq/llama-3.1-8b-instant") -> dict:
        """Chat with RAG service"""
        try:
            session = await self.get_session()
            payload = {
                "question": question,
                "llm_model": model
            }
            async with session.post(
                f"{RAG_SERVICE_URL}/chat",
                json=payload,
                timeout=60
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Chat failed with status {response.status}"}
        except Exception as e:
            logger.error(f"RAG chat failed: {e}")
            return {"error": str(e)}

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
                    return {"error": f"Upload failed with status {response.status}"}
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            return {"error": str(e)}

# Initialize bot
rag_bot = RAGBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = """
🤖 *RAG Service Bot*

I can help you with:
📄 Upload documents (/upload)
🔍 Search documents (/search query)
💬 Chat with your documents (just send a message)
📊 Check service status (/status)

Send me a document to upload it, or ask me questions about your documents!
"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check RAG service status"""
    is_healthy = await rag_bot.check_rag_health()

    if is_healthy:
        status_text = "✅ RAG Service is healthy and ready!"
    else:
        status_text = "❌ RAG Service is not responding. Please check if it's running on port 8001."

    await update.message.reply_text(status_text)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Search documents command"""
    if not context.args:
        await update.message.reply_text("Please provide a search query: /search your question here")
        return

    query = " ".join(context.args)
    await update.message.reply_text(f"🔍 Searching for: *{query}*", parse_mode='Markdown')

    result = await rag_bot.search_documents(query)

    if "error" in result:
        await update.message.reply_text(f"❌ Search failed: {result['error']}")
        return

    if "results" in result and result["results"]:
        response = "📄 *Search Results:*\n\n"
        for i, doc in enumerate(result["results"][:3], 1):
            content = doc.get("content", "No content")[:200]
            score = doc.get("score", 0)
            response += f"{i}. Score: {score:.3f}\n{content}...\n\n"
    else:
        response = "No documents found matching your query."

    await update.message.reply_text(response, parse_mode='Markdown')

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads"""
    document: Document = update.message.document

    if document.file_size > 10 * 1024 * 1024:  # 10MB limit
        await update.message.reply_text("❌ File too large. Maximum size is 10MB.")
        return

    await update.message.reply_text(f"📤 Uploading {document.file_name}...")

    try:
        file = await context.bot.get_file(document.file_id)
        file_content = await file.download_as_bytearray()

        result = await rag_bot.upload_document(bytes(file_content), document.file_name)

        if "error" in result:
            await update.message.reply_text(f"❌ Upload failed: {result['error']}")
        else:
            await update.message.reply_text(f"✅ Successfully uploaded {document.file_name}!")

    except Exception as e:
        await update.message.reply_text(f"❌ Upload error: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages as chat queries"""
    question = update.message.text

    if len(question) < 3:
        await update.message.reply_text("Please ask a more detailed question.")
        return

    await update.message.reply_text("🤔 Thinking...")

    result = await rag_bot.chat_with_rag(question)

    if "error" in result:
        await update.message.reply_text(f"❌ Chat failed: {result['error']}")
        return

    if "answer" in result:
        response = f"💡 *Answer:*\n{result['answer']}"

        if "sources" in result and result["sources"]:
            response += f"\n\n📚 *Sources:* {len(result['sources'])} documents used"
    else:
        response = "I couldn't find an answer to your question."

    # Split long messages
    if len(response) > 4000:
        response = response[:4000] + "...\n\n_Response truncated_"

    await update.message.reply_text(response, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = """
🤖 *RAG Bot Commands:*

/start - Welcome message
/status - Check service health
/search <query> - Search documents
/help - Show this help

*Usage:*
• Send documents to upload them
• Send text messages to chat with your documents
• Use /search for quick document searches

*Supported formats:* PDF, DOCX, TXT, MD, and more!
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main():
    """Main function"""
    print("🤖 Starting RAG Telegram Bot...")
    print(f"📡 Connecting to RAG service at: {RAG_SERVICE_URL}")

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
    print("✅ RAG Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        asyncio.run(rag_bot.close_session())