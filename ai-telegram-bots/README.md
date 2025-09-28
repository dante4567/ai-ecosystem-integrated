# AI Ecosystem Telegram Bots

Three ready-to-use Telegram bots that connect to your AI ecosystem services.

## 🚀 Quick Start

### 1. Get Bot Token
```bash
# Message @BotFather on Telegram
# Create new bot: /newbot
# Get token: 123456789:ABC-DEF...
export TELEGRAM_BOT_TOKEN="your_token_here"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Choose Your Bot

**RAG Bot (Document Search):**
```bash
python rag_bot.py
```
Features: Document upload, search, AI chat with documents

**Organizer Bot (Personal Assistant):**
```bash
python organizer_bot.py
```
Features: Tasks, events, contacts, natural language commands

**Unified Bot (Complete Ecosystem):**
```bash
python unified_bot.py
```
Features: Everything combined with intelligent routing

## 📱 Bot Features

### RAG Bot Commands
- `/start` - Welcome and instructions
- `/status` - Check RAG service health
- `/search <query>` - Search documents
- `/help` - Show all commands
- Send documents to upload them
- Send messages to chat with documents

### Organizer Bot Commands
- `/start` - Welcome and overview
- `/status` - Check organizer service health
- `/today` - Today's overview
- `/todos` - Show pending tasks
- `/events` - Show upcoming events
- `/contacts` - Show contacts
- `/add` - Quick add menu
- Natural language: "Schedule meeting tomorrow at 3pm"

### Unified Bot Commands
- All commands from both bots
- Intelligent routing based on message content
- Unified search across documents and personal data
- Cross-service functionality

## ⚙️ Configuration

### Environment Variables
```bash
# Required
export TELEGRAM_BOT_TOKEN="your_bot_token"

# Service URLs (default values shown)
export RAG_SERVICE_URL="http://localhost:8001"
export ORGANIZER_SERVICE_URL="http://localhost:8002"
export GATEWAY_URL="http://localhost:8003"
```

### Service Requirements
- **RAG Bot**: Requires RAG service on port 8001
- **Organizer Bot**: Requires organizer service on port 8002
- **Unified Bot**: Requires both services + gateway on port 8003

## 🔧 Development

### Adding New Features
Each bot is self-contained and can be extended:

```python
async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Your command implementation
    await update.message.reply_text("Response")

# Add to application
application.add_handler(CommandHandler("newcmd", new_command))
```

### Error Handling
All bots include comprehensive error handling:
- Service unavailability fallbacks
- Network timeout handling
- User-friendly error messages
- Logging for debugging

## 📊 Testing

### Test Bot Functionality
```bash
# Send these messages to test
"Search for AI information"
"What is microservices architecture?"
"Schedule meeting tomorrow at 3pm"
"Add task: Buy groceries"
"Show my upcoming events"
```

### Monitor Service Health
All bots include `/status` commands to check service connectivity.

## 🚨 Troubleshooting

**Bot not responding:**
- Check TELEGRAM_BOT_TOKEN is correct
- Verify bot is running with no errors in console
- Test with /start command

**Service errors:**
- Check if required services are running
- Verify service URLs are accessible
- Use /status command to check connectivity

**Permission errors:**
- Ensure bot has message permissions in group chats
- Check bot privacy settings in @BotFather

---

**Ready to chat with your AI ecosystem via Telegram!** 🤖