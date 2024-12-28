import logging
import sqlite3
import requests
import time
import os
from dotenv import load_dotenv
from telegram import Update, User
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Bot & LM Studio Config
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set. Please set it in the .env file.")
  
DB_FILE = "conversations.db"

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODELS_URL = f"{LM_STUDIO_BASE_URL}/models"
LM_STUDIO_CHAT_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/chat/completions"
LM_STUDIO_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/completions"
LM_STUDIO_EMBEDDINGS_URL = f"{LM_STUDIO_BASE_URL}/embeddings"

DEFAULT_MODEL = "llama-3.2-3b-instruct"
TOKEN_THRESHOLD = 7975

# Main conversation parameters (extend as needed)
conversation_params = {
    "max_tokens": 300,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 40,
    "stop": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": None,
    "logit_bias": {},
    "seed": None,
}

# ------------------------------------------------------------------------------
# Database Initialization & CRUD
# ------------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    with conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                last_seen INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                default_model TEXT,
                active_conversation_id INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                conversation_name TEXT NOT NULL,
                model TEXT,
                system_prompt TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
    conn.close()

def upsert_user(telegram_user: User):
    user_id = telegram_user.id
    username = telegram_user.username or ""
    first_name = telegram_user.first_name or ""
    last_name = telegram_user.last_name or ""
    last_seen = int(time.time())
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO users (user_id, username, first_name, last_name, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username=excluded.username,
                first_name=excluded.first_name,
                last_name=excluded.last_name,
                last_seen=excluded.last_seen
        """, (user_id, username, first_name, last_name, last_seen))
        c.execute("""
            INSERT INTO user_settings (user_id, default_model, active_conversation_id)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO NOTHING
        """, (user_id, DEFAULT_MODEL, None))

def get_user_settings(user_id: int) -> dict:
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT default_model, active_conversation_id FROM user_settings WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        if row:
            return {"default_model": row[0], "active_conversation_id": row[1]}
    return {"default_model": DEFAULT_MODEL, "active_conversation_id": None}

def set_user_setting(user_id: int, field: str, value):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        query = f"UPDATE user_settings SET {field} = ? WHERE user_id = ?"
        c.execute(query, (value, user_id))

def create_conversation(user_id: int, name: str, model: str = None) -> int:
    if not model:
        model = get_user_settings(user_id)["default_model"]
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO user_conversations (user_id, conversation_name, model, system_prompt)
            VALUES (?, ?, ?, ?)
        """, (user_id, name, model, ""))
        return c.lastrowid

def get_user_conversations(user_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT conversation_id, conversation_name, model, system_prompt
            FROM user_conversations
            WHERE user_id = ?
            ORDER BY conversation_id ASC
        """, (user_id,))
        rows = c.fetchall()
    return [
        {
            "conversation_id": r[0],
            "conversation_name": r[1],
            "model": r[2],
            "system_prompt": r[3]
        }
        for r in rows
    ]

def switch_conversation(user_id: int, conversation_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT conversation_id FROM user_conversations
            WHERE conversation_id = ? AND user_id = ?
        """, (conversation_id, user_id))
        if c.fetchone():
            set_user_setting(user_id, "active_conversation_id", conversation_id)
            return True
    return False

def update_conversation_model(conversation_id: int, model: str):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("UPDATE user_conversations SET model = ? WHERE conversation_id = ?", (model, conversation_id))

def update_conversation_system_prompt(conversation_id: int, prompt: str):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("UPDATE user_conversations SET system_prompt = ? WHERE conversation_id = ?", (prompt, conversation_id))

def get_messages(conversation_id: int) -> list:
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        return [{"role": r[0], "content": r[1]} for r in c.fetchall()]

def append_message(conversation_id: int, role: str, content: str):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, int(time.time()))
        )

def clear_conversation_messages(conversation_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

def append_summary(conversation_id: int, summary: str):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversation_summary (conversation_id, summary, timestamp)
            VALUES (?, ?, ?)
        """, (conversation_id, summary, int(time.time())))

def get_summaries(conversation_id: int) -> list:
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT summary FROM conversation_summary
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        return [r[0] for r in c.fetchall()]

# ------------------------------------------------------------------------------
# LM Studio API Calls
# ------------------------------------------------------------------------------
def list_models() -> dict:
    try:
        r = requests.get(LM_STUDIO_MODELS_URL)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.error(f"Error listing models: {e}")
        return {"error": str(e)}

def call_lm_studio_chat(messages: list, model: str) -> dict:
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": conversation_params["max_tokens"],
            "temperature": conversation_params["temperature"],
            "top_p": conversation_params["top_p"],
            "top_k": conversation_params["top_k"],
            "presence_penalty": conversation_params["presence_penalty"],
            "frequency_penalty": conversation_params["frequency_penalty"],
            "logit_bias": conversation_params["logit_bias"],
        }
        if conversation_params["stop"] is not None:
            payload["stop"] = conversation_params["stop"]
        if conversation_params["repeat_penalty"] is not None:
            payload["repeat_penalty"] = conversation_params["repeat_penalty"]
        if conversation_params["seed"] is not None:
            payload["seed"] = conversation_params["seed"]
        resp = requests.post(LM_STUDIO_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Error calling chat completions: {e}")
        return {"error": str(e)}

def call_lm_studio_completions(prompt: str, model: str) -> dict:
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": conversation_params["max_tokens"],
            "temperature": conversation_params["temperature"],
            "top_p": conversation_params["top_p"],
            "top_k": conversation_params["top_k"],
            "presence_penalty": conversation_params["presence_penalty"],
            "frequency_penalty": conversation_params["frequency_penalty"],
            "logit_bias": conversation_params["logit_bias"],
        }
        if conversation_params["stop"] is not None:
            payload["stop"] = conversation_params["stop"]
        if conversation_params["repeat_penalty"] is not None:
            payload["repeat_penalty"] = conversation_params["repeat_penalty"]
        if conversation_params["seed"] is not None:
            payload["seed"] = conversation_params["seed"]
        resp = requests.post(LM_STUDIO_COMPLETIONS_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Error calling completions: {e}")
        return {"error": str(e)}

def call_lm_studio_embeddings(text: str, model: str) -> dict:
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "input": [text]}
        resp = requests.post(LM_STUDIO_EMBEDDINGS_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Error calling embeddings: {e}")
        return {"error": str(e)}

def summarize_conversation(conversation_id: int, model: str) -> str:
    msgs = get_messages(conversation_id)
    if not msgs:
        return "Nothing to summarize."
    joined = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])
    prompt = f"Summarize:\n\n{joined}\n\nSummary:"
    data = call_lm_studio_completions(prompt, model)
    if "error" in data:
        return "Summary error."
    try:
        return data["choices"][0]["text"].strip()
    except:
        return "Failed to extract summary."

# ------------------------------------------------------------------------------
# Telegram Handlers
# ------------------------------------------------------------------------------
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text
    upsert_user(user)

    s = get_user_settings(user.id)
    cid = s["active_conversation_id"]
    if not cid:
        cid = create_conversation(user.id, "Default Thread")
        set_user_setting(user.id, "active_conversation_id", cid)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    append_message(cid, "user", text)

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT model, system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
        row = c.fetchone()

    model = row[0] if row and row[0] else s["default_model"]
    system_prompt = row[1] if row and row[1] else ""

    msgs = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})
    for m in get_messages(cid):
        msgs.append({"role": m["role"], "content": m["content"]})

    data = call_lm_studio_chat(msgs, model)
    if "error" in data:
        await update.message.reply_text(
            f"API Error: {data['error']}", parse_mode=ParseMode.MARKDOWN
        )
        return

    try:
        assistant_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
    except:
        await update.message.reply_text(
            "No content in response.", parse_mode=ParseMode.MARKDOWN
        )
        return

    append_message(cid, "assistant", assistant_text)

    usage_msg = ""
    if usage:
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens > TOKEN_THRESHOLD:
            # Summarize automatically
            summary = summarize_conversation(cid, model)
            append_summary(cid, summary)
            clear_conversation_messages(cid)
            usage_msg = "\n\n*Context summarized and reset.*"

    await update.message.reply_text(
        assistant_text + usage_msg, parse_mode=ParseMode.MARKDOWN
    )

async def summarize_thread_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args

    # If user typed /summarize_thread <id>, use that ID; else active conversation
    cid = None
    if args:
        try:
            cid = int(args[0])
        except ValueError:
            await update.message.reply_text(
                "*Invalid conversation ID.*", parse_mode=ParseMode.MARKDOWN
            )
            return

    if not cid:
        s = get_user_settings(user_id)
        cid = s["active_conversation_id"]

    if not cid:
        await update.message.reply_text(
            "*No active conversation or invalid ID.*", parse_mode=ParseMode.MARKDOWN
        )
        return

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT model FROM user_conversations WHERE conversation_id = ?", (cid,))
        row = c.fetchone()

    if not row:
        await update.message.reply_text(
            "*Conversation not found.*", parse_mode=ParseMode.MARKDOWN
        )
        return

    model = row[0] if row[0] else DEFAULT_MODEL

    summary_text = summarize_conversation(cid, model)
    if summary_text.startswith("Summary error") or summary_text.startswith("Failed"):
        await update.message.reply_text(
            f"*Error summarizing conversation {cid}.*", parse_mode=ParseMode.MARKDOWN
        )
        return

    # Store summary
    append_summary(cid, summary_text)

    await update.message.reply_text(
        f"*Summary for conversation {cid}:*\n\n{summary_text}",
        parse_mode=ParseMode.MARKDOWN
    )

async def set_parameter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: `/set <param> <value>`", parse_mode=ParseMode.MARKDOWN
        )
        return

    param = args[0].lower()
    val = " ".join(args[1:])
    if param in conversation_params:
        try:
            if param in ["max_tokens", "top_k"]:
                conversation_params[param] = int(val)
            elif param in ["temperature", "top_p", "presence_penalty", "frequency_penalty"]:
                conversation_params[param] = float(val)
            elif param == "stop":
                conversation_params["stop"] = None if val.lower() == "none" else val
            elif param == "repeat_penalty":
                conversation_params["repeat_penalty"] = None if val.lower() == "none" else float(val)
            elif param == "seed":
                conversation_params["seed"] = None if val.lower() == "none" else int(val)
            else:
                conversation_params[param] = val
            await update.message.reply_text(
                f"*Set* `{param}` *to* `{conversation_params[param]}`",
                parse_mode=ParseMode.MARKDOWN
            )
        except:
            await update.message.reply_text(
                f"Invalid value for `{param}`: `{val}`", parse_mode=ParseMode.MARKDOWN
            )
    else:
        await update.message.reply_text(
            f"Unknown parameter: `{param}`", parse_mode=ParseMode.MARKDOWN
        )

async def show_parameters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [f"`{k}` = `{v}`" for k, v in conversation_params.items()]
    formatted = "\n".join(lines)
    msg = "*Current parameters:*\n" + formatted
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def clear_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_user_settings(user_id)
    cid = s["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active context to clear.*", parse_mode=ParseMode.MARKDOWN
        )
        return
    clear_conversation_messages(cid)
    await update.message.reply_text(
        "*Context cleared.*", parse_mode=ParseMode.MARKDOWN
    )

async def show_summaries_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cid = get_user_settings(user_id)["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active conversation.*", parse_mode=ParseMode.MARKDOWN
        )
        return
    sums = get_summaries(cid)
    if sums:
        bullet_sums = "\n\n".join(f" {s}" for s in sums)
        await update.message.reply_text(
            f"*Summaries for Conversation {cid}:*\n\n{bullet_sums}",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text("*No summaries found.*", parse_mode=ParseMode.MARKDOWN)

async def new_conversation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    name = " ".join(context.args) or "Unnamed"
    conv_id = create_conversation(user_id, name)
    switch_conversation(user_id, conv_id)
    await update.message.reply_text(
        f"New conversation *'{name}'* created.\nSwitched to conversation *ID={conv_id}*",
        parse_mode=ParseMode.MARKDOWN
    )

async def list_threads_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    convs = get_user_conversations(user_id)
    s = get_user_settings(user_id)

    if not convs:
        await update.message.reply_text("*No conversations.*", parse_mode=ParseMode.MARKDOWN)
        return

    lines = []
    for c in convs:
        cid = c["conversation_id"]
        cname_escaped = c["conversation_name"].replace("_", "\\_")
        active_prefix = "**(active)** " if cid == s["active_conversation_id"] else ""
        lines.append(
            f"{active_prefix}**ID {cid}**: [{cname_escaped}](/{'switch_thread'} {cid}) -> Model: `{c['model']}`"
        )

    out_text = "*Your conversations:*\n\n" + "\n".join(lines)
    await update.message.reply_text(out_text, parse_mode=ParseMode.MARKDOWN)

async def switch_thread_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "Usage: `/switch_thread <id>`", parse_mode=ParseMode.MARKDOWN
        )
        return

    try:
        cid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("*Invalid conversation ID.*", parse_mode=ParseMode.MARKDOWN)
        return

    if switch_conversation(user_id, cid):
        await update.message.reply_text(
            f"Switched to conversation *ID={cid}*", parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            "*Not found or not owned by you.*", parse_mode=ParseMode.MARKDOWN
        )

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cid = get_user_settings(user_id)["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active conversation.*", parse_mode=ParseMode.MARKDOWN
        )
        return
    if not context.args:
        await update.message.reply_text(
            "Usage: `/set_model <model_name>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    model = " ".join(context.args)
    update_conversation_model(cid, model)
    await update.message.reply_text(
        f"Conversation *{cid}* model set to `{model}`", parse_mode=ParseMode.MARKDOWN
    )

async def set_system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cid = get_user_settings(user_id)["active_conversation_id"]
    if not cid:
        await update.message.reply_text("*No active conversation.*", parse_mode=ParseMode.MARKDOWN)
        return
    prompt = " ".join(context.args)
    update_conversation_system_prompt(cid, prompt)
    await update.message.reply_text("*System prompt updated.*", parse_mode=ParseMode.MARKDOWN)

async def show_system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_user_settings(user_id)
    cid = s["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active conversation.*", parse_mode=ParseMode.MARKDOWN
        )
        return

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
        row = c.fetchone()

    if row and row[0]:
        system_prompt = row[0]
        await update.message.reply_text(
            f"*System Prompt for conversation {cid}:*\n\n```{system_prompt}```",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text("*No system prompt set.*", parse_mode=ParseMode.MARKDOWN)

async def list_models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    r = list_models()
    if "error" in r:
        await update.message.reply_text(f"*Error:* {r['error']}", parse_mode=ParseMode.MARKDOWN)
        return
    if "data" in r:
        models = [m["id"] for m in r["data"]]
        lines = "\n".join(f"- `{m}`" for m in models)
        await update.message.reply_text(f"*Models loaded:*\n\n{lines}", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("*Unexpected /v1/models response.*", parse_mode=ParseMode.MARKDOWN)

async def completion_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt = " ".join(context.args)
    if not prompt.strip():
        await update.message.reply_text(
            "Usage: `/completion <prompt>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    s = get_user_settings(user_id)
    cid = s["active_conversation_id"]
    model = s["default_model"]
    if cid:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT model FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
            if row and row[0]:
                model = row[0]

    await update.message.reply_text(
        f"Requesting completion with model: `{model}`...",
        parse_mode=ParseMode.MARKDOWN
    )
    data = call_lm_studio_completions(prompt, model)
    if "error" in data:
        await update.message.reply_text(f"*Error:* {data['error']}", parse_mode=ParseMode.MARKDOWN)
        return

    try:
        txt = data["choices"][0]["text"]
        if not txt.strip():
            txt = "*No content returned.*"
        await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)
    except:
        await update.message.reply_text("*No text in response.*", parse_mode=ParseMode.MARKDOWN)

async def embedding_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    txt = " ".join(context.args)
    if not txt.strip():
        await update.message.reply_text(
            "Usage: `/embedding <text>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    s = get_user_settings(user_id)
    cid = s["active_conversation_id"]
    model = s["default_model"]
    if cid:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT model FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
            if row and row[0]:
                model = row[0]

    await update.message.reply_text(
        f"Requesting embedding with model: `{model}`...",
        parse_mode=ParseMode.MARKDOWN
    )
    data = call_lm_studio_embeddings(txt, model)
    if "error" in data:
        await update.message.reply_text(f"*Error:* {data['error']}", parse_mode=ParseMode.MARKDOWN)
        return

    try:
        emb = data["data"][0]["embedding"]
        truncated = emb[:10]
        emb_preview = ", ".join(str(x) for x in truncated)
        await update.message.reply_text(
            f"*Embedding (first 10 values):* `{emb_preview}`",
            parse_mode=ParseMode.MARKDOWN
        )
    except:
        await update.message.reply_text("*No embedding data returned.*", parse_mode=ParseMode.MARKDOWN)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("set", set_parameter))
    app.add_handler(CommandHandler("show_params", show_parameters))
    app.add_handler(CommandHandler("clear_context", clear_context_command))
    app.add_handler(CommandHandler("show_summaries", show_summaries_command))
    app.add_handler(CommandHandler("new_thread", new_conversation_command))
    app.add_handler(CommandHandler("list_threads", list_threads_command))
    app.add_handler(CommandHandler("switch_thread", switch_thread_command))
    app.add_handler(CommandHandler("set_model", set_model_command))
    app.add_handler(CommandHandler("set_system_prompt", set_system_prompt_command))
    app.add_handler(CommandHandler("show_system_prompt", show_system_prompt_command))
    app.add_handler(CommandHandler("list_models", list_models_command))
    app.add_handler(CommandHandler("completion", completion_command))
    app.add_handler(CommandHandler("embedding", embedding_command))
    app.add_handler(CommandHandler("summarize_thread", summarize_thread_command))  # New command

    # Fallback chat messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    app.run_polling()

if __name__ == "__main__":
    main()

