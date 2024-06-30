import os
import logging
import structlog
import json
import sys
import datetime
import time
import tiktoken
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
import chromadb
import httpx


# Константы
OLLAMA_MODEL = "gemma2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 360.0
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMT = "Отвечай на русском языке!"

MODELS = {
    "Gemma 2": OLLAMA_MODEL,
    "GPT-3.5": "gpt-3.5-turbo",
    "GPT-4": "gpt-4o",
}

PRICE_PER_1K_TOKENS = {
    OLLAMA_MODEL: 0.0,
    "gpt-3.5-turbo": 0.002,
    "gpt-4o": 0.06,
}

def setup_logging():
    # Отключение логов httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("bot.log", encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)])

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()

def initialize_index():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("docs_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model

    persist_dir = "./stored_index"

    if os.path.exists(persist_dir):
        logger.info("Loading existing index")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
        index = load_index_from_storage(storage_context)
    else:
        logger.info("Creating new index")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = SimpleDirectoryReader('docs').load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir=persist_dir)

    # Создаем движок запросов с текущими настройками
    query_engine = index.as_query_engine()

    return query_engine

def get_model_keyboard(selected_model=None):
    keyboard = []
    for model in MODELS.keys():
        button_text = f"✅ {model}" if model == selected_model else model
        keyboard.append(KeyboardButton(button_text))
    return ReplyKeyboardMarkup([keyboard], resize_keyboard=True)

def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(string))
    except KeyError:
        # Если модель неизвестна tiktoken, используем примерный подсчет
        return len(string.split())  # Примерный подсчет по словам

def get_llm(model):
    if model == OLLAMA_MODEL:
        return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=OLLAMA_REQUEST_TIMEOUT)
    elif model in ["gpt-3.5-turbo", "gpt-4o"]:
        return OpenAI(model=model, api_key=OPENAI_API_KEY)
    else:
        return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.info("Received /start command", user_id=user.id, username=user.username)
    await update.message.reply_text(
        'Привет! Я бот, который может отвечать на вопросы. Выберите модель:',
        reply_markup=get_model_keyboard()
    )

def process_model_selection(message):
    selected_model = message.replace("✅ ", "")
    if selected_model in MODELS:
        return selected_model
    return None

async def generate_response(message: str, model: str, query_engine):
    start_time = time.time()

    message += PROMT
    
    llm = get_llm(MODELS[model])
    if llm:
        Settings.llm = llm
    
    response = query_engine.query(message)
    response_text = str(response)

    end_time = time.time()
    response_time = end_time - start_time

    if model == "Gemma 2":
        # Для Ollama используем примерный подсчет токенов
        input_tokens = len(message.split())
        output_tokens = len(response_text.split())
    else:
        input_tokens = num_tokens_from_string(message, MODELS[model])
        output_tokens = num_tokens_from_string(response_text, MODELS[model])
    
    cost = (input_tokens + output_tokens) / 1000 * PRICE_PER_1K_TOKENS.get(MODELS[model], 0.0)

    return response_text, response_time, input_tokens, output_tokens, cost

def create_log_entry(user, model, message, response_text, response_time, input_tokens, output_tokens, cost):
    return {
        "user_id": user.id,
        "username": user.username,
        "model_used": model,
        "question": message,
        "response": response_text,
        "response_time": response_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": cost,
        "timestamp": datetime.datetime.now().isoformat()
    }

def format_response_info(model, response_time, input_tokens, output_tokens, cost):
    return (f"\n\nМодель: {model}\nВремя ответа: {response_time:.2f} сек.\n"
            f"Токены: {input_tokens} (вход) + {output_tokens} (выход)\n"
            f"Примерная стоимость: ${cost:.5f}")

async def send_response_in_chunks(update, response_text, user_id):
    MAX_MESSAGE_LENGTH = 4000
    for i in range(0, len(response_text), MAX_MESSAGE_LENGTH):
        chunk = response_text[i:i+MAX_MESSAGE_LENGTH]
        if i + MAX_MESSAGE_LENGTH >= len(response_text):
            keyboard = [
                [InlineKeyboardButton("👍 +1", callback_data=f"rate_up_{user_id}"),
                 InlineKeyboardButton("👎 -1", callback_data=f"rate_down_{user_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(chunk, reply_markup=reply_markup)
        else:
            await update.message.reply_text(chunk)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message = update.message.text

    logger.info("Received message", user_id=user.id, username=user.username, message=message)

    selected_model = process_model_selection(message)
    if selected_model:
        context.user_data['model'] = selected_model
        await update.message.reply_text(
            f"Выбрана модель: {selected_model}. Задайте ваш вопрос.",
            reply_markup=get_model_keyboard(selected_model)
        )
        return

    model = context.user_data.get('model')
    if not model:
        await update.message.reply_text("Пожалуйста, сначала выберите модель.", reply_markup=get_model_keyboard())
        return
    
    response_text, response_time, input_tokens, output_tokens, cost = await generate_response(message, model, query_engine)

    logger.info("Sent response", 
                user_id=user.id, 
                username=user.username, 
                response=response_text, 
                response_time=response_time, 
                model_used=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=cost)
    
    log_entry = create_log_entry(user, model, message, response_text, response_time, input_tokens, output_tokens, cost)
    
    response_info = format_response_info(model, response_time, input_tokens, output_tokens, cost)
    response_text += response_info
    
    await send_response_in_chunks(update, response_text, user.id)
    
    context.user_data['last_log_entry'] = log_entry

async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id, rating = query.data.split('_')[1:]
    rating = 1 if rating == "up" else -1

    log_entry = context.user_data.get('last_log_entry', {})
    log_entry['rating'] = rating

    with open('chat_log.json', 'a', encoding='utf-8') as f:
        json.dump(log_entry, f, ensure_ascii=False)
        f.write('\n')

    logger.info("Received rating", user_id=user_id, rating=rating)
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text(f"Спасибо за вашу оценку! ({rating})")

def main():
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN_GALAKTIKA")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_rating))

    logger.info("Starting bot")
    application.run_polling()

if __name__ == '__main__':
    logger = setup_logging()
    query_engine = initialize_index()
    main()