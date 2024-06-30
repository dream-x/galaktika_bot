docker run -d \
  --network host \
  -e TELEGRAM_BOT_TOKEN_GALAKTIKA=your_token \
  -e OPENAI_API_KEY=your_openai_key \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  -v /path/to/your/docs:/app/docs \
  -v /path/to/your/stored_index:/app/stored_index \
  telegram-bot

poetry export -f requirements.txt --output requirements.txt

