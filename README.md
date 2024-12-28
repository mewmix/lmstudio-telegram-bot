  # lmstudio-telegram-bot
  
  this is a telegram bot that integrates with lmstudio's openai compatible api server providing chat, completion, and embeddings. it supports multiple conversation threads, custom system prompts, and manual or automatic summarization that gets stored to a local sqlite database.
  
  ## features
  
  - uses the open-ai compatible endpoints for now
  - conversation threads for separate contexts
  - on-demand and automatic summarization
  - some markdown-formatted responses
  - user-controllable parameters (temperature, max_tokens, etc.)
  
  ## installation
  
  1. clone this repo (or copy the files).
  2. make a copy of the env example file and save as .env ~ fill in your bot token in the file. 
  3. install dependencies via:
     ```bash
     pip install -r requirements.txt
     ```
     or, if you have `uv` installed:
     ```bash
     uv pip install -r requirements.txt
     ```
  4. run the bot:
     ```bash
     python bot.py
     ```
  
  ## usage
  
  once the bot is running, talk to it on telegram. you can create new threads, switch threads, set prompts or models, summarize, and more. check `/help` or the command list for details.
  
  ## commands
  send these to the botfather to update your command list. 
  
  - **set**: update conversation parameter (like temperature, max_tokens, etc.)
  - **show_params**: show current conversation parameters
  - **new_thread**: create a new conversation thread
  - **list_threads**: display your existing conversation threads
  - **switch_thread**: switch to a specified conversation thread by id
  - **summarize_thread**: summarize a conversation (by id or active)
  - **show_summaries**: show stored summaries for the active conversation
  - **set_model**: set the language model for the active conversation
  - **set_system_prompt**: set or update the system prompt
  - **show_system_prompt**: show the current system prompt
  - **clear_context**: clear messages in the active conversation
  - **list_models**: list all lm studio models
  - **completion**: use the legacy completion endpoint
  - **embedding**: get an embedding for input text
  

