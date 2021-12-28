import os
from functools import wraps

import torch
from telegram import Update, ChatAction
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
from telegram.ext.dispatcher import run_async
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer
import torch

import csv


# Place the token from bot Father in the token.txt file
token = open('token.txt', 'r').read()

TOKEN = token.replace('\n', '')

# tg_admin_id = os.environ['TELEGRAM_ADMIN_ID']  # Telegram admin id, ask @userinfobot (optional)

# Initialise the bot
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher


# Initialise Dialogpt related entities
print('Loading DialoGPT model...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')

# the directory to the trained model downloaded from Google Colab
model = AutoModelWithLMHead.from_pretrained('output/content/output-small')

def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(update, context, *args, **kwargs)

        return command_func

    return decorator


send_typing_action = send_action(ChatAction.TYPING)


@run_async
def start(update: Update, context: CallbackContext):
    context.chat_data.clear()
    context.chat_data['message_count'] = 0
    update.message.reply_text("Hello! I'm Siri 2.0 and I am a GPT2 NLP model (v0.01). Type /restart to restart the conversation with me")


dp.add_handler(CommandHandler(['start', 'restart', 'bye'], start))


def isEnglish(s):
    return s.isascii()


@run_async
@send_typing_action
def dialogpt(update: Update, context: CallbackContext):
    if context.chat_data.get('message_count') is None:
        start(update, context)

    if isEnglish(update.message.text) == False:
        update.message.reply_text("The bot doesn't like non ASCII characters")
        main()


    from datetime import datetime
    
    chat_id = update.message.chat_id
    first_name = update.message.chat.first_name
    last_name = update.message.chat.last_name
    username = update.message.chat.username
    fullname = "{} {}".format(first_name, last_name)

    text =  update.message.text

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")



    # encode the new user input, add the eos_token and return a tensor in PyTorch
    new_user_input_ids = tokenizer.encode(update.message.text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([context.chat_data['chat_history_ids'], new_user_input_ids], dim=-1) \
        if context.chat_data['message_count'] > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    context.chat_data['chat_history_ids'] = model.generate(
        bot_input_ids, max_length=200,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature=0.8
    )


    response = tokenizer.decode(context.chat_data['chat_history_ids'][:, bot_input_ids.shape[-1]:][0],
                                               skip_special_tokens=True)

    # decode and reply the user
    update.message.reply_text(response)
    context.chat_data['message_count'] += 1

    if context.chat_data.get('message_count') > 3:
        update.message.reply_text('End of conversation, reply /restart to start new conversation')
        context.chat_data.clear()

    ## Log bot response

    logfile = [dt_string, chat_id, fullname, username, text, response]

    with open('telegrambotlog.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(logfile)


def main():
    """Run bot."""

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", start))
    
    dp.add_handler(MessageHandler(Filters.text, dialogpt))

    print("Bot running...")

    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()



if __name__ == '__main__':
    main()


