import os
import telebot
import torch

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

bot = telebot.TeleBot(token)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, 'Привет! Данный бот может ответить тебе практически на все вопросы. К сожалению, пока что он неидеален, и поэтому может вести диалог только на английском языке.')


@bot.message_handler(func=lambda message: True)
def handle_text(message):
    peft_model_name = "/content/drive/MyDrive/lora-flan-t5-base-chat"
    config = PeftConfig.from_pretrained(peft_model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_name, device_map={"":0}).to(device)
    model.eval()

    sample = message.text

    input_ids = tokenizer(sample, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
    outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_length=256)

    response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

    bot.reply_to(message, response)

bot.infinity_polling()
