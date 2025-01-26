import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, CallbackContext
from telegram.ext.filters import TEXT

config = json.loads(open("env.json").read())

class TelegramBot():
    def __init__(self) -> None:
        self.app = ApplicationBuilder().token(config["telegram_token"]).build()
        self.add_commands()
        self.add_message_handler()


    async def command_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        print(context._chat_id)
        await update.message.reply_text(f'Hello {update.effective_user.first_name}')


    def add_commands(self):
        self.app.add_handler(CommandHandler("start", self.command_start))


    async def handle_message(self, update: Update, context: CallbackContext):
        if update.message.text.startswith("/"): return
        await update.message.reply_text(update.message.text)


    def add_message_handler(self):
        self.app.add_handler(MessageHandler(filters=TEXT, callback=self.handle_message))


    def run(self):
        print("Bot running")
        self.app.run_polling()



if __name__ == '__main__':
    telegram_bot = TelegramBot()
    telegram_bot.run()

    