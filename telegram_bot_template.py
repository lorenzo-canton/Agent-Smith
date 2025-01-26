import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, CallbackContext
from telegram.ext.filters import TEXT
from agent import Agent

config = json.loads(open("config.json").read())

class TelegramBot():
    def __init__(self) -> None:
        self.app = ApplicationBuilder().token(config["telegram_token"]).build()
        self.agent = Agent()  # Crea un'istanza dell'agente
        self.conversations = {}  # Dizionario per memorizzare lo stato delle conversazioni
        self.add_commands()
        self.add_message_handler()


    async def command_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.message.chat_id
        # Inizializza una nuova conversazione
        self.conversations[chat_id] = [self.agent.system_prompt]
        await update.message.reply_text(f'Ciao {update.effective_user.first_name}! Sono il tuo assistente AI. Come posso aiutarti?')


    def add_commands(self):
        self.app.add_handler(CommandHandler("start", self.command_start))


    async def handle_message(self, update: Update, context: CallbackContext):
        if update.message.text.startswith("/"):
            return
            
        chat_id = update.message.chat_id
        
        # Se è una nuova conversazione, inizializza
        if chat_id not in self.conversations:
            self.conversations[chat_id] = [self.agent.system_prompt]
            
        # Aggiungi il messaggio dell'utente
        self.conversations[chat_id].append({
            "role": "user",
            "content": update.message.text
        })
        
        # Mostra lo stato di elaborazione
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        # Processa il messaggio con l'agente
        try:
            response = self.agent.process_conversation(self.conversations[chat_id])
            await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text(f"Si è verificato un errore: {str(e)}")
            # Resetta la conversazione in caso di errore
            self.conversations[chat_id] = [self.agent.system_prompt]


    def add_message_handler(self):
        self.app.add_handler(MessageHandler(filters=TEXT, callback=self.handle_message))


    def run(self):
        print("Bot running")
        self.app.run_polling()



if __name__ == '__main__':
    telegram_bot = TelegramBot()
    telegram_bot.run()

    
