import logging
import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode, ChatAction
from telegram.helpers import escape_markdown

from src.config import settings
from src.models.document import QueryRequest

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def query_api(query: str, top_k: int = 5, filters: dict | None = None):
    url = settings.server_url + "/query"

    payload = {
        "query": query,
        "top_k": top_k,
        "filters": filters or {}
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


class SimpleTelegramBot:
    """Простой Telegram бот для ответов на вопросы"""

    def __init__(self):
        """Инициализация бота"""
        self.token = settings.telegram_bot_token
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env")

        logger.info("Telegram Bot инициализирован с общими сервисами")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        welcome_message = f"""
Привет, {user.first_name}!

Я — AI-репетитор. Задавай мне любые вопросы по учебным материалам!

Просто напиши свой вопрос, и я отвечу на основе загруженной базы знаний.

Команды:
/help - Справка
"""
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
*Как пользоваться ботом*

Просто напиши свой вопрос обычным сообщением, и я дам ответ на основе учебных материалов.

*Примеры вопросов:*
• Что такое машинное обучение?
• Объясни принцип работы нейронных сетей
• Как создать функцию в Python?

Удачи в обучении!
"""
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик текстовых сообщений с вопросами
        Переиспользует логику из API routes.py
        """
        user = update.effective_user
        query_text = update.message.text

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        logger.info(f"Вопрос от {user.username} ({user.id}): {query_text}")

        try:
            api_response = await query_api(
                query=query_text,
                top_k=settings.top_k,
                filters={}
            )

            answer = api_response.get("answer")
            sources = api_response.get("sources", [])
            confidence = api_response.get("confidence")

            answer_text = f"*Ответ:*\n\n{answer}"

            max_length = getattr(settings, 'max_message_length', 4000)
            logger.info("Ответ: " + answer_text)
            if len(answer_text) > max_length:
                chunks = self._split_message(answer_text, max_length)
                for chunk in chunks:
                    await update.message.reply_text(
                        chunk,
                        parse_mode=ParseMode.MARKDOWN
                    )
            else:
                await update.message.reply_text(
                    answer_text,
                    parse_mode=ParseMode.MARKDOWN
                )

            if sources:
                sources_text = "Источники:\n"
                for i, source in enumerate(sources[:3], 1):
                    escaped_file = escape_markdown(source['file'])
                    sources_text += (
                        f"{i}. {escaped_file} "
                        f"(релевантность: {source['similarity']:.0%})\n"
                    )

                sources_text += f"Уверенность: {confidence:.0%}"
                logger.info(f"Текст источников {sources_text}")

                await update.message.reply_text(
                    sources_text,
                    parse_mode=ParseMode.MARKDOWN
                )

            logger.info(f"Ответ отправлен пользователю {user.id}")

        except Exception as e:
            logger.error(f"Ошибка обработки вопроса: {e}", exc_info=True)
            await update.message.reply_text(
                "Извините, произошла ошибка при обработке вашего вопроса.\n"
                "Попробуйте еще раз или переформулируйте вопрос."
            )

    def _split_message(self, text: str, max_length: int) -> list:
        """Разбивает длинное сообщение на части"""
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            split_pos = text.rfind('\n', 0, max_length)
            if split_pos == -1:
                split_pos = max_length

            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip()

        return chunks

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок"""
        logger.error(f"Ошибка: {context.error}", exc_info=context.error)

        if update and update.effective_message:
            await update.effective_message.reply_text(
                "Произошла непредвиденная ошибка. Попробуйте позже."
            )

    def run(self):
        """Запуск бота"""
        logger.info("Запуск Telegram бота...")

        # Создаем приложение
        application = Application.builder().token(self.token).build()

        # Регистрируем обработчики команд
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))

        # Регистрируем обработчик текстовых сообщений (вопросов)
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        # Регистрируем обработчик ошибок
        application.add_error_handler(self.error_handler)

        # Запускаем бота
        logger.info("Бот запущен и готов к работе!")
        logger.info("Отправьте боту любой вопрос для получения ответа")
        application.run_polling(allowed_updates=Update.ALL_TYPES)