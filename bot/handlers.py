from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from rag.qa_pipeline import answer_question
from rag.question_gen import generate_questions
from rag.recommender import recommend_resources
from .state import get_state

# Добавили красивый вывод в телеграмме
def to_telegram_markdown(text: str) -> str:
    lines = text.splitlines()
    out_lines = []

    for line in lines:
        stripped = line.lstrip()
        # "#### ..." → жирный
        if stripped.startswith("#### "):
            title = stripped[5:].strip()
            out_lines.append(f"*{title}*")
        # "### ..." → жирный
        elif stripped.startswith("### "):
            title = stripped[4:].strip()
            out_lines.append(f"*{title}*")
        # "## ..." → жирный
        elif stripped.startswith("## "):
            title = stripped[3:].strip()
            out_lines.append(f"*{title}*")
        else:
            out_lines.append(line)

    return "\n".join(out_lines)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я AI-репетитор по техническим дисциплинам.\n\n"
        "Команды:\n"
        "/ask — задать вопрос по учебным материалам\n"
        "/quiz — получить вопросы для самопроверки\n"
        "/resources — рекомендации, что почитать\n"
        "Просто напиши вопрос — и я постараюсь помочь."
    )
    await update.message.reply_text(text)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Напиши свой вопрос, я попробую ответить по материалам.")
    state = get_state(update.effective_user.id)
    state.mode = "chat"


async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = get_state(update.effective_user.id)
    topic = " ".join(context.args) if context.args else state.last_topic or "основы курса"
    qs = generate_questions(topic, n=5)
    state.mode = "quiz"
    state.current_questions = qs
    text = "Вопросы для самопроверки:\n\n" + "\n".join(f"- {q}" for q in qs)
    await update.message.reply_text(text)


async def cmd_resources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = get_state(update.effective_user.id)
    topic = " ".join(context.args) if context.args else state.last_topic or "основы курса"
    recs = recommend_resources(topic, n=3)
    if not recs:
        await update.message.reply_text("Пока не могу ничего посоветовать по этой теме.")
        return
    text = "Рекомендованные материалы:\n" + "\n".join(f"- {r}" for r in recs)
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text
    state = get_state(user_id)
    state.last_topic = text

    answer, sources = answer_question(text)
# добавили список источников (также красивый вывод)
    if sources:
        src_text = "\n".join(
            f"- `{s['meta'].get('source_file', 'unknown')}` (chunk {s['meta'].get('chunk_id')})"
            for s in sources
        )
        answer += "\n\nИсточники:\n" + src_text

    formatted = to_telegram_markdown(answer)

    await update.message.reply_text(
        formatted,
        parse_mode=ParseMode.MARKDOWN,
    )
