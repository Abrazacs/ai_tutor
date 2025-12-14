# rag/qa_pipeline.py
from config import TOP_K
from .embeddings import embed_query
from .vector_store import search
from .safety import is_text_safe, sanitize_user_text
from .local_llm import call_llm   # вот главное изменение


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Источник: {c['meta'].get('source_file', 'unknown')} #{c['meta'].get('chunk_id')}] {c['text']}"
        for c in context_chunks
    )
    
    prompt = f"""
Ты выступаешь как AI-репетитор по техническим дисциплинам.
Отвечай понятно и структурированно, с примерами.
Список источников будет добавлен автоматически, не добавляй его

Вопрос студента:
{question}

Релевантный контекст:
{context_text}

Формат ответа:
1) Пошаговое объяснение.
2) Краткое резюме.

"""
    return prompt.strip()


def answer_question(user_question: str) -> tuple[str, list[dict]]:
    if not is_text_safe(user_question):
        return "Извини, я не могу обработать этот запрос из-за содержимого.", []

    q = sanitize_user_text(user_question)
    q_emb = embed_query(q)
    ctx = search(q_emb, top_k=TOP_K)
    if not ctx:
        return "Я не нашёл подходящую информацию в материале. Попробуй переформулировать вопрос.", []

    prompt = build_prompt(q, ctx)
    answer = call_llm(prompt)
    return answer, ctx
