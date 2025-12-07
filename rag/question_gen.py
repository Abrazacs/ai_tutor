from .vector_store import search
from .embeddings import embed_query
from config import TOP_K


def generate_questions(topic: str, n: int = 5) -> list[str]:
    # В идеале тут тоже вызывать LLM, но сделаем простую заготовку.
    emb = embed_query(topic)
    ctx = search(emb, top_k=TOP_K)
    base = topic if topic.strip() else "тема из материалов"
    questions = []
    for i in range(1, n + 1):
        questions.append(f"Вопрос {i}: объясни ключевые идеи по теме «{base}»?")
    # Можно добавить вопросы по файлам
    for c in ctx:
        questions.append(
            f"Что описывается в материале {c['meta'].get('source_file')} (chunk {c['meta'].get('chunk_id')})?"
        )
    return questions[:n]
