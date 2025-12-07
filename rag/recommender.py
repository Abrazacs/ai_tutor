from .embeddings import embed_query
from .vector_store import search
from config import TOP_K


def recommend_resources(topic: str, n: int = 3) -> list[str]:
    emb = embed_query(topic)
    ctx = search(emb, top_k=TOP_K)
    recs = []
    for c in ctx[:n]:
        src = c["meta"].get("source_file", "unknown")
        chunk_id = c["meta"].get("chunk_id", 0)
        recs.append(f"{src}, фрагмент #{chunk_id}")
    return recs
