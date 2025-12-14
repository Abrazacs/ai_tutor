import os
from pathlib import Path
from config import DATA_RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .embeddings import embed_texts
from .vector_store import add_documents


def _iter_files():
    for root, _, files in os.walk(DATA_RAW_DIR):
        for name in files:
            if name.endswith((".txt", ".md")):
                yield Path(root) / name


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, chunk_size: int, overlap: int):
    text = text.replace("\n", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_index():
    doc_ids, texts, metas = [], [], []
    for f in _iter_files():
        full_text = _read_text(f)
        chunks = _chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, ch in enumerate(chunks):
            doc_ids.append(f"{f.name}_{idx}")
            texts.append(ch)
            metas.append({"source_file": f.name, "chunk_id": idx})

    if not texts:
        print("No documents found in data/raw")
        return

    emb = embed_texts(texts)
    add_documents(doc_ids, texts, metas, emb) # передали эмбеддинги в базу
    print(f"Indexed {len(texts)} chunks.")


if __name__ == "__main__":
    build_index()
