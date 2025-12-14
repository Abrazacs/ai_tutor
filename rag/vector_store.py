import chromadb
from chromadb.config import Settings
from config import INDEX_DIR

_client = chromadb.PersistentClient(path=INDEX_DIR, settings=Settings(allow_reset=True))
_COLLECTION_NAME = "tutor_docs"


def get_collection():
    return _client.get_or_create_collection(_COLLECTION_NAME)

#Этот вариант add_documents делает одну  вещь: сохраняет векторные представления документов, 
#которые заранее посчитали, вместо того чтобы позволять векторному хранилищу считать самостоятельно

def add_documents(
    doc_ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
):
    col = get_collection()
    col.add(
        ids=doc_ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def search(query_embedding: list[float], top_k: int = 5):
    col = get_collection()
    res = col.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    return [{"id": i, "text": d, "meta": m} for i, d, m in zip(ids, docs, metas)]
