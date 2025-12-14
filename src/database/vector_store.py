from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.models.document import DocumentChunk
from src.config import settings


class VectorStore:
    """Хранилище векторных представлений документов"""

    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """
        Инициализация векторного хранилища

        Args:
            persist_directory: Путь для сохранения БД
            collection_name: Название коллекции
        """
        self.persist_directory = persist_directory or settings.vector_db_path
        self.collection_name = collection_name or settings.collection_name

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=True,
                allow_reset=True
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Добавляет чанки в векторную БД

        Args:
            chunks: Список чанков с эмбеддингами
        """
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Добавляем в батчах для лучшей производительности
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))

            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

        print(f"Добавлено {len(chunks)} чанков в векторную БД")

    def search(
            self,
            query_embedding: List[float],
            top_k: int = None,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск наиболее похожих векторов

        Args:
            query_embedding: Вектор запроса
            top_k: Количество результатов
            filters: Фильтры для метаданных

        Returns:
            Список найденных документов с метаданными
        """
        top_k = top_k or settings.top_k

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        search_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'similarity': 1 - results['distances'][0][i] if 'distances' in results else None
            }
            search_results.append(result)

        return search_results

    def delete_by_source(self, source: str) -> None:
        """
        Удаляет документы по источнику

        Args:
            source: Источник документа
        """
        self.collection.delete(where={"source": source})
        print(f"Удалены документы из источника: {source}")

    def delete_all(self) -> None:
        """Удаляет все документы из коллекции"""
        self.client.delete_collection(self.collection_name)
        print("Все документы удалены из векторной БД")

        # Пересоздаем коллекцию и обновляем ссылку
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Коллекция векторной БД пересоздана")

    def get_stats(self) -> Dict[str, Any]:
        """
        Получает статистику по коллекции

        Returns:
            Словарь со статистикой
        """
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_directory': self.persist_directory
        }