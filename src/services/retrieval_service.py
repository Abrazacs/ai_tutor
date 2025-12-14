from typing import List, Dict, Any, Optional
from src.database.vector_store import VectorStore
from src.pipeline.embedder import Embedder
from src.config import settings


class RetrievalService:
    """Сервис для поиска релевантного контекста"""

    def __init__(self, vector_store: VectorStore = None, embedder: Embedder = None):
        """
        Инициализация сервиса поиска

        Args:
            vector_store: Векторное хранилище
            embedder: Эмбеддер для векторизации запросов
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()

    def retrieve_context(
            self,
            query: str,
            top_k: int = None,
            filters: Optional[Dict[str, Any]] = None,
            similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Получает релевантный контекст для запроса

        Args:
            query: Запрос пользователя
            top_k: Количество результатов
            filters: Фильтры для метаданных
            similarity_threshold: Порог сходства

        Returns:
            Список релевантных документов
        """
        top_k = top_k or settings.top_k
        similarity_threshold = similarity_threshold or settings.similarity_threshold

        # Векторизация запроса
        query_embedding = self.embedder.embed_text(query)

        # Поиск в векторной БД
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        # Фильтрация по порогу сходства
        filtered_results = [
            result for result in results
            if result['similarity'] and result['similarity'] >= similarity_threshold
        ]

        return filtered_results

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Форматирует результаты поиска в контекст для LLM

        Args:
            results: Результаты поиска

        Returns:
            Отформатированный контекст
        """
        if not results:
            return "Релевантная информация не найдена."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            content = result['content']
            similarity = result.get('similarity', 0)

            context_part = f"""
Источник {i} (релевантность: {similarity:.2%}):
Файл: {source}
Содержание: {content}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def retrieve_and_format(
            self,
            query: str,
            top_k: int = None,
            filters: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Получает и форматирует контекст для запроса

        Args:
            query: Запрос пользователя
            top_k: Количество результатов
            filters: Фильтры для метаданных

        Returns:
            Кортеж (результаты поиска, отформатированный контекст)
        """
        results = self.retrieve_context(query, top_k, filters)
        formatted_context = self.format_context(results)

        return results, formatted_context

    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Дополнительное ранжирование результатов

        Args:
            query: Запрос пользователя
            results: Результаты поиска

        Returns:
            Переранжированные результаты
        """
        # Простая реранжировка по длине документа (можно использовать более сложные методы)
        query_length = len(query.split())

        for result in results:
            content_length = len(result['content'].split())
            length_score = min(content_length / (query_length * 10), 1.0)

            # Комбинируем similarity и length_score
            result['final_score'] = (
                    result.get('similarity', 0) * 0.7 +
                    length_score * 0.3
            )

        return sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)