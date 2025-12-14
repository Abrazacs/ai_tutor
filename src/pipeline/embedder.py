from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from src.models.document import DocumentChunk
from src.config import settings

# Опционально импортируем GigaChat только если используем
try:
    from gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False


class Embedder:
    """Создание векторных представлений текста"""

    def __init__(self, model_name: str = None, use_gigachat: bool = None):
        """
        Инициализация эмбеддера

        Args:
            model_name: Название модели для эмбеддингов
            use_gigachat: Использовать ли embeddings от GigaChat
        """
        self.use_gigachat = use_gigachat if use_gigachat is not None else settings.use_gigachat_embeddings

        if self.use_gigachat:
            if not GIGACHAT_AVAILABLE:
                raise ImportError("GigaChat не установлен. Установите: pip install gigachat")

            print("Инициализация GigaChat Embeddings")
            self.gigachat_client = GigaChat(
                credentials=settings.gigachat_credentials,
                scope=settings.gigachat_scope,
                verify_ssl_certs=settings.gigachat_verify_ssl
            )
            self.dimension = 1024  # Размерность эмбеддингов GigaChat
            self.model_name = "GigaChat Embeddings"
        else:
            self.model_name = model_name or settings.embedding_model
            print(f"Загрузка модели эмбеддингов: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """
        Создает эмбеддинг для текста

        Args:
            text: Текст для эмбеддинга

        Returns:
            Вектор эмбеддинга
        """
        if self.use_gigachat:
            response = self.gigachat_client.embeddings(text)
            # GigaChat возвращает объект с полем data, содержащим список эмбеддингов
            return response.data[0].embedding
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов

        Args:
            texts: Список текстов

        Returns:
            Список векторов эмбеддингов
        """
        if self.use_gigachat:
            embeddings = []
            # GigaChat обрабатываем по одному или батчами
            for i in range(0, len(texts), 10):  # Батчи по 10
                batch = texts[i:i+10]
                for text in batch:
                    response = self.gigachat_client.embeddings(text)
                    embeddings.append(response.data[0].embedding)
                print(f"Обработано {min(i+10, len(texts))}/{len(texts)} текстов")
            return embeddings
        else:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return embeddings.tolist()

    def embed_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Добавляет эмбеддинг к чанку

        Args:
            chunk: Чанк документа

        Returns:
            Чанк с эмбеддингом
        """
        chunk.embedding = self.embed_text(chunk.content)
        return chunk

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Добавляет эмбеддинги к списку чанков

        Args:
            chunks: Список чанков

        Returns:
            Список чанков с эмбеддингами
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Вычисляет косинусное сходство между двумя эмбеддингами

        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг

        Returns:
            Значение сходства от -1 до 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))