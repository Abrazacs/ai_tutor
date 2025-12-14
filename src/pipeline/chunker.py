from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.models.document import Document, DocumentChunk
from src.config import settings
import uuid


class DocumentChunker:
    """Разбиение документов на чанки"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Инициализация чанкера

        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Размер перекрытия между чанками
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Разбивает документ на чанки

        Args:
            document: Документ для разбиения

        Returns:
            Список чанков
        """
        text_chunks = self.splitter.split_text(document.content)

        chunks = []
        for i, text in enumerate(text_chunks):
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                content=text,
                metadata={
                    **document.metadata,
                    'chunk_size': len(text)
                },
                document_id=document.id,
                chunk_index=i
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Разбивает список документов на чанки

        Args:
            documents: Список документов

        Returns:
            Список всех чанков
        """
        all_chunks = []
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)

        return all_chunks

    def optimize_chunk_size(self, document: Document, target_chunks: int = 10) -> int:
        """
        Определяет оптимальный размер чанка для документа

        Args:
            document: Документ
            target_chunks: Желаемое количество чанков

        Returns:
            Рекомендуемый размер чанка
        """
        doc_length = len(document.content)
        optimal_size = doc_length // target_chunks

        # Округляем до ближайшей сотни
        return max(100, (optimal_size // 100) * 100)