import os
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from src.models.document import Document
import uuid


class DocumentLoader:
    """Загрузчик документов различных форматов"""

    LOADERS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
    }

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Загружает документ из файла

        Args:
            file_path: Путь к файлу

        Returns:
            Список документов
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        extension = file_path.suffix.lower()

        if extension not in self.LOADERS:
            raise ValueError(f"Неподдерживаемый формат файла: {extension}")

        loader_class = self.LOADERS[extension]
        loader = loader_class(str(file_path))

        langchain_docs = loader.load()

        documents = []
        for doc in langchain_docs:
            documents.append(Document(
                id=str(uuid.uuid4()),
                content=doc.page_content,
                metadata={
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': extension,
                    **doc.metadata
                }
            ))

        return documents

    def load_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """
        Загружает все поддерживаемые документы из директории

        Args:
            directory_path: Путь к директории

        Returns:
            Список документов
        """
        directory_path = Path(directory_path)
        all_documents = []

        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.LOADERS:
                try:
                    documents = self.load_file(file_path)
                    all_documents.extend(documents)
                    print(f"Загружено: {file_path}")
                except Exception as e:
                    print(f"Ошибка при загрузке {file_path}: {e}")

        return all_documents

    def load_text(self, text: str, metadata: dict = None) -> Document:
        """
        Создает документ из текста

        Args:
            text: Текстовое содержимое
            metadata: Метаданные

        Returns:
            Документ
        """
        return Document(
            id=str(uuid.uuid4()),
            content=text,
            metadata=metadata or {}
        )