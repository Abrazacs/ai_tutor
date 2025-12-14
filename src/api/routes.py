from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
import tempfile
import os
from pathlib import Path

from src.models.document import QueryRequest, QueryResponse
from src.pipeline.document_loader import DocumentLoader
from src.pipeline.chunker import DocumentChunker
from src.pipeline.embedder import Embedder
from src.database.vector_store import VectorStore
from src.services.retrieval_service import RetrievalService
from src.services.llm_service import LLMService

app = FastAPI(title="AI Tutor API", version="1.0.0")

document_loader = DocumentLoader()
chunker = DocumentChunker()

embedder = Embedder()
vector_store = VectorStore()
retrieval_service = RetrievalService(vector_store, embedder)
llm_service = LLMService()


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {"message": "AI Tutor API", "status": "running"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Загрузка и обработка документа

    Args:
        file: Загружаемый файл

    Returns:
        Информация о загруженном документе
    """
    try:
        # Проверяем расширение файла
        file_extension = Path(file.filename).suffix.lower()
        supported_formats = ['.pdf', '.docx', '.txt', '.md']

        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла: {file_extension}. "
                       f"Поддерживаемые форматы: {', '.join(supported_formats)}"
            )

        # Сохраняем файл временно
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Загружаем документ
            documents = document_loader.load_file(tmp_file_path)

            if not documents or not documents[0].content.strip():
                raise ValueError("Документ пуст или не содержит текста")

            # Разбиваем на чанки
            chunks = chunker.chunk_documents(documents)

            if not chunks:
                raise ValueError("Не удалось создать фрагменты документа")

            # Создаем эмбеддинги
            chunks_with_embeddings = embedder.embed_chunks(chunks)

            # Сохраняем в векторную БД
            vector_store.add_chunks(chunks_with_embeddings)

            return {
                "status": "success",
                "filename": file.filename,
                "documents_count": len(documents),
                "chunks_count": len(chunks),
                "message": f"Документ '{file.filename}' успешно обработан и добавлен в базу знаний"
            }

        finally:
            # Удаляем временный файл
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Ошибка при обработке документа: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке документа: {str(e)}"
        )


@app.post("/documents/upload-directory")
async def upload_directory(directory_path: str):
    """
    Загрузка всех документов из директории

    Args:
        directory_path: Путь к директории

    Returns:
        Информация о загруженных документах
    """
    try:
        # Загружаем документы
        documents = document_loader.load_directory(directory_path)

        if not documents:
            return {"status": "warning", "message": "Документы не найдены"}

        # Разбиваем на чанки
        chunks = chunker.chunk_documents(documents)

        # Создаем эмбеддинги
        chunks_with_embeddings = embedder.embed_chunks(chunks)

        # Сохраняем в векторную БД
        vector_store.add_chunks(chunks_with_embeddings)

        return {
            "status": "success",
            "documents_count": len(documents),
            "chunks_count": len(chunks),
            "message": f"Загружено {len(documents)} документов из {directory_path}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке директории: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Обработка запроса пользователя

    Args:
        request: Запрос с вопросом пользователя

    Returns:
        Ответ с использованием RAG
    """
    try:
        # Получаем релевантный контекст
        sources, formatted_context = retrieval_service.retrieve_and_format(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        if not sources:
            return QueryResponse(
                answer="К сожалению, я не нашел информации в базе знаний, которая могла бы ответить на ваш вопрос. Попробуйте переформулировать запрос или загрузите дополнительные материалы.",
                sources=[],
                confidence=0.0
            )

        # Генерируем ответ с помощью LLM
        response = llm_service.generate_with_sources(
            query=request.query,
            context=formatted_context,
            sources=sources
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Получить статистику по базе знаний"""
    try:
        stats = vector_store.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении статистики: {str(e)}")


@app.delete("/documents")
async def delete_all_documents():
    """Удалить все документы из базы знаний"""
    try:
        vector_store.delete_all()
        return {"status": "success", "message": "Все документы удалены"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении документов: {str(e)}")


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "service": "AI Tutor"}