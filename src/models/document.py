from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class Document(BaseModel):
    """Модель документа"""
    id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Модель чанка документа"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: Optional[str] = None
    chunk_index: int = 0


class QueryRequest(BaseModel):
    """Запрос пользователя"""
    query: str
    top_k: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Ответ на запрос"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 0.0