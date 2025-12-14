from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    gigachat_credentials: str = "MDE5YjE2ODQtMTUwNi03NzBlLWFjZGEtM2QzZWY4OGUxOTQ3OjVkNTViYTBjLWZkMGEtNDk1Zi1hMzNmLTk2N2FhZWFiN2RkMg=="
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_verify_ssl: bool = False

    # Embedding settings
    use_gigachat_embeddings: bool = False
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384  # 384 для sentence-transformers, 1024 для GigaChat

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Vector DB settings
    vector_db_path: str = "./chroma_db"
    collection_name: str = "documents"

    # LLM settings
    llm_model: str = "GigaChat"
    llm_temperature: float = 0.5
    max_tokens: int = 1000

    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.5

    telegram_bot_token: str = "8438406219:AAHkc1jw9eqgeluR-UhV8izWPTEQZlX8jY0"
    max_message_length: int = 4000
    server_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()