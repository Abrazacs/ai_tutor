#  AI Tutor

Интеллектуальный AI-репетитор по техническим дисциплинам с использованием RAG (Retrieval-Augmented Generation), ChromaDB и GigaChat.

## Описание

AI Tutor — это система для создания персонализированного обучающего опыта на основе ваших учебных материалов. Проект состоит из двух компонентов:

1. **REST API** (FastAPI) — для загрузки документов и обработки запросов
2. **Telegram Bot** — удобный интерфейс для студентов

Система использует векторный поиск для извлечения релевантного контекста из загруженных материалов и GigaChat для генерации понятных ответов.

##  Основные возможности

-  **Загрузка учебных материалов** в форматах PDF, DOCX, TXT, MD
-  **Семантический поиск** по базе знаний с помощью ChromaDB
-  **Генерация ответов** с использованием GigaChat LLM
-  **Telegram-интерфейс** для удобного взаимодействия
-  **Указание источников** и уровня уверенности в ответах
-  **RAG pipeline** для точных и контекстуальных ответов
-  **REST API** для интеграции с другими системами

## Технологический стек

- **Backend Framework**: FastAPI
- **LLM**: GigaChat (Сбер)
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers / GigaChat Embeddings
- **Bot Framework**: python-telegram-bot
- **Document Processing**: LangChain, PyPDF2, python-docx
- **Language**: Python 3.11

## Установка

### Предварительные требования

- Python 3.11 или выше
- pip (менеджер пакетов Python)
- Токен GigaChat API (получите на [developers.sber.ru](https://developers.sber.ru/))
- Telegram Bot Token (получите у [@BotFather](https://t.me/botfather))

### Шаги установки

1. **Клонируйте репозиторий:**

```bash
git clone https://github.com/Abrazacs/ai_tutor.git
cd ai_tutor
```

2. **Создайте виртуальное окружение:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Установите зависимости:**

```bash
pip install -r requirements.txt
```

## Запуск
Откройте два терминала:

**Терминал 1 (API):**
```bash
python main.py
```

**Терминал 2 (Bot):**
```bash
python run_bot.py
```

API будет доступен по адресу: `http://localhost:8000`

Документация API: `http://localhost:8000/docs`

## Использование

### 1. Загрузка учебных материалов

#### Через API

```bash
# Загрузка одного файла
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"

# Загрузка всех файлов из директории
curl -X POST "http://localhost:8000/documents/upload-directory?directory_path=/path/to/documents"
```
<img width="1317" height="741" alt="image" src="https://github.com/user-attachments/assets/ce4fcb18-259a-46bd-aa6d-6f24a4392907" />


### 2. Задавание вопросов

#### Через API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Что такое машинное обучение?",
    "top_k": 5
  }'
```
Запрос:
<img width="1350" height="422" alt="image" src="https://github.com/user-attachments/assets/dee6f07b-9f01-4aad-8447-944b1b286142" />

Ответ:
<img width="1304" height="637" alt="image" src="https://github.com/user-attachments/assets/c8771e73-f9ce-460b-aead-7e7598737ed4" />


#### Через Telegram

1. Найдите вашего бота в Telegram
2. Отправьте команду `/start`
3. Задайте любой вопрос обычным сообщением

## Структура проекта

```
ai_tutor/
│
├── src/
│   ├── api/
│   │   └── routes.py              # FastAPI маршруты и endpoints
│   │
│   ├── bot/
│   │   └── simple_bot.py          # Telegram bot логика
│   │
│   ├── database/
│   │   └── vector_store.py        # ChromaDB интерфейс
│   │
│   ├── models/
│   │   └── document.py            # Pydantic модели данных
│   │
│   ├── pipeline/
│   │   ├── chunker.py             # Разбиение документов на chunks
│   │   ├── document_loader.py     # Загрузка документов
│   │   └── embedder.py            # Создание embeddings
│   │
│   └── services/
│       ├── llm_service.py         # GigaChat LLM сервис
│       └── retrieval_service.py   # RAG поиск и форматирование
│
├── chroma_db/                     # Векторная база данных (создается автоматически)
├── config.py                      # Конфигурация приложения
├── main.py                        # Запуск API сервера
├── run_bot.py                     # Запуск Telegram бота
├── requirements.txt               # Python зависимости
└── README.md                      # Документация
```

## API Endpoints

### Документы

- `POST /documents/upload` - Загрузка одного файла
- `POST /documents/upload-directory` - Загрузка директории с файлами
- `DELETE /documents` - Удаление всех документов

### Запросы

- `POST /query` - Задать вопрос и получить ответ
- `GET /stats` - Статистика по базе знаний
- `GET /health` - Проверка здоровья сервиса

## Конфигурация

### Основные параметры в `config.py`:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `chunk_size` | Размер фрагмента документа | 500 |
| `chunk_overlap` | Перекрытие между фрагментами | 50 |
| `top_k` | Количество результатов поиска | 5 |
| `similarity_threshold` | Порог релевантности | 0.5 |
| `llm_temperature` | Креативность ответов (0-1) | 0.5 |
| `max_tokens` | Макс. длина ответа | 1000 |

### Выбор модели embeddings

```python
# Вариант 1: Sentence Transformers (быстрее, локально)
USE_GIGACHAT_EMBEDDINGS=False
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Вариант 2: GigaChat Embeddings (точнее, через API)
USE_GIGACHAT_EMBEDDINGS=True
```

## Telegram Bot команды

- `/start` - Начать работу с ботом
- `/help` - Справка по использованию

## Как это работает (RAG Pipeline)

1. **Загрузка документов**
    - Пользователь загружает учебные материалы (PDF, DOCX, etc.)
    - Документы разбиваются на небольшие фрагменты (chunks)

2. **Создание embeddings**
    - Каждый chunk преобразуется в векторное представление
    - Векторы сохраняются в ChromaDB

3. **Обработка запроса**
    - Вопрос пользователя векторизуется
    - Выполняется семантический поиск похожих chunks

4. **Генерация ответа**
    - Релевантные chunks передаются в GigaChat как контекст
    - LLM генерирует ответ на основе найденной информации

5. **Возврат результата**
    - Ответ отправляется пользователю
    - Указываются источники и уровень уверенности
