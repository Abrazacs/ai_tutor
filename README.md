# ai_tutor
AI-репетитора по техническим дисциплинам

## Структура проекта
```commandline
ai_tutor/
├─ main.py
├─ config.py
├─ requirements.txt
├─ README.md
│
├─ data/
│  ├─ raw/            # исходные учебные материалы (.txt, .md)
│  └─ index/          # векторный индекс (создаётся автоматически)
│
├─ rag/
│  ├─ ingestion.py    # подготовка материалов → чанки → индекс
│  ├─ embeddings.py   # эмбеддинги sentence-transformers
│  ├─ vector_store.py # работа с ChromaDB
│  ├─ qa_pipeline.py  # весь RAG-пайплайн
│  ├─ local_llm.py    # локальная LLM (Qwen2.5-0.5B-Instruct)
│  └─ safety.py
│
└─ bot/
   ├─ telegram_bot.py
   ├─ handlers.py
   └─ state.py
```


# Инструкция по сборке и запуску AI-Tutor (RAG + Local LLM)

## 1. Установка и подготовка окружения

### 1.1. Создать виртуальное окружение

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

Linux/macOS:

```python3 -m venv venv
source venv/bin/activate
```

1.2. Установить зависимости
```pip install -r requirements.txt```

2. Подготовка данных для RAG

Поместите учебные материалы в текстовом формате в каталог:

data/raw/


Например:

data/raw/network_basics.txt
data/raw/linux_basics.txt
data/raw/kubernetes_basics.txt
data/raw/cloud_intro.txt


Поддерживаются .txt и .md.
Другие форматы необходимо предварительно конвертировать в текст.

3. Построение векторного индекса

Запустить из корня проекта:

python -m rag.ingestion


Пример вывода:

Indexed N chunks.


После этого в каталоге:

data/index/


появится векторный индекс ChromaDB.

4. Запуск Telegram-бота

Создайте файл .env в корне проекта:

TELEGRAM_TOKEN=ваш_токен_бота


Токен выдаёт BotFather.

Запуск бота:

python main.py


Если всё в порядке — бот начнёт принимать сообщения.

5. Как работает AI-репетитор

Поддерживаемые команды:

/start      — приветствие и помощь
/ask        — задать вопрос
/quiz       — сгенерировать вопросы по теме
/resources  — рекомендовать материалы по теме


Также можно просто написать вопрос. Тогда бот:

Находит релевантные документы через эмбеддинги и ChromaDB

Формирует RAG-промпт

Передаёт его в локальную LLM

Возвращает развернутый ответ + список источников

6. Локальная LLM

Используемая модель:

Qwen/Qwen2.5-0.5B-Instruct


Загружается автоматически при первом запуске.

Пример импорта:

from transformers import AutoModelForCausalLM, AutoTokenizer

Требования:

1.5–2 ГБ RAM минимум

работает на CPU (медленно)

GPU ускоряет, но не обязателен

7. Расширение проекта

Можно добавить:

автоматический парсер PDF/DOCX

дополнительные предметы (просто добавьте файлы в data/raw/)

web-интерфейс (Streamlit, Gradio)

подключение облачных LLM вместо локальной

токсичность-фильтр

мониторинг качества

8. Команда полного запуска проекта
# 1. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\Activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Построить векторный индекс
python -m rag.ingestion

# 4. Запустить Telegram-бота
python main.py