from typing import List, Dict, Any
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from src.config import settings
from src.models.document import QueryResponse


class LLMService:
    """Сервис для работы с языковой моделью GigaChat"""

    SYSTEM_PROMPT = """Ты — AI-репетитор, который помогает студентам учиться и понимать материал.

Твоя задача:
1. Внимательно анализировать предоставленный контекст из учебных материалов
2. Давать четкие, структурированные и понятные ответы на вопросы студентов
3. Использовать только информацию из предоставленного контекста
4. Если ответа нет в контексте, честно сообщить об этом
5. Приводить примеры и объяснения для лучшего понимания
6. Адаптировать сложность объяснений под уровень студента

Правила:
- Всегда основывай ответ на предоставленном контексте
- Будь точным и конкретным
- Используй простой и понятный язык
- Структурируй ответ для лучшего восприятия
- Если нужно, задавай уточняющие вопросы"""

    def __init__(self, credentials: str = None, model: str = None):
        """
        Инициализация LLM сервиса с GigaChat

        Args:
            credentials: Авторизационные данные GigaChat
            model: Модель GigaChat (GigaChat, GigaChat-Plus, GigaChat-Pro)
        """
        self.credentials = credentials or settings.gigachat_credentials
        self.model = model or settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.max_tokens

        self.client = GigaChat(
            credentials=self.credentials,
            scope=settings.gigachat_scope,
            verify_ssl_certs=settings.gigachat_verify_ssl,
            model=self.model
        )

        print(f"GigaChat инициализирован: модель {self.model}")

    def generate_prompt(self, query: str, context: str) -> str:
        """
        Генерирует промпт для LLM

        Args:
            query: Вопрос пользователя
            context: Контекст из векторной БД

        Returns:
            Сформированный промпт
        """
        prompt = f"""Контекст из учебных материалов:
{context}

Вопрос студента: {query}

Дай развернутый ответ на вопрос, используя информацию из контекста выше."""

        return prompt

    def generate_answer(
            self,
            query: str,
            context: str,
            temperature: float = None,
            max_tokens: int = None
    ) -> str:
        """
        Генерирует ответ на вопрос с учетом контекста

        Args:
            query: Вопрос пользователя
            context: Контекст из векторной БД
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов

        Returns:
            Ответ LLM
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        user_prompt = self.generate_prompt(query, context)

        try:
            # Формируем сообщения для GigaChat
            messages = [
                Messages(
                    role=MessagesRole.SYSTEM,
                    content=self.SYSTEM_PROMPT
                ),
                Messages(
                    role=MessagesRole.USER,
                    content=user_prompt
                )
            ]

            # Создаем chat объект
            chat = Chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Получаем ответ от GigaChat
            response = self.client.chat(chat)

            # Извлекаем текст ответа
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            print(f"Ошибка при обращении к GigaChat: {e}")
            return f"Извините, произошла ошибка при генерации ответа: {str(e)}"

    def generate_with_sources(
            self,
            query: str,
            context: str,
            sources: List[Dict[str, Any]]
    ) -> QueryResponse:
        """
        Генерирует ответ с указанием источников

        Args:
            query: Вопрос пользователя
            context: Контекст из векторной БД
            sources: Список источников

        Returns:
            Структурированный ответ с источниками
        """
        answer = self.generate_answer(query, context)

        # Вычисляем среднюю уверенность по источникам
        avg_confidence = (
            sum(s.get('similarity', 0) for s in sources) / len(sources)
            if sources else 0.0
        )

        formatted_sources = []
        for source in sources:
            formatted_sources.append({
                'file': source['metadata'].get('source', 'Unknown'),
                'similarity': source.get('similarity', 0),
                'preview': source['content'][:200] + '...'
            })

        return QueryResponse(
            answer=answer,
            sources=formatted_sources,
            confidence=avg_confidence
        )

    def generate_followup_questions(self, query: str, answer: str) -> List[str]:
        """
        Генерирует дополнительные вопросы для углубления в тему

        Args:
            query: Исходный вопрос
            answer: Данный ответ

        Returns:
            Список дополнительных вопросов
        """
        prompt = f"""На основе этого вопроса и ответа:

Вопрос: {query}
Ответ: {answer}

Предложи 3 дополнительных вопроса, которые помогут студенту углубить понимание темы.
Вопросы должны быть конкретными и связанными с темой."""

        try:
            messages = [
                Messages(
                    role=MessagesRole.SYSTEM,
                    content="Ты помогаешь формулировать учебные вопросы."
                ),
                Messages(
                    role=MessagesRole.USER,
                    content=prompt
                )
            ]

            chat = Chat(
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            response = self.client.chat(chat)
            questions_text = response.choices[0].message.content

            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and q.strip()[0].isdigit()]

            return questions[:3]

        except Exception as e:
            print(f"Ошибка при генерации дополнительных вопросов: {e}")
            return []

    def stream_answer(self, query: str, context: str):
        """
        Генерирует ответ в потоковом режиме (streaming)

        Args:
            query: Вопрос пользователя
            context: Контекст из векторной БД

        Yields:
            Части ответа по мере генерации
        """
        user_prompt = self.generate_prompt(query, context)

        try:
            messages = [
                Messages(role=MessagesRole.SYSTEM, content=self.SYSTEM_PROMPT),
                Messages(role=MessagesRole.USER, content=user_prompt)
            ]

            chat = Chat(messages=messages, temperature=self.temperature)

            for chunk in self.client.stream(chat):
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Ошибка: {str(e)}"