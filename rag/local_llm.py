# rag/local_llm.py
import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Настройки GigaChat от Cloud.ru 
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY")
CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://foundation-models.api.cloud.ru/v1")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "ai-sage/GigaChat3-10B-A1.8B")

USE_CLOUD = bool(CLOUD_API_KEY)



# ВАРИАНТ 1: Облако (GigaChat через OpenAI)

if USE_CLOUD:
    print(f"[local_llm] Используем облачную модель: {CLOUD_MODEL}")
    from openai import OpenAI

    client = OpenAI(
        api_key=CLOUD_API_KEY,
        base_url=CLOUD_BASE_URL
    )

    def call_llm(prompt: str) -> str:
        """
        Отправляет запрос в GigaChat (Cloud.ru).
        """
        try:
            response = client.chat.completions.create(
                model=CLOUD_MODEL,
                messages=[
                    {"role": "system", "content": "Ты — AI-репетитор по техническим дисциплинам. Объясняй понятно и по шагам."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Ошибка API GigaChat: {e}"


# ВАРИАНТ 2: Локальная Qwen
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"[local_llm] Ключ CLOUD_API_KEY не найден. Загружаю локальную модель {_MODEL_NAME}...")
    
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        torch_dtype=torch.float32,     # на CPU
    )
    _model.eval()

    _DEVICE = "cpu"  # если есть GPU и установлен cuda-торч, можно поменять на "cuda"

    def call_llm(prompt: str) -> str:
        """
        Локальный инференс Qwen (CPU).
        """
        # Твой системный промпт
        system_msg = "Ты AI-репетитор по техническим дисциплинам. Объясняй понятно и по шагам."
        
        # Твоя ручная сборка промпта
        text = f"System: {system_msg}\nUser: {prompt}\nAssistant:"

        inputs = _tokenizer(text, return_tensors="pt").to(_DEVICE)

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=_tokenizer.eos_token_id,
            )

        full_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Твой парсинг ответа
        if "Assistant:" in full_text:
            return full_text.split("Assistant:")[-1].strip()

        return full_text.strip()
