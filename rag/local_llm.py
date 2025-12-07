# rag/local_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"[local_llm] Загружаю модель {_MODEL_NAME} (это может занять время)...")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    torch_dtype=torch.float32,     # на CPU
)
_model.eval()

_DEVICE = "cpu"  # если есть GPU и установлен cuda-торч, можно поменять на "cuda"


def call_llm(prompt: str) -> str:
    """
    Получает строку-promt, возвращает ответ модели (строкой).
    """
    system_msg = "Ты AI-репетитор по техническим дисциплинам. Объясняй понятно и по шагам."
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

    # Очень грубо: пытаемся вернуть текст после "Assistant:"
    if "Assistant:" in full_text:
        return full_text.split("Assistant:")[-1].strip()

    return full_text.strip()
