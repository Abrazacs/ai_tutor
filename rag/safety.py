_BAD_WORDS = {"бля", "хуй", "сука"}  # можно расширить


def is_text_safe(text: str) -> bool:
    lower = text.lower()
    return not any(bad in lower for bad in _BAD_WORDS)


def sanitize_user_text(text: str) -> str:
    return text.strip()
