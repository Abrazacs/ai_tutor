from dataclasses import dataclass, field
from typing import Dict


@dataclass
class UserState:
    mode: str = "chat"       # chat | quiz
    last_topic: str = ""
    current_questions: list[str] = field(default_factory=list)


_states: Dict[int, UserState] = {}


def get_state(user_id: int) -> UserState:
    if user_id not in _states:
        _states[user_id] = UserState()
    return _states[user_id]
