"""utils/helpers.py — Small shared utilities."""


def memory_label(store: list[dict]) -> str:
    """Return a short label like '4 turns remembered'."""
    turns = len(store) // 2
    if turns == 0:
        return ""
    return f"{turns} turn{'s' if turns != 1 else ''} remembered"


def sanitise(text: str, max_len: int = 2000) -> str:
    return text.strip()[:max_len]
