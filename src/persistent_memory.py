"""
Titans Memory — Persistent Memory
===================================
Fixed text tokens prepended to every prompt.
Encodes stable task knowledge: persona, rules, domain context.

Uses <SOS>/<EOS> delimiters to mark boundaries and supports
max_length to control the total persistent context size.

Paper Section 3.3: P = [p1, p2, ..., p_Np] — learnable but input-independent.
"""

from typing import List, Optional


class PersistentMemory:
    """Manages fixed context strings that are injected into every prompt.

    Each persistent block is wrapped with start/end-of-sequence markers:
        <SOS> [PERSISTENT] token text <EOS>

    max_length controls the total character budget — oldest tokens are
    dropped from the front when the rendered output would exceed it.
    """

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        max_length: int = 0,
    ):
        self._tokens: List[str] = list(tokens) if tokens else []
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length      # 0 = unlimited

    def add(self, text: str) -> None:
        """Append a persistent context token."""
        self._tokens.append(text)

    def remove(self, index: int) -> None:
        """Remove a persistent token by index."""
        if 0 <= index < len(self._tokens):
            self._tokens.pop(index)

    def clear(self) -> None:
        """Remove all persistent tokens."""
        self._tokens.clear()

    def render(self) -> str:
        """Render persistent tokens wrapped with SOS/EOS markers.

        If max_length > 0, drops oldest tokens until the rendered
        output fits within the character budget.
        """
        if not self._tokens:
            return ""

        lines = [
            f"{self.sos_token} [PERSISTENT] {t} {self.eos_token}"
            for t in self._tokens
        ]

        if self.max_length > 0:
            # Keep most recent tokens that fit within budget
            selected: List[str] = []
            total = 0
            for line in reversed(lines):
                cost = len(line) + 1  # +1 for the newline separator
                if total + cost > self.max_length:
                    break
                selected.append(line)
                total += cost
            lines = list(reversed(selected))

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tokens)

    def state_dict(self) -> dict:
        return {
            "tokens": list(self._tokens),
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "max_length": self.max_length,
        }

    def load_state_dict(self, d: dict) -> None:
        self._tokens = list(d.get("tokens", []))
        self.sos_token = d.get("sos_token", "<SOS>")
        self.eos_token = d.get("eos_token", "<EOS>")
        self.max_length = d.get("max_length", 0)
