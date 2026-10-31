"""用量预算：单 episode ≤2 次调用、输入 ≤3000、输出 ≤400 token（docs/02 D5）。

超了不是「再试一次」，而是**降级**：改用启发式打分，并记录原因。
"""

from __future__ import annotations

from dataclasses import dataclass

from persona.cognition.policy import Prior


@dataclass
class Budget:
    max_calls: int = 2
    max_input_tokens: int = 3000
    max_output_tokens: int = 400
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def can_call(self) -> bool:
        return self.calls < self.max_calls

    def charge(self, p: Prior) -> None:
        self.calls += 1
        self.tokens_in += max(0, int(p.tokens_in))
        self.tokens_out += max(0, int(p.tokens_out))

    @property
    def exhausted(self) -> bool:
        return not self.can_call()

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "max_calls": self.max_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
        }
