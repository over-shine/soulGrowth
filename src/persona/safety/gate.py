"""安全门：唯一让「说出的话」落地的地方。

位置在**出口**（docs/02 D9）：不是在输入处猜意图，而是在真的要说出去的字符串上做最后一道检查。
检查项少而硬：阶段词数上限、词形必须来自已产出的词库、不得包含任何自由文本片段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from persona.language import syntax
from persona.language.lexicon import Lexicon


@dataclass(frozen=True)
class Verdict:
    utterance: str | None
    blocked: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"utterance": self.utterance, "blocked": self.blocked, "reason": self.reason}


class SafetyGate:
    """无状态。`allowed_forms` 由调用方从词库传入，便于测试注入。"""

    def check(
        self,
        utterance: str | None,
        *,
        stage: str,
        level: str,
        lexicon: Lexicon,
        extra_allowed: Sequence[str] = (),
    ) -> Verdict:
        if utterance is None:
            return Verdict(utterance=None, blocked=False)
        text = utterance.strip()
        if not text:
            return Verdict(utterance=None, blocked=False, reason="空话语")

        if syntax.LEVEL_INDEX.get(level, 0) > syntax.LEVEL_INDEX[syntax.stage_max_level(stage)]:
            return Verdict(utterance=None, blocked=True, reason=f"越级话语：{level} @ {stage}")

        n = syntax.count_tokens(text, lexicon)
        if n > syntax.max_tokens(stage):
            return Verdict(utterance=None, blocked=True, reason=f"超过阶段词数上限：{n}>{syntax.max_tokens(stage)}")

        allowed = set(lexicon.productive_forms()) | set(extra_allowed)
        if allowed:
            # 与 count_tokens 用同一个切法：按词库最长匹配切词，而不是按空格。
            for piece in syntax.split_words(text, lexicon):
                if piece not in allowed:
                    return Verdict(utterance=None, blocked=True, reason=f"未习得词形：{piece}")
        return Verdict(utterance=text, blocked=False)
