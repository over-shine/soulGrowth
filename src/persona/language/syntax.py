"""句法等级与话语生成。

「不许越级说话」在这里变成可执行的规则：
* 阶段决定 `L` 的上限（S0 只能是 L0 = 没有词）；
* `L` 只能单调上升（已学会的等级不会掉）；
* 生成时只允许使用词库里 `productive` 的词形，等级装不下的意思**整句放弃**，不截断、不换词。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from persona.errors import CompileError
from persona.language.lexicon import Lexicon
from persona.types import Intent

LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5")
LEVEL_INDEX = {lv: i for i, lv in enumerate(LEVELS)}

#: 阶段上限 / 下限（docs/06）
STAGE_MAX: dict[str, str] = {"S0": "L0", "S1": "L1", "S2": "L2", "S3": "L3", "S4": "L4", "S5": "L5", "S6": "L5"}
STAGE_MIN: dict[str, str] = {"S0": "L0", "S1": "L0", "S2": "L1", "S3": "L2", "S4": "L3", "S5": "L4", "S6": "L5"}

#: 一条话语最多几个词（docs/06 MAX_TOKENS）
MAX_TOKENS: dict[str, int] = {"S0": 0, "S1": 1, "S2": 2, "S3": 4, "S4": 8, "S5": 20, "S6": 999}

#: 无词时期（S0）的表达形态
PRE_VERBAL_MODES = ("pre_verbal", "cry", "coo", "babble")

#: 构式：等级 → 槽位模板。`{a}` 是主语/指称，`{b}` 是谓词。
CONSTRUCTIONS: dict[str, str] = {
    "L0": "",
    "L1": "{a}",
    "L2": "{a}{b}",
    "L3": "{a}{b}{c}",
}

_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = set("，。！？、,.!?；：;:「」“”‘’（）()")


def stage_max_level(stage: str) -> str:
    return STAGE_MAX.get(stage, "L0")


def max_tokens(stage: str) -> int:
    return int(MAX_TOKENS.get(stage, 0))


def clip_level(level: str, stage: str) -> str:
    """把等级夹到该阶段允许的区间；永不越级。"""
    if level not in LEVEL_INDEX:
        level = "L0"
    lo, hi = LEVEL_INDEX[STAGE_MIN.get(stage, "L0")], LEVEL_INDEX[STAGE_MAX.get(stage, "L0")]
    return LEVELS[min(hi, max(lo, LEVEL_INDEX[level]))]


def monotonic_level(prev: str, level: str) -> str:
    """等级单调不减（docs/06：L 只能往上）。"""
    return LEVELS[max(LEVEL_INDEX.get(prev, 0), LEVEL_INDEX.get(level, 0))]


def split_words(text: str | None, lexicon: Lexicon | None = None) -> list[str]:
    """按词库最长匹配把话语切成词（切不出的字各算一个词）。

    出口那道门要用**同一个**切法来判断「这个词是不是他学会的词」。
    早先它按空格切，于是「宝宝球球」被当成一个词，双词句在门口被拦成 None：
    说出去的话变成了空，而事件里记着他说了。
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    forms = sorted(
        lexicon.productive_forms() if lexicon is not None else (), key=len, reverse=True
    )
    if not forms:
        if _HAS_CJK.search(text):
            return [c for c in text if not c.isspace()]
        return text.split()
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch in _PUNCT:
            i += 1
            continue
        hit = next((w for w in forms if text.startswith(w, i)), None)
        if hit:
            out.append(hit)
            i += len(hit)
        else:
            out.append(ch)
            i += 1
    return out


def count_tokens(text: str | None, lexicon: Lexicon | None = None) -> int:
    """**词**数，不是字数（docs/06 的 MAX_TOKENS 说的是「最多几个词」）。

    有词库时按最长匹配切词：`宝宝` 是一个词，不是两个字。拿字数当词数会把
    S1 的一句话永远判成超长——那样他一张嘴就被挡回去，一句话也说不出来。
    """
    return len(split_words(text, lexicon))


@dataclass
class Realized:
    utterance: str | None
    level: str
    mode: str
    degraded: bool = False
    chain: tuple[str, ...] = ()


def realize(intent: Intent | None, *, level: str, stage: str, lexicon: Lexicon) -> Realized:
    """把结构化意图实现成一条**可以说出口**的话语。

    S0 返回 `utterance=None`（`mode="pre_verbal"`）：这不是「没想好」，而是「没有词」。
    """
    lv = clip_level(level, stage)
    if stage == "S0" or lv == "L0":
        return Realized(utterance=None, level="L0", mode="pre_verbal")

    if intent is None:
        raise CompileError("没有结构化意图，无法生成话语")

    slots = _bind_slots(intent, lexicon)
    template = CONSTRUCTIONS.get(lv, "{a}")
    try:
        text = template.format(**slots)
    except KeyError as exc:  # 槽位缺失 → 放弃（不降级到「凑词」）
        raise CompileError(f"槽位未绑定：{exc}") from exc

    problems = check_utterance(text, level=lv, stage=stage, lexicon=lexicon)
    if problems:
        raise CompileError("；".join(problems))
    return Realized(utterance=text, level=lv, mode="verbal")


def _bind_slots(intent: Intent, lexicon: Lexicon) -> dict[str, str]:
    """槽位绑定：只从 productive 词里挑，挑不到就报错。"""
    slots: dict[str, str] = {}
    a = _pick_form(intent.referent_id or intent.goal, lexicon)
    if a is None:
        raise CompileError("指称没有可产出的词")
    slots["a"] = a
    if LEVEL_INDEX["L1"] < LEVEL_INDEX["L2"]:
        pass
    b = _pick_form(intent.predicate_type or "", lexicon)
    if b is not None:
        slots["b"] = b
    slots.setdefault("c", "")
    return slots


def _pick_form(hint: str | None, lexicon: Lexicon) -> str | None:
    """在已产出的词里找与提示最匹配的那个。

    匹配顺序：form 命中 → gloss 命中 → semantic_field 命中。取不到就返回 None，
    **绝不**用近义词或拼音替代（docs/06 三条硬规则之二）。
    """
    if not hint:
        return None
    pool = lexicon.productive()
    for e in pool:
        if e.form == hint:
            return e.form
    for e in pool:
        if e.gloss and hint in e.gloss:
            return e.form
    for e in pool:
        if e.semantic_field and e.semantic_field == hint:
            return e.form
    return None


def check_utterance(
    text: str, *, level: str, stage: str, lexicon: Lexicon, allow_imitation: bool = False
) -> list[str]:
    """出口检查：等级、词表、长度。三道闸门对应 docs/06 的第 2、3 类反作弊。"""
    problems: list[str] = []
    if not text:
        return problems
    if LEVEL_INDEX[level] > LEVEL_INDEX[stage_max_level(stage)]:
        problems.append(f"越级：{level} > 阶段上限 {stage_max_level(stage)}")
    n = count_tokens(text, lexicon)
    if n > max_tokens(stage):
        problems.append(f"超长：{n} 词 > 上限 {max_tokens(stage)}")
    productive = lexicon.productive_forms()
    if productive:
        for piece in _tokens_of(text):
            if piece not in productive:
                problems.append(f"未习得词形：{piece}")
    return problems


def _tokens_of(text: str) -> list[str]:
    """把话语按空格/标点切块（词形内部的字不拆）。"""
    return [t for t in re.split(r"[\s，。！？、,.!?]+", text.strip()) if t]
