"""词库：这孩子**已经会说的词**。

四条不变式（docs/06）：
* `form` 必须是照料者话语的连续子串——词不是凭空生成的；
* `gloss` 只给内部看，永不进 prompt、永不落到 episode；
* 只有 `status == "productive"` 的词能进 utterance；
* `first_heard_day <= first_produced_day`。

S0 时词库是空的，而且**必须**是空的：前语言期没有词，只有哭声与咕咕声。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from persona.types import STAGES

#: 产出阈值 k（docs/06）：听到/尝试多少次之后才算「会说」。
#: S2 取区间下沿（2–4 次）：幼儿学词靠的是高频重复，不是「练够八次」。
PRODUCTION_THRESHOLD: dict[str, int] = {"S0": 99, "S1": 8, "S2": 2, "S3": 1, "S4": 1, "S5": 1, "S6": 1}
#: 理解词库的进入条件
COMPREHENSION_MIN = 0.30
#: 统计学分段：一个片段至少在**多少种不同说法**里出现过，才可能成为一个词形
MIN_SPAN_UTTERANCES = 3
#: 每个片段最多记住多少种来源说法（只用来判断「够不够常见」）
MAX_SPAN_SOURCES = 12
#: 至少要跨几类情境出现过（只在一类情境里出现的，是一整句而不是词）
MIN_SPAN_CONTEXTS = 2
#: 词形最短几个字（单字的边界在句子里听不出来，先不进词库）
MIN_WORD_LEN = 2
#: 单字词的门槛更高：汉语里「我/你/要/不」确实是词，但单字片段太容易是巧合，
#: 所以要求它出现在更多说法、更多情境里，才承认它是个词（docs/06 的 M2.8 要用代词）。
SINGLE_CHAR_MIN_UTTERANCES = 8
SINGLE_CHAR_MIN_CONTEXTS = 3
SINGLE_CHAR_FORMS = frozenset({"我", "你", "要", "不", "是", "好", "看", "吃", "喝", "走", "抱", "给"})
#: 片段最长几个字（S1 的词不会更长）
MAX_SPAN = 3

STATUSES = ("heard_only", "emerging", "productive", "dormant")


def _spans_of(text: str) -> set[str]:
    """一句话里所有 1..MAX_SPAN 字的连续片段（同一条话语里只算一次）。"""
    out: set[str] = set()
    for n in range(1, MAX_SPAN + 1):
        for i in range(len(text) - n + 1):
            span = text[i : i + n]
            if span.strip():
                out.add(span)
    return out


@dataclass
class LexiconEntry:
    id: str
    form: str
    gloss: str
    kind: str = "word"  # word | frozen | gesture
    pos: str = "unknown"
    first_heard_day: int = 0
    first_produced_day: int | None = None
    source: list[str] = field(default_factory=list)  # 前 5 条来源句子
    exposures: int = 0
    produced_count: int = 0
    comprehension_confidence: float = 0.0
    production_threshold: int = 8
    syllables: list[str] = field(default_factory=list)
    phonotactic_ok: list[bool] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)
    semantic_field: str = "general"
    status: str = "heard_only"
    gloss_internal_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LexiconEntry:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class Lexicon:
    """词库。写操作只有 `hear` / `produce` / `promote` 三个入口。"""

    def __init__(self, entries: Iterable[LexiconEntry] = (), *, stage: str = "S0") -> None:
        self.entries: dict[str, LexiconEntry] = {e.id: e for e in entries}
        self.stage = stage
        self._by_form: dict[str, str] = {e.form: e.id for e in self.entries.values()}
        self._seq = len(self.entries)
        # 统计学分段的原料：一个片段在**多少种不同的说法**里出现过。
        # 词会出现在很多句子里的不同位置；一整句只在一句话里出现，所以长不成词。
        self.spans: dict[str, int] = {}
        self.span_sources: dict[str, list[str]] = {}
        self.span_contexts: dict[str, list[str]] = {}
        self.speech_count: int = 0

    # -- 查询 ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def get(self, entry_id: str) -> LexiconEntry | None:
        return self.entries.get(entry_id)

    def by_form(self, form: str) -> LexiconEntry | None:
        eid = self._by_form.get(form)
        return self.entries.get(eid) if eid else None

    def productive(self) -> list[LexiconEntry]:
        return [e for e in self.entries.values() if e.status == "productive"]

    def productive_forms(self) -> set[str]:
        return {e.form for e in self.entries.values() if e.status == "productive"}

    def can_produce(self, form: str) -> bool:
        return form in self.productive_forms()

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for e in self.entries.values():
            by_status[e.status] = by_status.get(e.status, 0) + 1
        return {
            "total": len(self.entries),
            "productive": len(self.productive()),
            "by_status": by_status,
            "stage": self.stage,
            "speech_count": self.speech_count,
            "spans": len(self.spans),
        }

    # -- 写入 ---------------------------------------------------------------
    def hear(self, form: str, *, day: int, utterance: str, gloss: str = "", now_iso: str = "") -> LexiconEntry:
        """听见一个词形。`form` 必须是 `utterance` 的连续子串（否则直接拒绝）。"""
        if not form or form not in utterance:
            raise ValueError(f"词形 {form!r} 不是话语 {utterance!r} 的子串——词不能凭空生成")
        existing = self.by_form(form)
        if existing:
            existing.exposures += 1
            existing.comprehension_confidence = min(
                1.0, existing.comprehension_confidence + 0.02
            )
            if len(existing.source) < 5:
                existing.source.append(utterance)
            return existing
        self._seq += 1
        entry = LexiconEntry(
            id=f"lex_{self._seq:06d}",
            form=form,
            gloss=gloss,
            first_heard_day=day,
            source=[utterance],
            exposures=1,
            comprehension_confidence=0.10,
            production_threshold=PRODUCTION_THRESHOLD.get(self.stage, 8),
            syllables=list(form),
            phonotactic_ok=[True] * len(form),
            status="heard_only",
        )
        self.entries[entry.id] = entry
        self._by_form[form] = entry.id
        return entry

    def produce(self, form: str, *, day: int, as_word: bool = False) -> LexiconEntry:
        """说一次（练习也算说）。达到阈值且当前阶段允许 → 升为 productive。

        `as_word=True` 表示这次是**真的当作一个词说出来**（A1.WORD），只有这种才记
        `first_produced_day`。咿呀练习同样计入 `produced_count`——练习次数决定他说不说
        得出来，但「第一次说出一个词」是另一件事，不能拿练习冒充。
        """
        entry = self.by_form(form)
        if entry is None:
            raise ValueError(f"词库里没有 {form!r}")
        entry.produced_count += 1
        if as_word and entry.first_produced_day is None:
            entry.first_produced_day = day
        threshold = PRODUCTION_THRESHOLD.get(self.stage, 8)
        entry.production_threshold = threshold
        if entry.status != "productive" and entry.produced_count >= threshold:
            entry.status = "productive"
        elif entry.status == "heard_only":
            entry.status = "emerging"
        return entry

    def promote(self, form: str) -> bool:
        """显式晋升（供巩固/夜间任务使用）。返回是否发生变化。"""
        entry = self.by_form(form)
        if entry is None or entry.status == "productive":
            return False
        if entry.comprehension_confidence < COMPREHENSION_MIN:
            return False
        entry.status = "productive"
        return True

    def set_stage(self, stage: str) -> None:
        if stage in STAGES:
            self.stage = stage

    # -- 统计学分段：词形是从听见过的话里**长**出来的，不是查词典查出来的 ----
    def hear_speech(self, utterance: str, *, day: int, context: str = "") -> list[LexiconEntry]:
        """听见一整句。

        切分不靠词典，靠**重复**：一个字串只有在多条不同说法、不同情境里反复出现，
        才可能是一个词（Saffran 式的统计学习）。返回这一次新长出来的词条。
        """
        text = (utterance or "").strip()
        if not text:
            return []
        self.speech_count += 1
        for span in _spans_of(text):
            srcs = self.span_sources.setdefault(span, [])
            if text not in srcs and len(srcs) < MAX_SPAN_SOURCES:
                srcs.append(text)
            self.spans[span] = len(srcs)
            if context:
                ctxs = self.span_contexts.setdefault(span, [])
                if context not in ctxs:
                    ctxs.append(context)

        fresh: list[LexiconEntry] = []
        for form in self._discovered():
            if self.by_form(form) is None:
                fresh.append(self.hear(form, day=day, utterance=text))
        # 已经认识的词，再听见一次也算一次暴露（理解是慢慢涨的）
        for form in sorted(self._by_form):
            if form in text and self.by_form(form) not in fresh:
                self.hear(form, day=day, utterance=text)
        return fresh

    def _discovered(self) -> list[str]:
        """当前够格成为词形的片段：够常见、跨情境、且不是更长片段的碎片。"""
        counts = self.spans
        out: list[str] = []
        for form, n in counts.items():
            if n < MIN_SPAN_UTTERANCES or self.by_form(form) is not None:
                continue
            # 单字只收真正的功能词（我/你/要/不…），而且门槛更高：单字片段太容易是巧合。
            # 汉语里「我」确实是一个词，S2 的 M2.8 要用它，不能因为短就永远排除。
            if len(form) < MIN_WORD_LEN:
                if form not in SINGLE_CHAR_FORMS:
                    continue
                if n < SINGLE_CHAR_MIN_UTTERANCES:
                    continue
                if len(self.span_contexts.get(form, [])) < SINGLE_CHAR_MIN_CONTEXTS:
                    continue
            # 只在一类情境里出现的，多半是跟那个情境绑死的一整句，不是词
            if len(self.span_contexts.get(form, [])) < MIN_SPAN_CONTEXTS:
                continue
            # 「妈」之于「妈妈」：如果某个更长的片段吃掉了它六成以上的出现，
            # 它多半只是那个词的一部分，还不是一个词。
            # 例外是封闭类功能词：「我」永远被「我是」「我抱」包着，但它确实是词，
            # 而且正是 M2.8 要的那个——闭类词不受这条判定约束。
            fragment = False
            if not (len(form) == 1 and form in SINGLE_CHAR_FORMS):
                fragment = any(
                    len(other) > len(form) and form in other and counts[other] >= 0.6 * n
                    for other in counts
                )
            if not fragment:
                out.append(form)
        out.sort(key=lambda s: (-len(s), s))
        return out

    def practice_form(self) -> LexiconEntry | None:
        """该练哪个词：还说不利索的里面，听得最多的那个。"""
        pool = [e for e in self.entries.values() if e.status in ("heard_only", "emerging")]
        if not pool:
            return None
        pool.sort(key=lambda e: (e.exposures, e.comprehension_confidence), reverse=True)
        return pool[0]

    def known_forms(self) -> list[str]:
        return sorted(self._by_form)

    # -- 序列化 -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "speech_count": self.speech_count,
            "spans": dict(sorted(self.spans.items())),
            "span_sources": {k: list(v) for k, v in sorted(self.span_sources.items())},
            "span_contexts": {k: list(v) for k, v in sorted(self.span_contexts.items())},
            "entries": [e.to_dict() for e in sorted(self.entries.values(), key=lambda x: x.id)],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lexicon:
        lex = cls(
            [LexiconEntry.from_dict(e) for e in d.get("entries", [])],
            stage=str(d.get("stage", "S0")),
        )
        lex.spans = {str(k): int(v) for k, v in (d.get("spans") or {}).items()}
        lex.span_sources = {
            str(k): [str(s) for s in v] for k, v in (d.get("span_sources") or {}).items()
        }
        lex.span_contexts = {
            str(k): [str(s) for s in v] for k, v in (d.get("span_contexts") or {}).items()
        }
        lex.speech_count = int(d.get("speech_count", 0))
        return lex

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> Lexicon:
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
