"""核心数据类型：Episode 及七个子结构。

契约见 `docs/13-数据契约.md`。两条实现约定：

1. 落盘一律 `r6()`（即 %.6f）+ `json.dumps(sort_keys=True)`，保证同一输入字节相同；
2. 每个结构都自带 `to_dict()`，字段顺序即契约顺序，便于人读与 diff。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"

STAGES: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")

#: docs/13 `action.source`
ACTION_SOURCES: tuple[str, ...] = ("llm", "heuristic", "reflex", "fallback")

#: docs/06 L0–L5
SYNTAX_LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5")


def r6(x: float) -> float:
    """统一落盘精度。非有限值直接拒绝——它们几乎总是 bug，不该被静默写进日志。"""
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"non-finite float: {x!r}")
    out = float(f"{v:.6f}")
    return 0.0 if out == 0 else out


def clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def clip11(x: float) -> float:
    return clip(float(x), -1.0, 1.0)


def clip01(x: float) -> float:
    return clip(float(x), 0.0, 1.0)


def _opt(v: Any) -> Any:
    """空集合在落盘时省略，避免 JSON 里大量 null/[] 噪声。"""
    return v if v not in (None, "", (), [], {}) else None


@dataclass(frozen=True)
class Appraisal:
    """评价向量。valence/pe/threat/control ∈[-1,1]；arousal/novelty ∈[0,1]。"""

    valence: float = 0.0
    arousal: float = 0.0
    threat: float = 0.0
    control: float = 0.0
    pe: float = 0.0
    novelty: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", clip11(self.valence))
        object.__setattr__(self, "arousal", clip01(self.arousal))
        object.__setattr__(self, "threat", clip11(self.threat))
        object.__setattr__(self, "control", clip11(self.control))
        object.__setattr__(self, "pe", clip11(self.pe))
        object.__setattr__(self, "novelty", clip01(self.novelty))

    def to_dict(self) -> dict[str, float]:
        return {
            "valence": r6(self.valence),
            "arousal": r6(self.arousal),
            "threat": r6(self.threat),
            "control": r6(self.control),
            "pe": r6(self.pe),
            "novelty": r6(self.novelty),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Appraisal:
        return cls(**{k: float(d.get(k, 0.0)) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Drives:
    """内驱力，全部 ∈[0,1]；1 = 饱和/满足，0 = 极度匮乏。"""

    energy: float = 1.0
    stimulation: float = 0.5
    social: float = 0.5
    competence: float = 0.5
    safety: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, clip01(getattr(self, name)))

    def to_dict(self) -> dict[str, float]:
        return {k: r6(getattr(self, k)) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Drives:
        return cls(**{k: float(d.get(k, 0.5)) for k in cls.__dataclass_fields__})

    def deficit(self) -> dict[str, float]:
        """缺口 = 1 - 当前值，打分函数直接用。"""
        return {k: 1.0 - getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class Intent:
    """结构化意图（docs/06 三分法：意图/话语/词条）。**没有任何自由文本字段**。"""

    goal: str = ""
    speech_act: str | None = None
    referent_id: str | None = None
    predicate_type: str | None = None
    urgency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.goal, "urgency": r6(self.urgency)}
        for key in ("speech_act", "referent_id", "predicate_type"):
            v = _opt(getattr(self, key))
            if v is not None:
                d[key] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Intent | None:
        if not d:
            return None
        return cls(
            goal=str(d.get("goal", "")),
            speech_act=d.get("speech_act"),
            referent_id=d.get("referent_id"),
            predicate_type=d.get("predicate_type"),
            urgency=float(d.get("urgency", 0.0)),
        )


@dataclass(frozen=True)
class Action:
    """动作。`primitive/target/params/source` 是契约字段；

    `intent/fidelity/degraded/chain/candidates` 是审计字段——保证「为什么选了它」可追。
    """

    primitive: str
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "heuristic"
    intent: Intent | None = None
    fidelity: float = 1.0
    degraded: bool = False
    chain: tuple[str, ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "primitive": self.primitive,
            "target": self.target,
            "params": {k: (r6(v) if isinstance(v, float) else v) for k, v in self.params.items()},
            "source": self.source,
            "fidelity": r6(self.fidelity),
        }
        if self.intent is not None:
            d["intent"] = self.intent.to_dict()
        if self.degraded:
            d["degraded"] = True
            d["chain"] = list(self.chain)
        if self.candidates:
            d["candidates"] = [dict(c) for c in self.candidates]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Action:
        return cls(
            primitive=str(d["primitive"]),
            target=d.get("target"),
            params=dict(d.get("params") or {}),
            source=str(d.get("source", "heuristic")),
            intent=Intent.from_dict(d.get("intent")),
            fidelity=float(d.get("fidelity", 1.0)),
            degraded=bool(d.get("degraded", False)),
            chain=tuple(d.get("chain") or ()),
            candidates=tuple(d.get("candidates") or ()),
        )


@dataclass(frozen=True)
class Outcome:
    """执行结果。`events` 是人类可读的结果串（如 "caregiver_approached"）。"""

    success: bool = True
    events: tuple[str, ...] = ()
    reward: float = 0.0
    unexpected: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward", clip11(self.reward))
        object.__setattr__(self, "unexpected", clip01(self.unexpected))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "events": list(self.events),
            "reward": r6(self.reward),
            "unexpected": r6(self.unexpected),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Outcome:
        return cls(
            success=bool(d.get("success", True)),
            events=tuple(d.get("events") or ()),
            reward=float(d.get("reward", 0.0)),
            unexpected=float(d.get("unexpected", 0.0)),
        )


@dataclass(frozen=True)
class Language:
    """话语。`mode` 是阶段性的表达形态；`utterance` 必须由编译器产出，不允许模型直写。"""

    mode: str = "pre_verbal"
    utterance: str | None = None
    syntax_level: str = "L0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "utterance": self.utterance,
            "syntax_level": self.syntax_level,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Language:
        return cls(
            mode=str(d.get("mode", "pre_verbal")),
            utterance=d.get("utterance"),
            syntax_level=str(d.get("syntax_level", "L0")),
        )


@dataclass(frozen=True)
class LLMDebug:
    """LLM 调用痕迹。`degraded=True` 时必须带原因（不变量 I8）。"""

    model: str = "none"
    tokens_in: int = 0
    tokens_out: int = 0
    degraded: bool = False
    degraded_reason: str | None = None

    def __post_init__(self) -> None:
        if self.degraded and not self.degraded_reason:
            raise ValueError("degraded=True 必须提供 degraded_reason（不变量 I8）")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "tokens_in": int(self.tokens_in),
            "tokens_out": int(self.tokens_out),
            "degraded": bool(self.degraded),
        }
        if self.degraded_reason:
            d["degraded_reason"] = self.degraded_reason
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMDebug:
        return cls(
            model=str(d.get("model", "none")),
            tokens_in=int(d.get("tokens_in", 0)),
            tokens_out=int(d.get("tokens_out", 0)),
            degraded=bool(d.get("degraded", False)),
            degraded_reason=d.get("degraded_reason"),
        )


@dataclass(frozen=True)
class Episode:
    """一天里的一次「醒着的片刻」。整个系统的唯一事实现场。"""

    id: str
    ts: str
    age_days: int
    stage: str
    kind: str
    context: dict[str, Any]
    observation: dict[str, Any]
    appraisal: Appraisal
    drives: Drives
    action: Action
    outcome: Outcome
    language: Language
    memory_refs: tuple[str, ...] = ()
    llm: LLMDebug = field(default_factory=LLMDebug)
    salience: float = 0.0
    catch_up: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "ts": self.ts,
            "age_days": int(self.age_days),
            "stage": self.stage,
            "kind": self.kind,
            "context": dict(self.context),
            "observation": dict(self.observation),
            "appraisal": self.appraisal.to_dict(),
            "drives": self.drives.to_dict(),
            "action": self.action.to_dict(),
            "outcome": self.outcome.to_dict(),
            "language": self.language.to_dict(),
            "memory_refs": list(self.memory_refs),
            "llm": self.llm.to_dict(),
            "salience": r6(self.salience),
            "catch_up": bool(self.catch_up),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        return cls(
            id=str(d["id"]),
            ts=str(d["ts"]),
            age_days=int(d["age_days"]),
            stage=str(d["stage"]),
            kind=str(d["kind"]),
            context=dict(d.get("context") or {}),
            observation=dict(d.get("observation") or {}),
            appraisal=Appraisal.from_dict(d.get("appraisal") or {}),
            drives=Drives.from_dict(d.get("drives") or {}),
            action=Action.from_dict(d["action"]),
            outcome=Outcome.from_dict(d.get("outcome") or {}),
            language=Language.from_dict(d.get("language") or {}),
            memory_refs=tuple(d.get("memory_refs") or ()),
            llm=LLMDebug.from_dict(d.get("llm") or {}),
            salience=float(d.get("salience", 0.0)),
            catch_up=bool(d.get("catch_up", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> Episode:
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        """一行摘要，进 FTS。"""
        say = self.language.utterance or "—"
        tgt = self.action.target or "—"
        return (
            f"d{self.age_days} {self.stage} {self.kind} "
            f"{self.action.primitive}->{tgt} say={say} "
            f"v={self.appraisal.valence:+.2f}"
        )


def canonical_json(obj: Any) -> str:
    """全系统唯一的序列化函数：键排序、紧凑分隔符、UTF-8 原样。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 契约级校验（verify 与测试共用）
# ---------------------------------------------------------------------------

_EVENT_LOG_TYPES = frozenset(
    {
        "person.born",
        "episode.created",
        "consolidation.step",
        "trait.updated",
        "memory.written",
        "memory.superseded",
        "memory.archived",
        "relationship.updated",
        "relationship.attachment.classified",
        "milestone.evaluated",
        "stage.advanced",
        "developmental_delay",
        "narrative.revised",
        "identity.crisis",
        "safety.blocked",
        "safety.rewritten",
        "degraded",
        "system.sleep",
        "system.wake",
    }
)


def validate_episode(d: dict[str, Any], *, max_per_day: int = 60) -> list[str]:
    """返回问题列表（空 = 通过）。只做**契约层**检查，阶段能力检查在 stages 里做。"""
    p: list[str] = []
    if d.get("schema_version") != SCHEMA_VERSION:
        p.append(f"schema_version 应为 {SCHEMA_VERSION}，得到 {d.get('schema_version')!r}")
    eid = str(d.get("id", ""))
    if not (
        len(eid) == 16
        and eid.startswith("ep-")
        and eid[3:11].isdigit()
        and eid[11] == "-"
        and eid[12:].isdigit()
    ):
        p.append(f"id 形状应为 ep-YYYYMMDD-NNNN，得到 {eid!r}")
    if d.get("stage") not in STAGES:
        p.append(f"stage 非法：{d.get('stage')!r}")
    ts = str(d.get("ts", ""))
    if len(ts) < 20 or ("+" not in ts[10:] and "Z" not in ts):
        p.append(f"ts 必须是带时区的 ISO8601：{ts!r}")
    if not isinstance(d.get("age_days"), int) or d.get("age_days", -1) < 0:
        p.append("age_days 必须是非负整数")
    sal = d.get("salience")
    if not isinstance(sal, (int, float)) or not (0.0 <= float(sal) <= 1.0):
        p.append("salience 必须在 [0,1]")

    act = d.get("action") or {}
    if not act.get("primitive"):
        p.append("action.primitive 不能为空")
    if act.get("source") not in ACTION_SOURCES:
        p.append(f"action.source 非法：{act.get('source')!r}")
    if not isinstance(act.get("params"), dict):
        p.append("action.params 必须是对象")

    for key, lo, hi in (
        ("valence", -1.0, 1.0),
        ("arousal", 0.0, 1.0),
        ("threat", -1.0, 1.0),
        ("control", -1.0, 1.0),
        ("pe", -1.0, 1.0),
        ("novelty", 0.0, 1.0),
    ):
        v = (d.get("appraisal") or {}).get(key)
        if not isinstance(v, (int, float)) or not (lo - 1e-9 <= float(v) <= hi + 1e-9):
            p.append(f"appraisal.{key} 必须在 [{lo},{hi}]，得到 {v!r}")
    for key in ("energy", "stimulation", "social", "competence", "safety"):
        v = (d.get("drives") or {}).get(key)
        if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            p.append(f"drives.{key} 必须在 [0,1]，得到 {v!r}")

    lang = d.get("language") or {}
    if lang.get("syntax_level") not in SYNTAX_LEVELS:
        p.append(f"language.syntax_level 非法：{lang.get('syntax_level')!r}")
    utt = lang.get("utterance")
    if utt is not None and not isinstance(utt, str):
        p.append("language.utterance 必须是字符串或 null")

    llm = d.get("llm") or {}
    if llm.get("degraded") and not llm.get("degraded_reason"):
        p.append("I8：llm.degraded=true 但缺少 degraded_reason")
    if d.get("catch_up") and "context" not in d:
        p.append("catch_up 必须带 context（用于投影降权 I10）")
    return p
