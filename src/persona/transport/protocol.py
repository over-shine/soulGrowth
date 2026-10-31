"""core ↔ body 之间的协议。

只有一件事要保证：**同一条 episode 不管身体在进程内还是在网线那头，产出的字节
完全一致**。所以这里的序列化必须是往返无损的——不能像 ``Bundle.to_dict`` 那样只挑
给人看的字段，否则远端算出来的显著度、追踪标记会和本地不同。

消息都是单层 JSON 对象，带 ``type`` 与 ``seq``。core 发起、body 应答，一问一答。
"""

from __future__ import annotations

import json
from typing import Any

from persona.body.compiler import Candidate, Compiled
from persona.body.perception import Bundle, Stimulus
from persona.types import Intent

VERSION = 1
PROTOCOL = "persona.body/1"


class ProtocolError(RuntimeError):
    """消息形状不对，或者版本不匹配。"""


# -- 基本编码 ----------------------------------------------------------------


def encode(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode(text: str) -> dict[str, Any]:
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - 防御性
        raise ProtocolError(f"不是合法 JSON：{exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("消息必须是 JSON 对象")
    return message


def check(message: dict[str, Any], *, seq: int | None = None) -> dict[str, Any]:
    if message.get("version") not in (None, VERSION):
        raise ProtocolError(f"版本不匹配：{message.get('version')} != {VERSION}")
    if seq is not None and message.get("seq") != seq:
        raise ProtocolError(f"应答的 seq 对不上：{message.get('seq')} != {seq}")
    if message.get("type") == "error":
        raise ProtocolError(f"body 报错（{message.get('kind')}）：{message.get('message')}")
    return message


# -- 往返无损的序列化 --------------------------------------------------------


def bundle_to_wire(bundle: Bundle) -> dict[str, Any]:
    return {
        "stage": bundle.stage,
        "modality": list(bundle.modality),
        "luminance": bundle.luminance,
        "ambient_db": bundle.ambient_db,
        "interoception": dict(bundle.interoception),
        "attention_max": bundle.attention_max,
        "dropped": list(bundle.dropped),
        "truncated": bundle.truncated,
        "budget_chars": bundle.budget_chars,
        "onsets": list(bundle.onsets),
        "moving": list(bundle.moving),
        "stimuli": [
            {
                "id": s.id,
                "label": s.label,
                "kind": s.kind,
                "salience": s.salience,
                "novelty": s.novelty,
                "channels": s.channels,
            }
            for s in bundle.stimuli
        ],
    }


def bundle_from_wire(data: dict[str, Any]) -> Bundle:
    stimuli = [
        Stimulus(
            id=str(s["id"]),
            label=str(s.get("label", "")),
            kind=str(s.get("kind", "")),
            salience=float(s.get("salience", 0.0)),
            novelty=float(s.get("novelty", 0.0)),
            channels=dict(s.get("channels") or {}),
        )
        for s in data.get("stimuli") or []
    ]
    return Bundle(
        stage=str(data["stage"]),
        modality=list(data.get("modality") or []),
        stimuli=stimuli,
        luminance=float(data.get("luminance", 0.0)),
        ambient_db=float(data.get("ambient_db", 0.0)),
        interoception={k: float(v) for k, v in (data.get("interoception") or {}).items()},
        attention_max=int(data.get("attention_max", 2)),
        dropped=list(data.get("dropped") or []),
        truncated=bool(data.get("truncated", False)),
        budget_chars=int(data.get("budget_chars", 0)),
        onsets=list(data.get("onsets") or []),
        moving=list(data.get("moving") or []),
    )


def candidate_to_wire(c: Candidate) -> dict[str, Any]:
    return {
        "primitive": c.primitive,
        "params": dict(c.params),
        "afford": dict(c.afford),
        "cost": c.cost,
        "fidelity": c.fidelity,
        "note": c.note,
    }


def candidate_from_wire(data: dict[str, Any]) -> Candidate:
    return Candidate(
        primitive=str(data["primitive"]),
        params=dict(data.get("params") or {}),
        afford={k: float(v) for k, v in (data.get("afford") or {}).items()},
        cost=float(data.get("cost", 0.0)),
        fidelity=float(data.get("fidelity", 1.0)),
        note=str(data.get("note", "")),
    )


def intent_to_wire(intent: Intent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    return {
        "goal": intent.goal,
        "speech_act": intent.speech_act,
        "referent_id": intent.referent_id,
        "predicate_type": intent.predicate_type,
        "urgency": intent.urgency,
    }


def intent_from_wire(data: dict[str, Any] | None) -> Intent | None:
    if not data:
        return None
    return Intent(
        goal=str(data.get("goal", "")),
        speech_act=str(data.get("speech_act", "")),
        referent_id=data.get("referent_id"),
        predicate_type=data.get("predicate_type"),
        urgency=float(data.get("urgency", 0.0)),
    )


def compiled_to_wire(compiled: Compiled) -> dict[str, Any]:
    return {
        "primitive": compiled.primitive,
        "params": dict(compiled.params),
        "utterance": compiled.utterance,
        "mode": compiled.mode,
        "level": compiled.level,
        "fidelity": compiled.fidelity,
        "degraded": compiled.degraded,
        "chain": list(compiled.chain),
        "denials": list(compiled.denials),
        "intent": intent_to_wire(compiled.intent),
    }


def compiled_from_wire(data: dict[str, Any]) -> Compiled:
    return Compiled(
        primitive=str(data.get("primitive", "A0.NONE")),
        params=dict(data.get("params") or {}),
        intent=intent_from_wire(data.get("intent")),
        utterance=data.get("utterance"),
        mode=str(data.get("mode", "pre_verbal")),
        level=str(data.get("level", "L0")),
        fidelity=float(data.get("fidelity", 0.0)),
        degraded=bool(data.get("degraded", False)),
        chain=tuple(data.get("chain") or ()),
        denials=tuple(data.get("denials") or ()),
    )


# -- 消息构造 ----------------------------------------------------------------


def hello(role: str = "core") -> dict[str, Any]:
    return {"version": VERSION, "type": "hello", "role": role, "protocol": PROTOCOL}


def observe(
    seq: int,
    *,
    stage: str,
    day: int,
    episode_id: str,
    clock_hours: float,
    goal_focus: str | None,
    world: dict[str, Any],
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "type": "observe",
        "seq": seq,
        "stage": stage,
        "day": day,
        "episode_id": episode_id,
        "clock_hours": clock_hours,
        "goal_focus": goal_focus,
        "world": world,
        "body": body,
    }


def afford(
    seq: int,
    *,
    stage: str,
    world: dict[str, Any],
    body: dict[str, Any],
    lexicon: dict[str, Any],
    social: float,
    limit: int,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "type": "afford",
        "seq": seq,
        "stage": stage,
        "world": world,
        "body": body,
        "lexicon": lexicon,
        "social": social,
        "limit": limit,
        "bundle": bundle,
    }


def compile_action(
    seq: int,
    *,
    stage: str,
    world: dict[str, Any],
    body: dict[str, Any],
    lexicon: dict[str, Any],
    social: float,
    primitive: str,
    params: dict[str, Any],
    intent: dict[str, Any] | None,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "type": "compile",
        "seq": seq,
        "stage": stage,
        "world": world,
        "body": body,
        "lexicon": lexicon,
        "social": social,
        "primitive": primitive,
        "params": params,
        "intent": intent,
        "bundle": bundle,
    }


def error(message: str, *, kind: str = "protocol", seq: int | None = None) -> dict[str, Any]:
    return {"version": VERSION, "type": "error", "seq": seq, "kind": kind, "message": message}
