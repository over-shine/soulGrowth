"""把一条 episode 还原成可读的因果链。

同一份逻辑给两个出口用：命令行 ``persona why``（文本）与观察台的
``/api/episode``（JSON）。它们不该各自解释一遍「为什么动了一下」——那会变成
两套真相。
"""

from __future__ import annotations

from typing import Any


def trace_episode(ep: dict[str, Any], unlocked: set[str] | None = None) -> dict[str, Any]:
    appr = ep.get("appraisal") or {}
    act = ep.get("action") or {}
    out = ep.get("outcome") or {}
    obs = ep.get("observation") or {}
    primitive = act.get("primitive")
    return {
        "id": ep.get("id"),
        "ts": ep.get("ts"),
        "stage": ep.get("stage"),
        "age_days": ep.get("age_days"),
        "kind": ep.get("kind"),
        "salience": ep.get("salience"),
        "catch_up": bool(ep.get("catch_up")),
        "context": ep.get("context"),
        "1_看见了什么": {
            "summary": obs.get("summary"),
            "salient": obs.get("salient"),
            "modality": obs.get("modality"),
            "motion_onsets": obs.get("motion_onsets"),
        },
        "2_评价": {k: appr.get(k) for k in ("valence", "arousal", "threat", "control", "pe", "novelty")},
        "3_驱力": ep.get("drives"),
        "4_候选": list(act.get("candidates") or []),
        "5_选择": {
            "primitive": primitive,
            "target": act.get("target"),
            "params": act.get("params"),
            "source": act.get("source"),
            "fidelity": act.get("fidelity"),
            "degraded": act.get("degraded"),
            "chain": act.get("chain"),
        },
        "6_结果": out,
        "7_话语": ep.get("language"),
        "8_记忆": ep.get("memory_refs"),
        "9_llm": ep.get("llm"),
        "不变量": {
            "动作已解锁": None if unlocked is None else primitive in unlocked,
        },
    }
