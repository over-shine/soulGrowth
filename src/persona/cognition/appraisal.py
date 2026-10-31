"""评价：把「我感知到什么」变成「这对我意味着什么」。

appraisal 是行为的枢纽——特质只在这里通过**参数**进入（threat 项），
其它通道（动作打分、执行阈值）在 policy 里。这样同一个 T 的不同投影
可以分别做消融（docs/04 三通道）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from persona.body.perception import Bundle
from persona.body.state import BodyState
from persona.cognition.traits import Bias
from persona.types import Appraisal, clip01, clip11, r6

#: threat 的评价权重（docs/04 通道 1）
A0 = -0.9      # 基线：陌生世界的默认警惕
A1 = 1.15      # 特质 N/NA 的增益
A3 = 0.85      # 历史威胁率
A4 = 1.05      # 安全线索的压制
A5 = 1.20      # 陌生面孔：S1 里最强的威胁线索（陌生人焦虑的入口）


@dataclass(frozen=True)
class History:
    """最近若干 episode 的行为统计。只做计数，不做模型自述（docs/10）。"""

    n: int = 0
    threat_rate: float = 0.0
    success_rate: float = 0.0

    @classmethod
    def from_episodes(cls, episodes: list[dict]) -> "History":
        n = len(episodes)
        if not n:
            return cls()
        threats = sum(1 for e in episodes if float(e.get("appraisal", {}).get("threat", 0.0)) >= 0.4)
        wins = sum(1 for e in episodes if e.get("outcome", {}).get("success"))
        return cls(n=n, threat_rate=threats / n, success_rate=wins / n)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safety_cue(bundle: Bundle, world_caregiver: bool) -> float:
    """安全线索：照料者在场 + 环境不吵 + 有熟悉的柔软物。"""
    cue = 0.0
    if world_caregiver:
        cue += 0.6
    if bundle.ambient_db <= 45.0:
        cue += 0.2
    if any(
        str(s.channels.get("TAC", {}).get("texture", "")) in ("soft", "warm_soft")
        for s in bundle.stimuli
    ):
        cue += 0.2
    return clip01(cue)


def stranger_signal(bundle: Bundle) -> float:
    """视野里有没有一张不认识的脸（S1 起面孔通道才有这个读数）。"""
    for s in bundle.stimuli:
        fac = s.channels.get("FAC")
        if fac and fac.get("present") and not fac.get("familiar", False):
            return 1.0
    return 0.0


def appraise(
    bundle: Bundle,
    body: BodyState,
    *,
    bias: Bias,
    history: History,
    caregiver_present: bool,
    expectation: float = 0.0,
    unexpected: float = 0.0,
) -> Appraisal:
    """产出六维评价。全部由可复算的量算出，没有一处让模型「感觉一下」。"""
    cue = safety_cue(bundle, caregiver_present)
    stranger = stranger_signal(bundle)
    logit = A0 + A1 * bias.threat_gain + A3 * history.threat_rate + A5 * stranger - A4 * cue
    threat = clip01(_sigmoid(logit))

    novelty = clip01(_mean_novelty(bundle))
    control = clip01(
        0.45 * (1.0 - body.fatigue)
        + 0.25 * bias.persistence
        + 0.20 * (1.0 - threat)
        + 0.10 * (1.0 if bundle.interoception.get("energy", 0.0) >= 0.4 else 0.0)
    )
    # 效价：安全 − 威胁 + 一点点新奇（新奇本身是中性偏正，取决于气质）
    valence = clip11(0.55 * cue - 0.75 * threat + 0.25 * (novelty - 0.5) + 0.2 * (control - 0.5))
    arousal = clip01(
        0.30 * body.arousal
        + 0.35 * novelty
        + 0.35 * threat
        + 0.10 * body.nociception(threat)
        + 0.05 * unexpected
    )
    sigma = 0.25 + 0.35 * novelty + 0.3 * threat
    pe = clip11(abs(valence - expectation) / max(sigma, 0.05))
    return Appraisal(
        valence=r6(valence),
        arousal=r6(arousal),
        threat=r6(threat),
        control=r6(control),
        pe=r6(pe),
        novelty=r6(novelty),
    )


def _mean_novelty(bundle: Bundle) -> float:
    if not bundle.stimuli:
        return 0.0
    # 注意：这里要的是**新异度**的均值，不是显著度的均值。
    # 显著度几乎不随习惯化变化（0.74 和 0.45 都是「很显眼」），
    # 用它当新异度，去习惯化就永远激不起唤醒变化（M0.4 一直是 0.02）。
    return sum(s.novelty for s in bundle.stimuli) / len(bundle.stimuli)
