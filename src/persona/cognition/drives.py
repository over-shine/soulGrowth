"""驱力：身体与情境的「缺口」。

驱力不是情绪，是**待满足的量**。它只回答「现在最缺什么」，
不做归因、不命名感受（那是叙事层的事，且在 S3+ 才出现）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from persona.body.perception import Bundle
from persona.body.state import BodyState
from persona.types import Drives, clip01, r6


@dataclass(frozen=True)
class DriveContext:
    """算驱力需要的外界事实（都来自可复算的来源）。"""

    caregiver_present: bool = False
    minutes_since_care: float = 0.0
    recent_novelty: float = 0.0   # 最近一小时平均新奇度
    recent_success: float = 0.0   # 最近若干 episode 成功率
    threat: float = 0.0
    care_n: float = 0.0           # 累计被照料的次数（熟悉度来源）


def compute(body: BodyState, bundle: Bundle, ctx: DriveContext) -> Drives:
    """五个驱力，全部在 0–1，越大越缺。"""
    # 「能量缺口」同时计入饥饿：血糖低与睡不够一样，都是要满足的生理需求
    energy = clip01(1.0 - (0.55 * body.energy / 100.0 + 0.45 * (1.0 - body.hunger)))
    # 刺激：新奇度越低越缺；已经看过的东西不算刺激
    stimulation = clip01(1.0 - ctx.recent_novelty)
    # 社交：照料者不在或久未回应 → 缺。熟悉度决定「不在」有多难受：
    # 认得出她的人才会因为分开而不安（这是 S0 唯一能观测到的「认照料者」）。
    # 熟悉度按累计照料次数缓慢饱和（care_n/300）：一个月里它一直在长，
    # 从约 0.1 长到约 0.6，于是「分开难受」有真正的生长空间，而不是几天就顶满。
    familiarity = 1.0 - math.exp(-max(0.0, ctx.care_n) / 300.0)
    social = clip01(
        0.30
        + 0.5 * ctx.minutes_since_care / 120.0
        - (0.45 if ctx.caregiver_present else 0.0)
        + (0.90 * familiarity if not ctx.caregiver_present else 0.0)
    )
    # 胜任：连续失败 → 缺
    competence = clip01(0.5 - 0.5 * ctx.recent_success)
    # 安全：威胁 + 身体损伤
    safety = clip01(0.6 * ctx.threat + 0.4 * body.nociception(ctx.threat))
    return Drives(
        energy=r6(energy),
        stimulation=r6(stimulation),
        social=r6(social),
        competence=r6(competence),
        safety=r6(safety),
    )


def from_mapping(d: Mapping[str, float]) -> Drives:
    """从已落盘的 episode.drives 还原（用于回放与周结算）。"""
    return Drives(
        energy=float(d.get("energy", 0.0)),
        stimulation=float(d.get("stimulation", 0.0)),
        social=float(d.get("social", 0.0)),
        competence=float(d.get("competence", 0.0)),
        safety=float(d.get("safety", 0.0)),
    )
