"""阶段判定：什么时候可以进入下一阶段。

规则（docs/03）：`can_advance(s) = 全部里程碑通过 and 在本阶段驻留 ≥ min_dwell`。
三条禁令：不降级、不跳级、补课要留痕（`developmental_delay` 事件）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from persona.stages.milestones import Evaluation, all_passed
from persona.types import STAGES


def next_stage(stage: str) -> str | None:
    if stage not in STAGES:
        return None
    i = STAGES.index(stage)
    return STAGES[i + 1] if i + 1 < len(STAGES) else None


@dataclass
class AdvanceDecision:
    advance: bool
    current: str
    target: str | None
    reason: str
    evaluations: list[Evaluation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "advance": self.advance,
            "current": self.current,
            "target": self.target,
            "reason": self.reason,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }


def can_advance(
    stage: str,
    evaluations: Sequence[Evaluation],
    *,
    days_in_stage: int,
    min_dwell_days: int,
) -> AdvanceDecision:
    target = next_stage(stage)
    if target is None:
        return AdvanceDecision(False, stage, None, "已是最后一个阶段", list(evaluations))
    if days_in_stage < min_dwell_days:
        return AdvanceDecision(
            False, stage, target, f"驻留不足：{days_in_stage}/{min_dwell_days} 天", list(evaluations)
        )
    if not evaluations:
        # 一个还没有任何判据的阶段，不能因为「没有证据能推翻」就放行。
        # docs/03 给 S2–S6 都定义了里程碑，只是还没实现判定器；这里必须挡住。
        return AdvanceDecision(False, stage, target, "这一阶段还没有判定器（不得空手升级）", [])
    if not all_passed(evaluations):
        pending = [e.milestone_id for e in evaluations if e.status != "passed"]
        return AdvanceDecision(False, stage, target, "未通过里程碑：" + "、".join(pending), list(evaluations))
    return AdvanceDecision(True, stage, target, "全部里程碑通过且驻留满足", list(evaluations))
