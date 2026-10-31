"""确定性的假模型。

它的存在不是为了「假装有智能」，而是为了让**整条流水线能在没有 key 的情况下
端到端跑起来并被测试**。因此：

* 输出只依赖输入哈希，同输入必然同输出；
* 永远 `degraded=true`，并在 episode 里如实记下原因（I8）；
* 它给出的先验接近均匀——不能让它偷偷变成「人格」或「偏好」。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from persona.cognition.policy import Prior
from persona.llm.provider import LLMProvider
from persona.rng import derive


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, *, model: str = "mock-1", seed: str = "soul") -> None:
        self.model = model
        self.seed = seed

    def prior(self, *, observation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Prior:
        prims = [str(c.get("primitive", "")) for c in candidates]
        salient = ",".join(str(s) for s in observation.get("salient", []) or [])
        key = f"{salient}|{'/'.join(prims)}"
        rng = derive("mock-prior", self.seed, key)
        raw = {p: 1.0 + 0.25 * math.sin(rng.random() * math.pi * 2) for p in prims}
        total = sum(raw.values()) or 1.0
        return Prior(
            probs={k: v / total for k, v in raw.items()},
            model=self.model,
            degraded=True,
            degraded_reason="mock provider：无模型参与，先验为确定性伪分布",
        )
