"""提示词。

硬约束都写在这里，且写死了「不许」：
* 不提供新能力（能力表在代码里，模型无权增删）；
* 不产出词语（话语由 `language/compiler` 生成，模型只给意图与先验）；
* 只允许 JSON，字段固定（`additionalProperties` 语义靠解析器强制）。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

PRIOR_SYSTEM = """你是一个发育中个体的「直觉先验」估计器，不是决策者。

你的唯一任务是：看到当前感知与候选动作列表，给出每个候选动作的**相对倾向权重**。

硬性规则（违反即输出无效）：
1. 只能使用候选列表中出现的 primitive id，不得新增、改写、类比。
2. 不得输出任何词语、句子、内心独白或建议文本。
3. 不得假设个体具备列表中未出现的能力（例如 S0 不会说话、不会走）。
4. 若信息不足，给接近均匀的权重，而不是编造理由。
5. 只输出 JSON：{"priors": {"<primitive_id>": <0-1 之间的数>, ...}}
"""


def build_prior_user(*, observation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "observation": dict(observation),
        "candidates": [
            {"primitive": c.get("primitive"), "params": c.get("params", {})} for c in candidates
        ],
    }
    return (
        "当前感知与候选动作如下。请给出先验权重，只返回 JSON。\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
