"""人格：气质、特质、行为投影与漂移。

三条设计红线（docs/04）：
1. 特质只通过**参数**影响行为——`bias()` 每 episode 由当前 T 现算，不缓存、不写提示词（I11）；
2. 观测用行为计数，不用模型自述；
3. 特质每周结算一次，缺测就重归一权重，两周全缺则 B 取 T（不更新）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from persona.rng import derive, truncated_gauss
from persona.types import clip11, r6, STAGES

OCEAN = ("O", "C", "E", "A", "N")

#: 气质四维：S 寻求新奇 / NA 负性情绪 / EC 努力控制 / AF 亲和（docs/04）
@dataclass(frozen=True)
class Temperament:
    s: float
    na: float
    ec: float
    af: float

    def to_dict(self) -> dict[str, float]:
        return {"S": r6(self.s), "NA": r6(self.na), "EC": r6(self.ec), "AF": r6(self.af)}


def sample_temperament(individual_id: str, birth_iso: str) -> Temperament:
    """出生时采样一次，之后终身不变（docs/04：seed = hash(个体ID + 出生时间)）。"""
    rng = derive("temperament", individual_id, birth_iso)
    return Temperament(
        s=truncated_gauss(rng, 0.0, 0.6, -1.0, 1.0),
        na=truncated_gauss(rng, 0.0, 0.6, -1.0, 1.0),
        ec=truncated_gauss(rng, 0.0, 0.6, -1.0, 1.0),
        af=truncated_gauss(rng, 0.0, 0.6, -1.0, 1.0),
    )


def t0_from_temperament(t: Temperament, *, individual_id: str = "", birth_iso: str = "") -> dict[str, float]:
    """气质 → 初始特质。ε 用确定性 RNG，保证同一人重跑结果一致。"""
    rng = derive("t0", individual_id, birth_iso)
    eps = {k: truncated_gauss(rng, 0.0, 0.05, -0.2, 0.2) for k in OCEAN}
    return {
        "O": clip11(0.6 * t.s + 0.2 * t.ec + eps["O"]),
        "C": clip11(0.8 * t.ec - 0.2 * t.na + eps["C"]),
        "E": clip11(0.7 * t.af + 0.3 * t.s + eps["E"]),
        "A": clip11(0.7 * t.af - 0.3 * t.na + eps["A"]),
        "N": clip11(0.8 * t.na - 0.2 * t.ec + eps["N"]),
    }


#: 行为指标定义：名称 → (特质维度, μ0, σ0, 权重)。顺序即 docs/04 的 15 项。
INDICATORS: dict[str, tuple[str, float, float, float]] = {
    "novelty_contact_rate": ("O", 0.30, 0.15, 1.0),
    "activity_variety": ("O", 0.45, 0.20, 0.8),
    "cross_category_try": ("O", 0.20, 0.15, 0.6),
    "plan_completion": ("C", 0.45, 0.25, 1.2),
    "persistence_s": ("C", 4.00, 2.00, 1.0),
    "order_preference": ("C", 0.15, 0.10, 0.6),
    "social_initiation": ("E", 0.30, 0.20, 1.2),
    "vocalization": ("E", 0.10, 0.08, 1.0),
    "activity_intensity": ("E", 0.35, 0.20, 0.8),
    "prosocial_ratio": ("A", 0.25, 0.15, 1.2),
    "yielding_rate": ("A", 0.40, 0.20, 1.0),
    "soothing_freq": ("A", 0.20, 0.15, 0.8),
    "threat_appraisal_rate": ("N", 0.20, 0.12, 1.2),
    "frustration_recovery_s": ("N", 8.00, 4.00, 1.0),
    "comfort_seeking": ("N", 0.30, 0.20, 0.8),
}

#: 漂移速率与回归强度（docs/04）
ETA: dict[str, float] = {"S0": 0.12, "S1": 0.10, "S2": 0.09, "S3": 0.06, "S4": 0.04, "S5": 0.02, "S6": 0.01}
LAMBDA: dict[str, float] = {"S0": 0.0, "S1": 0.0, "S2": 0.020, "S3": 0.025, "S4": 0.020, "S5": 0.015, "S6": 0.010}

#: 补做的格子（catch_up）在投影里算多重的经历（I10：≤0.5）
CATCH_UP_WEIGHT: float = 0.5

#: 阶段目标（年龄段的社会期望），S2 起线性插值到 S6
MU_TARGET: dict[str, tuple[float, float]] = {
    "O": (0.0, 0.0),
    "C": (0.15, 0.30),
    "E": (0.0, 0.0),
    "A": (0.20, 0.40),
    "N": (-0.15, -0.30),
}


def mu_for(stage: str) -> dict[str, float]:
    """当前阶段的特质目标值。S0–S1 用起点，S2–S6 线性插值。"""
    idx = STAGES.index(stage) if stage in STAGES else 0
    span = max(1, STAGES.index("S6") - STAGES.index("S2"))
    w = 0.0 if idx <= STAGES.index("S1") else min(1.0, (idx - STAGES.index("S1")) / span)
    return {k: lo + (hi - lo) * w for k, (lo, hi) in MU_TARGET.items()}


def episode_weight(ep: Mapping[str, Any]) -> float:
    """这条经历在投影里算多重。

    补做的格子（catch_up）权重减半（I10）：停机期间「没被经历过」的时间不该和
    真的活着的时间一样重——否则一次断电重启就能把一周的画像拽过去。
    """
    return CATCH_UP_WEIGHT if ep.get("catch_up") else 1.0


def project_indicators(
    episodes: Sequence[Mapping[str, Any]], *, weights: Sequence[float] | None = None
) -> dict[str, float]:
    """从一天的 episode 序列算行为指标。缺测的不出现，由调用方重归一。

    所有比率都是**加权**比率：分母是权重之和，不是条数之和。这样「补做的一天」
    说话就轻一些，而不会被当成同样多的一天的经历。
    """
    n = len(episodes)
    if not n:
        return {}
    w = list(weights) if weights is not None else [episode_weight(e) for e in episodes]
    total = sum(w)
    if total <= 0:
        return {}
    prims = [str(e.get("action", {}).get("primitive", "")) for e in episodes]

    def rate(flags: Sequence[bool]) -> float:
        return sum(wi for wi, ok in zip(w, flags) if ok) / total

    return {
        "novelty_contact_rate": rate(
            [p in ("A0.FIXATE", "A1.REACH", "A1.CRAWL", "A2.WALK", "A2.MANIPULATE", "A3.ASK") for p in prims]
        ),
        "activity_variety": len(set(prims)) / max(1, len(prims)),
        "plan_completion": rate([bool(e.get("outcome", {}).get("success")) for e in episodes]),
        "social_initiation": rate(
            [bool(str(e.get("context", {}).get("others") or "")) for e in episodes]
        ),
        "vocalization": rate(
            [bool(e.get("language", {}).get("utterance")) for e in episodes]
        ),
        "activity_intensity": rate(
            [p in ("A0.FIXATE", "A1.REACH", "A1.CRAWL", "A2.WALK", "A2.MANIPULATE", "A3.ASK") for p in prims]
        ),
        "threat_appraisal_rate": rate(
            [float(e.get("appraisal", {}).get("threat", 0.0)) >= 0.4 for e in episodes]
        ),
        "comfort_seeking": rate([p == "A0.CRY" for p in prims]),
    }


def project_traits(episodes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """加权投影：b=tanh((x−μ0)/σ0)，B_j=Σw·b/Σw。返回 (B, indicators, missing)。"""
    ind = project_indicators(episodes)
    num = {k: 0.0 for k in OCEAN}
    den = {k: 0.0 for k in OCEAN}
    for name, value in ind.items():
        dim, mu0, sigma0, w = INDICATORS[name]
        b = math.tanh((value - mu0) / sigma0 if sigma0 else 0.0)
        num[dim] += w * b
        den[dim] += w
    b: dict[str, float] = {}
    for k in OCEAN:
        b[k] = clip11(num[k] / den[k]) if den[k] > 0 else 0.0
    missing = [k for k in OCEAN if den[k] == 0]
    return b, ind, missing


@dataclass
class Traits:
    """一周一条的特质记录（docs/13 traits 表）。"""

    week: str
    stage: str
    T: dict[str, float]
    B: dict[str, float] = field(default_factory=dict)
    eta: float = 0.0
    xi: float = 0.0
    indicators: dict[str, float] = field(default_factory=dict)
    n_episodes: int = 0
    missing_indicators: list[str] = field(default_factory=list)
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "stage": self.stage,
            "T": {k: r6(v) for k, v in self.T.items()},
            "B": {k: r6(v) for k, v in self.B.items()},
            "eta": r6(self.eta),
            "xi": r6(self.xi),
            "indicators": {k: r6(v) for k, v in self.indicators.items()},
            "n_episodes": self.n_episodes,
            "missing_indicators": list(self.missing_indicators),
            "confirmed": self.confirmed,
        }

    def t_of(self, dim: str) -> float:
        return float(self.T.get(dim, 0.0))


def update_traits(
    prev: Traits,
    episodes: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    week: str,
    rng_seed: str,
) -> Traits:
    """周结算：T_{w+1}=clip(T+η(B̄−T)+λ(μ−T)+ξ)。无数据周 T 不变（n_episodes=0）。"""
    if not episodes:
        return Traits(
            week=week,
            stage=stage,
            T={k: r6(v) for k, v in prev.T.items()},
            B={k: r6(v) for k, v in prev.T.items()},
            eta=ETA.get(stage, 0.05),
            xi=0.0,
            n_episodes=0,
            missing_indicators=list(OCEAN),
            confirmed=True,
        )
    b, ind, missing = project_traits(episodes)
    rng = derive("traits-drift", rng_seed, week, stage)
    eta, lam = ETA.get(stage, 0.05), LAMBDA.get(stage, 0.0)
    mu = mu_for(stage)
    t_new: dict[str, float] = {}
    xi_used = 0.0
    for k in OCEAN:
        xi = truncated_gauss(rng, 0.0, 0.01, -0.05, 0.05)
        xi_used = max(xi_used, abs(xi))
        t_new[k] = clip11(prev.t_of(k) + eta * (b[k] - prev.t_of(k)) + lam * (mu[k] - prev.t_of(k)) + xi)
    return Traits(
        week=week,
        stage=stage,
        T={k: r6(v) for k, v in t_new.items()},
        B={k: r6(v) for k, v in b.items()},
        eta=eta,
        xi=max(xi_used, 0.0),
        indicators={k: r6(v) for k, v in ind.items()},
        n_episodes=len(episodes),
        missing_indicators=missing,
        confirmed=True,
    )


@dataclass(frozen=True)
class Bias:
    """当前特质的行为化投影。**每 episode 现算**，绝不落库（I11）。"""

    o: float
    c: float
    e: float
    a: float
    n: float
    explore: float
    social: float
    persistence: float
    prosocial: float
    risk: float
    threat_gain: float
    safety_gain: float

    def to_dict(self) -> dict[str, float]:
        return {k: r6(v) for k, v in self.__dict__.items()}


def bias(traits: Mapping[str, float] | Traits, *, temperament: Temperament | None = None) -> Bias:
    """T → 行为参数。只做线性、有界映射，方向由 docs/04 规定。"""
    t = traits.T if isinstance(traits, Traits) else dict(traits)
    o, c, e, a, n = (float(t.get(k, 0.0)) for k in OCEAN)
    na = temperament.na if temperament else 0.0
    return Bias(
        o=r6(o),
        c=r6(c),
        e=r6(e),
        a=r6(a),
        n=r6(n),
        explore=r6(0.5 + 0.5 * o),          # 高 O → 更愿意碰新东西
        social=r6(0.5 + 0.5 * e),           # 高 E → 更主动找人
        persistence=r6(0.5 + 0.5 * c),      # 高 C → 更能坚持
        prosocial=r6(0.5 + 0.5 * a),        # 高 A → 更愿意让/安抚
        risk=r6(0.5 + 0.5 * n),             # 高 N → 更回避风险
        threat_gain=r6(0.9 * n + 0.7 * na), # 高 N/NA → 威胁评价更高
        safety_gain=r6(0.6 * c),            # 高 C → 安全线索更能压住威胁
    )
