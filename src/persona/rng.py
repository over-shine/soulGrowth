"""唯一随机源。

两条硬规则（docs/16 §5）：
* 业务代码**不得** import `random` 的模块级函数，只能用 `derive()` 返回的 `Random` 实例；
* 种子只由「谁 + 哪一天 + 哪个时间片 + 用途」决定，绝不掺入系统时间。

这样重跑任何一天都能逐字节复现。
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

MASK64 = (1 << 64) - 1


def seed_for(*parts: object) -> int:
    """把任意标识拼成 64 位种子。用 blake2b 而不是内置 hash()——后者随进程随机化。"""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def derive(*parts: object) -> random.Random:
    """派生一个独立的随机流。调用点越多，流越不会互相干扰。"""
    return random.Random(seed_for(*parts))


def bernoulli(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def uniform(rng: random.Random, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    return lo + (hi - lo) * rng.random()


def truncated_gauss(
    rng: random.Random, mu: float, sigma: float, lo: float = -1.0, hi: float = 1.0
) -> float:
    """截断正态：超界就重抽（sigma 远小于区间，最多几次就收敛）。

    用重抽而不是 clip，是为了不让边界堆积过多样本——气质分布本身要好看。
    """
    for _ in range(32):
        v = rng.gauss(mu, sigma)
        if lo <= v <= hi:
            return v
    return max(lo, min(hi, v))


def pick_weighted(rng: random.Random, items: Iterable[tuple[str, float]]) -> str:
    """按权重抽一个 id。权重需非负；全 0 时退化为等概率。"""
    pool = [(k, max(0.0, float(w))) for k, w in items]
    total = sum(w for _, w in pool)
    if total <= 0:
        return pool[rng.randrange(len(pool))][0]
    x = rng.random() * total
    acc = 0.0
    for k, w in pool:
        acc += w
        if x <= acc:
            return k
    return pool[-1][0]
