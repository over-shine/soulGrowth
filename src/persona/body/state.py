"""身体状态：能量、疲劳、饥饿、体温、完整度、唤醒。

数值取自 docs/08「身体状态」一节。这里刻意**不做**主观痛觉：只有 `nociception` 这个
0–1 的量，它影响评价与动作选择，但不会被翻译成「我疼」——那是叙事层的活，而且需要证据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from persona.types import Drives, clip01


@dataclass
class BodyState:
    energy: float = 72.0  # 0–100
    fatigue: float = 0.15  # 0–1
    hunger: float = 0.30  # 0–1
    temperature: float = 0.62  # 0 冷 1 热
    integrity: float = 1.0  # 0–1，受伤下降
    arousal: float = 0.30  # 0–1，90 秒半衰
    # 速率（来自配置，便于做实验时整体缩放）
    awake_energy_per_hour: float = -3.7
    sleep_energy_per_hour: float = 6.5
    hunger_per_hour: float = 0.06

    # -- 推进 ---------------------------------------------------------------
    def advance(self, hours: float, *, awake: bool = True) -> None:
        rate = self.awake_energy_per_hour if awake else self.sleep_energy_per_hour
        self.energy = max(0.0, min(100.0, self.energy + rate * hours))
        self.hunger = clip01(self.hunger + self.hunger_per_hour * hours)
        if awake:
            self.temperature = max(0.0, min(1.0, self.temperature - 0.004 * hours))
            self.fatigue = clip01(self.fatigue + 0.03 * hours)

    def note_episode(self, delta: float = 0.01) -> None:
        self.fatigue = clip01(self.fatigue + delta)

    def decay_arousal(self, seconds: float) -> None:
        """90 秒半衰：a ← a · 0.5^(t/90)。"""
        self.arousal = clip01(self.arousal * (0.5 ** (max(0.0, seconds) / 90.0)))

    def sleep(self, minutes: float) -> None:
        hours = minutes / 60.0
        self.advance(hours, awake=False)
        self.fatigue = clip01(self.fatigue - 0.55 * hours)

    def eat(self, amount: float = 0.4) -> None:
        self.hunger = clip01(self.hunger - amount)
        self.energy = min(100.0, self.energy + 6.0)

    def be_held(self) -> None:
        self.temperature = clip01(self.temperature + 0.03)
        self.arousal = clip01(self.arousal * 0.85)

    def hurt(self, amount: float = 0.05) -> None:
        self.integrity = clip01(self.integrity - amount)

    # -- 派生量 -------------------------------------------------------------
    @property
    def tired(self) -> bool:
        return self.fatigue >= 0.8 or self.energy <= 18.0

    def nociception(self, threat: float) -> float:
        return clip01(0.7 * (1.0 - self.integrity) + 0.3 * max(0.0, threat))

    def core_drives(self) -> Drives:
        """身体能自己回答的两个缺口；社交/胜任由 cognition/drives.py 补齐。"""
        return Drives(
            energy=clip01(self.energy / 100.0),
            stimulation=clip01(0.5),
            social=clip01(0.5),
            competence=clip01(0.5),
            safety=clip01(self.integrity * (1.0 - 0.5 * (1.0 - self.energy / 100.0))),
        )

    def to_dict(self, *, precise: bool = False) -> dict[str, Any]:
        def r6(x: float) -> float:
            # 存档收成 6 位；交给远端身体时必须原样传（舍入会让两具身体看到不同的内感受）。
            return float(x) if precise else round(float(x), 6)

        return {
            "energy": r6(self.energy),
            "fatigue": r6(self.fatigue),
            "hunger": r6(self.hunger),
            "temperature": r6(self.temperature),
            "integrity": r6(self.integrity),
            "arousal": r6(self.arousal),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BodyState:
        return cls(
            energy=float(d.get("energy", 72.0)),
            fatigue=float(d.get("fatigue", 0.15)),
            hunger=float(d.get("hunger", 0.30)),
            temperature=float(d.get("temperature", 0.62)),
            integrity=float(d.get("integrity", 1.0)),
            arousal=float(d.get("arousal", 0.30)),
        )
