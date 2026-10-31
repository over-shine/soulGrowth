"""时钟：real / scaled / mock 三态。

业务代码只允许从这里拿时间（`datetime.now()` 只出现在本文件），否则「重跑同一天」无法复现。
时区固定 +08:00：中国无夏令时，用固定偏移可以避免 Windows 上缺 tzdata 时
`zoneinfo.ZoneInfo("Asia/Shanghai")` 直接失败的问题。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from persona.errors import ConfigError

TZ = timezone(timedelta(hours=8), "Asia/Shanghai")


class Clock:
    """时钟协议。子类只实现 `now()`。"""

    def now(self) -> datetime:  # pragma: no cover - 抽象
        raise NotImplementedError

    def iso(self) -> str:
        return self.now().isoformat(timespec="seconds")

    def today(self) -> str:
        """本地日期，用于目录/文件命名（YYYY-MM-DD）。"""
        return self.now().date().isoformat()


class RealClock(Clock):
    def now(self) -> datetime:
        return datetime.now(TZ)


@dataclass
class MockClock(Clock):
    """手动推进的时钟。跑测试与「快进 30 天」用。"""

    start: datetime = datetime(2026, 1, 1, 6, 0, 0, tzinfo=TZ)
    _t: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=TZ)
        self._t = self.start

    def now(self) -> datetime:
        return self._t

    def advance(self, seconds: float) -> datetime:
        self._t = self._t + timedelta(seconds=float(seconds))
        return self._t

    def set(self, when: datetime) -> None:
        self._t = when if when.tzinfo else when.replace(tzinfo=TZ)


@dataclass
class ScaledClock(Clock):
    """按比例放大的真实时钟：1 个真实秒 = `scale` 个模拟秒。

    `scale=2880` 即 1 真实秒 ≈ 48 模拟分钟（约 30 真实分钟走完 30 天），方便看长跑。
    """

    start: datetime = datetime(2026, 1, 1, 6, 0, 0, tzinfo=TZ)
    scale: float = 1.0
    _mono0: float = 0.0
    _t0: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._t0 is None:
            self._t0 = self.start if self.start.tzinfo else self.start.replace(tzinfo=TZ)
        self._mono0 = time.monotonic()

    def now(self) -> datetime:
        elapsed = (time.monotonic() - self._mono0) * float(self.scale)
        return self._t0 + timedelta(seconds=elapsed)


def make_clock(*, mode: str, start: str | None = None, scale: float = 1.0) -> Clock:
    """构造时钟。`start` 是 ISO8601 字符串，缺省时区按 +08:00。"""
    when = parse_iso(start) if start else None
    if mode == "real":
        return RealClock()
    if mode == "mock":
        return MockClock(start=when or MockClock.start)
    if mode == "scaled":
        return ScaledClock(start=when or ScaledClock.start, scale=scale)
    raise ConfigError(f"未知时钟模式：{mode!r}（可选 real/scaled/mock）")


def parse_iso(text: str) -> datetime:
    """解析 ISO8601；无时区则按 +08:00 解释。"""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=TZ)


def day_index(birth: datetime, when: datetime) -> int:
    """从出生时刻起算的日序号（0 = 出生当天）。"""
    return max(0, (when.date() - birth.date()).days)


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


def at_time_of_day(day: datetime, hour: int, minute: int = 0) -> datetime:
    """把某天挪到指定时刻（保留时区）。"""
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)
