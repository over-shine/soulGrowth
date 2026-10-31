"""按真实时间一格格往前走。

批量模式（``persona run``）是"把 30 天压缩进 1 分钟"——它回答的是"**会发生**什么"；
live 模式回答的是另一个问题："**现在**他在做什么"。两者跑的是同一段状态机
（``Simulation.begin_day`` / ``run_step``），区别只在两格之间睡多久。

pace 的含义：**1 个真实秒 = pace 个模拟秒**。
- pace=60  → 一个 30 分钟的格子等于 30 真实秒，一天（16 小时清醒）约 16 分钟；
- pace=1800→ 一个格子 1 真实秒，一天 32 秒（快速看完整天）；
- pace=1   → 真实时间，一天就是一天（长跑用）。

停机补做（docs/02 D4）：进程被杀、机器睡了、你按了 Ctrl+C——重新起来时会先把
中断的那一天**补完**，这些格子带 ``catch_up=true``。它们照样进库、照样算里程碑，
但在特质投影里权重被压低（I10）：没被"经历过"的时间不该和经历过的一样重。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from persona.clock import TZ, parse_iso
from persona.log import get
from persona.runtime.runner import DAY_START_HOUR, SLOT_MINUTES, Simulation

logger = get("scheduler")

SLOT_SECONDS = SLOT_MINUTES * 60


@dataclass
class LiveStats:
    """这一轮 live 干了什么。给人看的，不是给系统的。"""

    start_day: int = 0
    day: int = 0
    slots: int = 0
    episodes: int = 0
    catch_up: int = 0
    days_done: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_day": self.start_day,
            "day": self.day,
            "slots": self.slots,
            "episodes": self.episodes,
            "catch_up": self.catch_up,
            "days_done": self.days_done,
            "elapsed_s": round(time.monotonic() - self.started_at, 3),
        }


class LiveScheduler:
    """把「一天」拆成 32 格，按 pace 摊到真实时间里。"""

    def __init__(
        self,
        sim: Simulation,
        *,
        pace: float = 60.0,
        tick_s: float = 0.25,
        catch_up_after_s: float = 5.0,
        on_episode: Callable[[Any], None] | None = None,
        on_day: Callable[[Any], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sim = sim
        self.pace = max(1e-6, float(pace))
        self.tick_s = max(0.01, float(tick_s))
        #: 落后超过这么多**真实秒**才标记 catch_up（默认 5s；pace=60 时约等于 5 个模拟小时）
        self.catch_up_after_s = float(catch_up_after_s)
        self.on_episode = on_episode
        self.on_day = on_day
        self._sleep = sleep
        self._clock = clock
        self._stop = False
        self.stats = LiveStats()

    # -- 时间换算 -----------------------------------------------------------
    @property
    def slot_real_s(self) -> float:
        return SLOT_SECONDS / self.pace

    # -- 启动时：把中断的那一天补完 -----------------------------------------
    def recover(self) -> int:
        """返回「接下来该过哪一天」。中途被杀的那一天先补完，留下的格子标 catch_up。"""
        repo = self.sim.repo
        ts = repo.last_episode_ts()
        if not ts:
            self.stats.start_day = 0
            self.stats.day = 0
            return 0
        date = ts[:10]
        episodes = repo.episodes_of_day(date)
        last = episodes[-1] if episodes else None
        day = int(last.get("age_days", 0)) if last else 0
        if last is not None and last.get("kind") != "sleep":
            logger.warning("上次停在第 %d 天中间（%s），先把它补完", day, last.get("id"))
            session = self.sim.begin_day(day, resume=True)
            if not session.done:
                while not self._stop:
                    kind, _payload = self.sim.run_step(catch_up=True)
                    self.stats.slots += 1
                    self.stats.catch_up += 1
                    if kind == "day_end":
                        self.stats.days_done += 1
                        break
        # 不管刚才补没补，最后一个过完的日子都是 day
        day += 1
        self.stats.start_day = day
        self.stats.day = day
        return day

    # -- 主循环 -------------------------------------------------------------
    def run(self, *, until_day: int | None = None, max_days: int | None = None) -> LiveStats:
        day = self.recover()
        started = day
        try:
            while not self._stop:
                if until_day is not None and day > until_day:
                    break
                if max_days is not None and (day - started) >= max_days:
                    break
                if not self._run_day(day):
                    break
                day += 1
                self.stats.days_done += 1
                self.stats.day = day
        finally:
            self.sim.save()
        return self.stats

    def _run_day(self, day: int) -> bool:
        """走完一天。返回 False 表示被 stop 打断。"""
        session = self.sim.begin_day(day)
        if session.done:
            if self.on_day is not None:
                self.on_day(session.result)
            return True
        next_at = self._clock()
        while not self._stop:
            now = self._clock()
            wait = next_at - now
            if wait > 0:
                self._sleep(min(self.tick_s, wait))
                continue
            late = -wait
            kind, payload = self.sim.run_step(catch_up=late >= self.catch_up_after_s)
            self.stats.slots += 1
            if late >= self.catch_up_after_s:
                self.stats.catch_up += 1
            if kind == "episode":
                self.stats.episodes += 1
                if self.on_episode is not None:
                    self.on_episode(payload)
            next_at += self.slot_real_s
            if kind == "day_end":
                if self.on_day is not None:
                    self.on_day(payload)
                return True
        return False

    def stop(self) -> None:
        """让循环在走完当前这一格后停下（Ctrl+C 用）。"""
        self._stop = True


def describe_ts(ts: str) -> str:
    """把 ISO 时间戳变成 HH:MM（本地时区）。"""
    try:
        return parse_iso(ts).astimezone(TZ).strftime("%H:%M")
    except (TypeError, ValueError):
        return str(ts)[11:16]


def now_iso() -> str:
    return datetime.now(TZ).isoformat()


__all__ = ["LiveScheduler", "LiveStats", "describe_ts", "DAY_START_HOUR"]
