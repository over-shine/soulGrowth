"""live 模式：把一天拆成格子摊到真实时间里，且被杀之后能从半路接上。

这里用假时钟，不真的等——但走的是 LiveScheduler 真正的循环代码，包括
`recover()` 的续跑判断（那是崩溃恢复唯一可能出错的地方）。
"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from persona.config import load
from persona.runtime.runner import open_simulation
from persona.runtime.scheduler import LiveScheduler

TMP_ROOT = Path(__file__).resolve().parents[2] / "data" / "tmp"


def _scratch() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"live-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def _config(tmp: Path) -> Path:
    cfg = {
        "individual": {"id": "soul-live", "name": "小跑", "birth": "2026-01-01T06:00:00+08:00"},
        "store": {"path": str(tmp / "soul.db"), "events": str(tmp / "events.jsonl")},
        "clock": {"mode": "mock", "start": "2026-01-01T06:00:00+08:00"},
        "runtime": {"diary": False},
    }
    path = tmp / "cfg.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return path


class FakeClock:
    """now() 返回累加的时间；sleep(x) 让它前进 x 秒（不真睡）。"""

    def __init__(self) -> None:
        self.t = 1000.0
        self.slept = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.t += seconds

    def jump(self, seconds: float) -> None:
        self.t += seconds


class LiveSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _scratch()
        self.cfg_path = _config(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _open(self):
        return open_simulation(load([str(self.cfg_path)]))

    def test_pace_maps_slot_to_real_seconds(self) -> None:
        sim = self._open()
        try:
            fast = LiveScheduler(sim, pace=60.0)
            slow = LiveScheduler(sim, pace=1.0)
            self.assertAlmostEqual(fast.slot_real_s, 30.0)   # 30 分钟 / 60
            self.assertAlmostEqual(slow.slot_real_s, 1800.0)  # 真实时间
        finally:
            sim.close()

    def test_one_day_walks_all_slots_and_reports_no_catch_up(self) -> None:
        sim = self._open()
        try:
            clock = FakeClock()
            seen: list[str] = []
            days: list[int] = []
            sched = LiveScheduler(
                sim,
                pace=3600.0,  # 一个格子 0.5 真实秒
                tick_s=0.1,
                sleep=clock.sleep,
                clock=clock.now,
                on_episode=lambda ep: seen.append(ep.id),
                on_day=lambda res: days.append(res.day),
            )
            stats = sched.run(max_days=1)
            self.assertEqual(stats.days_done, 1)
            self.assertEqual(days, [0])
            # 32 个清醒格子各打一条；夜里那一条属于「一天结束」，不算一步
            self.assertEqual(len(seen), 32)
            self.assertEqual(stats.episodes, 32)
            self.assertEqual(stats.catch_up, 0)
            # 一天 32 格 × 0.5s + 夜间那一条不等：睡眠没有时间成本
            self.assertLess(clock.slept, 33.0)
            self.assertEqual(sim.repo.count_episodes_of_day("2026-01-01"), 33)
            # 每条 live episode 都带真实的钟点（06:00 起、每 30 分钟一格）
            first = sim.repo.episodes_of_day("2026-01-01")[0]
            last_awake = sim.repo.episodes_of_day("2026-01-01")[31]
            self.assertTrue(first["ts"].endswith("06:00:00+08:00"))
            self.assertTrue(last_awake["ts"].endswith("21:30:00+08:00"))
        finally:
            sim.close()

    def test_behind_schedule_is_marked_catch_up(self) -> None:
        sim = self._open()
        try:
            clock = FakeClock()
            flagged: list[bool] = []
            sched = LiveScheduler(
                sim,
                pace=3600.0,
                tick_s=0.1,
                catch_up_after_s=5.0,
                sleep=clock.sleep,
                clock=clock.now,
                on_episode=lambda ep: flagged.append(bool(ep.catch_up)),
            )
            # 半路让时钟跳 10 分钟：后面的格子都会被标成补做
            original = clock.sleep

            def sleep_then_jump(seconds: float) -> None:
                original(seconds)
                if len(flagged) == 3:
                    clock.jump(600.0)

            sched._sleep = sleep_then_jump
            stats = sched.run(max_days=1)
            self.assertGreater(stats.catch_up, 0)
            self.assertIn(True, flagged)
            self.assertEqual(len(flagged), 32)
            self.assertEqual(stats.episodes, 32)
            self.assertEqual(len(sim.repo.episodes_of_day("2026-01-01")), 33)
        finally:
            sim.close()

    def test_killed_mid_day_resumes_where_it_stopped(self) -> None:
        sim = self._open()
        try:
            session = sim.begin_day(0)
            for _ in range(10):
                kind, _ = sim.run_step()
                self.assertEqual(kind, "episode")
            self.assertEqual(sim.repo.count_episodes_of_day("2026-01-01"), 10)
        finally:
            sim.close()  # 相当于进程被杀：状态留在磁盘上

        sim = self._open()
        try:
            clock = FakeClock()
            sched = LiveScheduler(sim, pace=3600.0, sleep=clock.sleep, clock=clock.now)
            self.assertEqual(sched.recover(), 1)  # 补完第 0 天，接下来是第 1 天
            self.assertEqual(sim.repo.count_episodes_of_day("2026-01-01"), 33)
            self.assertGreaterEqual(sched.stats.catch_up, 22)
        finally:
            sim.close()

    def test_finished_day_is_not_rerun(self) -> None:
        sim = self._open()
        try:
            sim.run_day(0)
            clock = FakeClock()
            sched = LiveScheduler(sim, pace=3600.0, sleep=clock.sleep, clock=clock.now)
            self.assertEqual(sched.recover(), 1)
            self.assertEqual(sched.stats.catch_up, 0)
            self.assertEqual(sim.repo.count_episodes_of_day("2026-01-01"), 33)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
