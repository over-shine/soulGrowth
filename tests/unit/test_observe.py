"""观察台是只读的：只回答 GET，写不进去任何东西。

这些用例跑一遍真实的一天（mock，无网络），然后起一个真正的 HTTP 服务去访问它
——观察台的价值全在「它和主进程是两个东西」，所以不能在进程内函数调用里假装。
"""

from __future__ import annotations

import json
import shutil
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from persona.config import load
from persona.errors import StoreError
from persona.observe.server import Observer, WatchServer
from persona.runtime.runner import open_simulation

TMP_ROOT = Path(__file__).resolve().parents[2] / "data" / "tmp"


def _scratch() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"obs-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def _config(tmp: Path) -> Path:
    cfg = {
        "individual": {"id": "soul-test", "name": "小测", "birth": "2026-01-01T06:00:00+08:00"},
        "store": {"path": str(tmp / "soul.db"), "events": str(tmp / "events.jsonl")},
        "clock": {"mode": "mock", "start": "2026-01-01T06:00:00+08:00"},
        "runtime": {"diary": True},
    }
    path = tmp / "cfg.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return path


class ObserverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = _scratch()
        cfg_path = _config(cls.tmp)
        cls.cfg = load(["config/default.json", str(cfg_path)])
        sim = open_simulation(cls.cfg)
        try:
            sim.born()
            sim.run_day(0)
        finally:
            sim.close()
        cls.observer = Observer(cls.cfg)
        cls.server = WatchServer(cls.observer, port=0).start()
        cls.base = cls.server.url

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.server.stop()
        finally:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _get(self, path: str) -> tuple[int, str, str]:
        with urllib.request.urlopen(self.base + path.lstrip("/"), timeout=10) as res:
            return res.status, res.headers.get("Content-Type", ""), res.read().decode("utf-8")

    # -- 正常路径 -----------------------------------------------------------
    def test_index_is_served(self) -> None:
        status, ctype, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("成长观察台", body)

    def test_state_matches_the_day(self) -> None:
        _, ctype, body = self._get("/api/state")
        self.assertIn("application/json", ctype)
        state = json.loads(body)
        self.assertEqual(state["individual"]["id"], "soul-test")
        self.assertEqual(state["stage"], "S0")
        self.assertEqual(state["today"]["episodes"], 33)
        self.assertEqual(len(state["milestones"]), 7)
        self.assertEqual(state["diary_dates"], ["2026-01-01"])
        self.assertGreater(state["counts"]["episodes"], 0)
        self.assertEqual(len(state["traits"]["T"]), 5)

    def test_episode_detail_has_the_chain(self) -> None:
        _, _, body = self._get("/api/day?date=2026-01-01")
        episodes = json.loads(body)["episodes"]
        self.assertEqual(len(episodes), 33)
        _, _, detail_body = self._get(f"/api/episode?id={episodes[0]['id']}")
        detail = json.loads(detail_body)
        self.assertIn("5_选择", detail)
        self.assertIn("2_评价", detail)
        self.assertIn("不变量", detail)

    def test_diary_is_returned_as_markdown(self) -> None:
        _, _, body = self._get("/api/diary?date=2026-01-01")
        payload = json.loads(body)
        self.assertEqual(payload["date"], "2026-01-01")
        self.assertIn("# 第 0 天", payload["markdown"])

    def test_trend_and_events(self) -> None:
        _, _, body = self._get("/api/trend")
        days = json.loads(body)["days"]
        self.assertEqual(len(days), 1)
        self.assertEqual(days[0]["episodes"], 33)
        _, _, body = self._get("/api/events?limit=5")
        self.assertEqual(len(json.loads(body)["events"]), 5)

    # -- 只读与拒绝 ---------------------------------------------------------
    def test_writes_are_rejected(self) -> None:
        req = urllib.request.Request(self.base + "api/state", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 405)
        body = ctx.exception.read().decode("utf-8")
        self.assertIn("只读", body)

    def test_unknown_path_is_404(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_static_traversal_is_blocked(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/static/%2e%2e/server.py")
        self.assertEqual(ctx.exception.code, 404)

    def test_missing_database_is_explained(self) -> None:
        empty = _scratch()
        try:
            cfg = load(["config/default.json", str(_config(empty))])
            observer = Observer(cfg)
            with self.assertRaises(StoreError) as ctx:
                observer.state()
            self.assertIn("还没有数据", str(ctx.exception))
        finally:
            shutil.rmtree(empty, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
