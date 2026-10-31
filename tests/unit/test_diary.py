"""日记是给人读的，所以它必须稳定、可 diff、不替他说话。"""

from __future__ import annotations

import unittest

from persona.observe.diary import render_day


def ep(*, eid: str, ts: str, prim: str, params: dict | None = None, kind: str = "awake", drives=None) -> dict:
    return {
        "id": eid,
        "ts": ts,
        "kind": kind,
        "stage": "S0",
        "age_days": 0,
        "action": {"primitive": prim, "params": params or {}},
        "context": {"others": ["mother"]},
        "observation": {"salient": ["mother"], "summary": "母亲在床边"},
        "drives": drives
        or {"energy": 0.4, "stimulation": 0.6, "social": 0.2, "competence": 0.3, "safety": 0.1},
    }


def day() -> list[dict]:
    return [
        ep(eid="ep-20260101-0001", ts="2026-01-01T06:00:00+08:00", prim="A0.FIXATE", params={"target": "mother"}),
        ep(eid="ep-20260101-0002", ts="2026-01-01T06:30:00+08:00", prim="A0.CRY", params={"reason": "hunger"}),
        ep(eid="ep-20260101-0003", ts="2026-01-01T07:00:00+08:00", prim="A0.CRY", params={"reason": "hunger"}),
        ep(eid="ep-20260101-0004", ts="2026-01-01T21:30:00+08:00", prim="A0.SLEEP", kind="sleep"),
    ]


class DiaryTest(unittest.TestCase):
    def test_renders_the_facts_it_was_given(self) -> None:
        text = render_day(
            day=0,
            date="2026-01-01",
            stage="S0",
            episodes=day(),
            milestones=[{"id": "M0.1", "status": "passed", "value": 1.0}],
            traits={"T": {"O": 0.1, "C": 0.2, "E": 0.3, "A": 0.4, "N": -0.5}},
        )
        self.assertIn("# 第 0 天 · 2026-01-01", text)
        self.assertIn("哭了 2 次：饿 2 次", text)
        self.assertIn("M0.1 视觉定向：通过，值 1.00", text)
        self.assertIn("开放 +0.10", text)
        self.assertNotIn("\r\n", text)

    def test_same_input_same_text(self) -> None:
        a = render_day(day=0, date="2026-01-01", stage="S0", episodes=day())
        b = render_day(day=0, date="2026-01-01", stage="S0", episodes=day())
        self.assertEqual(a, b)

    def test_sleep_is_reported_as_sleep(self) -> None:
        text = render_day(day=0, date="2026-01-01", stage="S0", episodes=day())
        self.assertIn("1 段睡眠", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
