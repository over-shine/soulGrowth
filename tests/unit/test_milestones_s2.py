"""S2 里程碑判据：自平复、词汇爆发、独立行走。

只喂构造出来的 episode，判的是「判据有没有按说好的规则算」，不判行为本身。
"""

from __future__ import annotations

import unittest

from persona.stages import milestones as ms


def e(
    *,
    eid: str,
    ts: str,
    prim: str = "A0.FIXATE",
    valence: float = 0.0,
    events: list[str] | None = None,
    utterance: str | None = None,
    params: dict | None = None,
    age: int = 30,
) -> dict:
    return {
        "id": eid,
        "ts": ts,
        "age_days": age,
        "appraisal": {"valence": valence},
        "action": {"primitive": prim, "params": params or {}},
        "language": {"utterance": utterance},
        "outcome": {"events": events or []},
    }


class SelfSootheTest(unittest.TestCase):
    def _series(self, n: int, *, recover: bool) -> list[dict]:
        eps = []
        for i in range(n):
            eps.append(e(eid=f"ep-20260101-{i:04d}", ts="2026-01-01T06:00:00+08:00", valence=-0.4))
            if recover:
                eps.append(e(eid=f"ep-20260101-{i:04d}b", ts="2026-01-01T07:00:00+08:00", valence=0.1))
            else:
                eps.append(e(eid=f"ep-20260101-{i:04d}b", ts="2026-01-01T07:00:00+08:00", valence=-0.4))
        return eps

    def test_self_recovery_counts(self) -> None:
        ev = ms.eval_m2_7(self._series(30, recover=True))
        self.assertEqual(ev.status, ms.PASSED)
        self.assertAlmostEqual(float(ev.value), 1.0, places=3)

    def test_staying_upset_does_not_count(self) -> None:
        ev = ms.eval_m2_7(self._series(30, recover=False))
        self.assertEqual(ev.status, ms.FAILED)
        self.assertAlmostEqual(float(ev.value), 0.0, places=3)

    def test_being_held_does_not_count(self) -> None:
        eps = []
        for i in range(30):
            eps.append(e(eid=f"ep-20260101-{i:04d}", ts="2026-01-01T06:00:00+08:00", valence=-0.4))
            eps.append(
                e(
                    eid=f"ep-20260101-{i:04d}b",
                    ts="2026-01-01T07:00:00+08:00",
                    valence=0.2,
                    events=["caregiver_hold"],
                )
            )
        ev = ms.eval_m2_7(eps)
        self.assertEqual(ev.status, ms.FAILED)

    def test_too_few_distress_episodes_is_insufficient(self) -> None:
        ev = ms.eval_m2_7(self._series(3, recover=True))
        self.assertEqual(ev.status, ms.INSUFFICIENT)


class VocabularyBurstTest(unittest.TestCase):
    def test_fifty_words_in_two_weeks(self) -> None:
        eps = []
        for i in range(50):
            day = i % 14
            eps.append(
                e(
                    eid=f"ep-202601{i:02d}-0001",
                    ts=f"2026-01-{day + 1:02d}T09:00:00+08:00",
                    prim="A1.WORD",
                    utterance=f"词{i}",
                )
            )
        ev = ms.eval_m2_2(eps)
        self.assertEqual(ev.status, ms.PASSED)
        self.assertGreaterEqual(float(ev.value), 50.0)

    def test_words_spread_over_months_are_not_a_burst(self) -> None:
        eps = []
        for i in range(50):
            month = i % 6 + 1
            eps.append(
                e(
                    eid=f"ep-2026{month:02d}01-0001",
                    ts=f"2026-{month:02d}-01T09:00:00+08:00",
                    prim="A1.WORD",
                    utterance=f"词{i}",
                )
            )
        ev = ms.eval_m2_2(eps)
        self.assertEqual(ev.status, ms.FAILED)


class WalkingTest(unittest.TestCase):
    def test_ten_walks_a_day_for_three_days(self) -> None:
        eps = []
        for day in range(1, 4):
            for i in range(10):
                eps.append(
                    e(
                        eid=f"ep-2026010{day}-{i:04d}",
                        ts=f"2026-01-0{day}T09:00:00+08:00",
                        prim="A2.WALK",
                        events=["stepped:0.1m"],
                    )
                )
        ev = ms.eval_m2_1(eps)
        self.assertEqual(ev.status, ms.PASSED)

    def test_walking_in_place_does_not_count(self) -> None:
        eps = []
        for day in range(1, 4):
            for i in range(10):
                eps.append(
                    e(
                        eid=f"ep-2026010{day}-{i:04d}",
                        ts=f"2026-01-0{day}T09:00:00+08:00",
                        prim="A2.WALK",
                        events=["already_there"],
                    )
                )
        ev = ms.eval_m2_1(eps)
        self.assertEqual(ev.status, ms.FAILED)


class StageGateTest(unittest.TestCase):
    def test_a_stage_without_evaluators_cannot_be_left(self) -> None:
        from persona.stages import judge

        d = judge.can_advance("S3", [], days_in_stage=999, min_dwell_days=1)
        self.assertFalse(d.advance)
        self.assertIn("判定器", d.reason)

    def test_a_stage_with_a_failing_milestone_cannot_be_left(self) -> None:
        from persona.stages import judge

        ev = ms.Evaluation("M2.2", "S2", ms.FAILED, 10.0, 50.0, [], "proxy")
        d = judge.can_advance("S2", [ev], days_in_stage=999, min_dwell_days=1)
        self.assertFalse(d.advance)


if __name__ == "__main__":
    unittest.main()
