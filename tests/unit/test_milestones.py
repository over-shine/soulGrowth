"""里程碑判定器：给定一段行为，判它通过、不通过，还是样本不够。

这些测试不跑模拟，只喂假的 episode 列表——判据的语义应该独立于世界实现。
"""

from __future__ import annotations

import unittest

from persona.stages import milestones as ms


def ep(
    *,
    eid: str = "ep-20260101-0001",
    ts: str = "2026-01-01T06:00:00+08:00",
    age: int = 0,
    prim: str = "A0.FIXATE",
    params: dict | None = None,
    others: list[str] | None = None,
    salient: list[str] | None = None,
    onsets: list[str] | None = None,
) -> dict:
    return {
        "id": eid,
        "ts": ts,
        "age_days": age,
        "action": {"primitive": prim, "params": params or {}},
        "context": {"others": others if others is not None else ["mother"]},
        "observation": {
            "salient": salient if salient is not None else ["mother", "mobile"],
            "motion_onsets": onsets or [],
            "novelty": {},
        },
    }


class TrackingTest(unittest.TestCase):
    def test_tracks_moving_object_more_than_still_one(self) -> None:
        eps = []
        # 晚期（第 3 周）：动的时候七成看它，不动的时候只有一成——差距真实存在
        for i in range(40):
            onset = i % 2 == 0
            hit = (i % 10 < 7) if onset else (i % 10 < 1)
            eps.append(
                ep(
                    eid=f"ep-20260121-{i:04d}",
                    ts=f"2026-01-21T06:{i:02d}:00+08:00",
                    age=20,
                    prim="A0.FIXATE" if hit else "A0.CRY",
                    params={"target": "mobile"} if hit else {"reason": "hunger"},
                    onsets=["mobile"] if onset else [],
                )
            )
        # 第 1 周：动与不动都只有两成被看——还没有追踪
        for i in range(40):
            onset = i % 2 == 0
            hit = i % 10 < 2
            eps.append(
                ep(
                    eid=f"ep-20260102-{i:04d}",
                    ts=f"2026-01-02T06:{i:02d}:00+08:00",
                    age=1,
                    prim="A0.FIXATE" if hit else "A0.CRY",
                    params={"target": "mobile"} if hit else {"reason": "discomfort"},
                    onsets=["mobile"] if onset else [],
                )
            )
        ev = ms.eval_m0_2(eps)
        self.assertEqual(ev.status, ms.PASSED, str(ev.to_dict()))
        self.assertGreaterEqual(ev.value, 0.30)

    def test_too_little_data_is_not_a_failure(self) -> None:
        ev = ms.eval_m0_2([ep(onsets=["mobile"])])
        self.assertEqual(ev.status, ms.INSUFFICIENT)


class CryDifferentiationTest(unittest.TestCase):
    def test_needs_five_of_each(self) -> None:
        eps = []
        for reason, n in (("hunger", 5), ("discomfort", 5), ("sleepy", 4)):
            for i in range(n):
                eps.append(
                    ep(eid=f"ep-2026010{len(eps) % 9}-{len(eps):04d}", prim="A0.CRY", params={"reason": reason})
                )
        ev = ms.eval_m0_6(eps)
        self.assertEqual(ev.status, ms.FAILED)
        self.assertEqual(ev.value, 4.0)


class CaregiverRecognitionTest(unittest.TestCase):
    def _block(self, *, age: int, lonely_rate: float, n: int = 40) -> list[dict]:
        eps = []
        for i in range(n):
            lonely = i < int(round(lonely_rate * n))
            eps.append(
                ep(
                    eid=f"ep-202601{age:02d}-{i:04d}",
                    age=age,
                    prim="A0.CRY",
                    params={"reason": "lonely" if lonely else "discomfort"},
                    others=[],  # 照料者不在
                )
            )
        return eps

    def test_lonely_cry_appears_only_later(self) -> None:
        eps = self._block(age=4, lonely_rate=0.0) + self._block(age=25, lonely_rate=0.5)
        ev = ms.eval_m0_7(eps)
        self.assertEqual(ev.status, ms.PASSED, str(ev.to_dict()))

    def test_flat_rate_is_not_growth(self) -> None:
        eps = self._block(age=4, lonely_rate=0.5) + self._block(age=25, lonely_rate=0.5)
        ev = ms.eval_m0_7(eps)
        self.assertEqual(ev.status, ms.FAILED)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
