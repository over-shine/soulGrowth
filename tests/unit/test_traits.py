"""特质投影的权重（I10）：补做的格子不该和真的活着一样重。

不是"打折听起来合理"，而是一个可检验的断言：同一天的经历，如果有一半是停机后
补做的，那天的哭声占比必须**低于**它们都是真实经历时的值。
"""

from __future__ import annotations

import unittest

from persona.cognition.traits import CATCH_UP_WEIGHT, episode_weight, project_indicators, project_traits


def ep(prim: str, *, catch_up: bool = False, valence: float = 0.0) -> dict:
    return {
        "action": {"primitive": prim},
        "appraisal": {"threat": 0.1, "valence": valence},
        "outcome": {"success": prim != "A0.CRY"},
        "context": {"others": "mother" if prim == "A0.SUCK" else ""},
        "language": {"utterance": None},
        "catch_up": catch_up,
    }


class WeightsTest(unittest.TestCase):
    def test_weight_is_half_for_catch_up(self) -> None:
        self.assertEqual(episode_weight(ep("A0.CRY")), 1.0)
        self.assertEqual(episode_weight(ep("A0.CRY", catch_up=True)), CATCH_UP_WEIGHT)
        self.assertLessEqual(CATCH_UP_WEIGHT, 0.5)  # I10

    def test_catch_up_cries_count_less(self) -> None:
        real = [ep("A0.CRY")] * 6 + [ep("A0.FIXATE")] * 6
        mixed = [ep("A0.CRY", catch_up=True)] * 6 + [ep("A0.FIXATE")] * 6
        all_real = project_indicators([dict(e, catch_up=False) for e in real])
        some_fake = project_indicators(mixed)
        self.assertAlmostEqual(all_real["comfort_seeking"], 0.5)
        # 6 条补做的哭 + 6 条真实的看 => 3/(3+6)
        self.assertAlmostEqual(some_fake["comfort_seeking"], 3.0 / 9.0)

    def test_all_catch_up_day_shrinks_to_the_same_shape(self) -> None:
        # 整天的权重一起变小，比率不该变（权重在分子分母里同时出现）
        plain = project_indicators([ep("A0.CRY"), ep("A0.FIXATE"), ep("A0.SUCK"), ep("A0.GRASP")])
        flagged = project_indicators(
            [ep("A0.CRY", catch_up=True), ep("A0.FIXATE", catch_up=True),
             ep("A0.SUCK", catch_up=True), ep("A0.GRASP", catch_up=True)]
        )
        for key in plain:
            self.assertAlmostEqual(plain[key], flagged[key], places=9)

    def test_projection_uses_the_weights(self) -> None:
        # 补做的哭更多 => 舒适寻求指标更低 => B 的 N 维更低（威胁评价、安抚寻求同向）
        a = [ep("A0.CRY")] * 8 + [ep("A0.FIXATE")] * 8
        b = [ep("A0.CRY", catch_up=True)] * 8 + [ep("A0.FIXATE")] * 8
        ba, _, _ = project_traits(a)
        bb, _, _ = project_traits(b)
        self.assertGreater(ba["N"], bb["N"])


if __name__ == "__main__":
    unittest.main()
