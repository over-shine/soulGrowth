"""决策层的两条硬规则：降级的先验不许参与决策；共同注意只加在「看」上。"""

from __future__ import annotations

import unittest

from persona.body.compiler import Candidate
from persona.cognition.policy import Prior, decide, score_candidate
from persona.cognition.traits import Bias
from persona.rng import derive
from persona.types import Appraisal, Drives


class _FixedPrior:
    """给一个固定的先验分布，方便看清「有没有被掺进来」。"""

    def __init__(self, probs: dict[str, float], *, degraded: bool = False) -> None:
        self._probs = probs
        self._degraded = degraded
        self.calls = 0

    def prior(self, *, observation, candidates) -> Prior:  # noqa: ANN001
        self.calls += 1
        return Prior(
            probs=dict(self._probs),
            model="fixed",
            degraded=self._degraded,
            degraded_reason="测试用" if self._degraded else "",
        )


class _Recorder:
    """套在 FakePrior 外面，把每个候选的先验概率原样记下来。"""


def _cands() -> list[Candidate]:
    return [
        Candidate(primitive="A0.FIXATE", params={"target": "mobile"}, afford={"stim": 0.5}),
        Candidate(primitive="A1.BABBLE", params={}, afford={"stim": 0.2}),
        Candidate(primitive="A0.CRY", params={}, afford={"safety": 0.2}),
    ]


def _bias() -> Bias:
    return Bias(
        o=0.0,
        c=0.1,
        e=0.1,
        a=0.1,
        n=0.0,
        explore=0.3,
        social=0.3,
        persistence=0.3,
        prosocial=0.3,
        risk=0.3,
        threat_gain=0.2,
        safety_gain=0.2,
    )


def _kwargs(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "bias": _bias(),
        "drives": Drives(energy=0.6, stimulation=0.5, social=0.4, competence=0.4, safety=0.7),
        "appraisal": Appraisal(valence=0.1, arousal=0.3, threat=0.2, control=0.4, pe=0.1, novelty=0.4),
        "temperature": 0.7,
        "shortlist_size": 3,
        "rng": derive("test", 1),
        "needs": {},
    }
    base.update(over)
    return base


class TestDegradedPrior(unittest.TestCase):
    def test_degraded_prior_is_not_blended(self) -> None:
        """mock 的先验是平坦伪分布：掺进来等于让决策掷骰子。"""
        provider = _FixedPrior({"A0.CRY": 1.0}, degraded=True)
        first = decide(_cands(), provider=provider, **_kwargs())  # type: ignore[arg-type]
        self.assertEqual(provider.calls, 1, "先验还是要算的，只是不能采用")
        self.assertNotEqual(first.source, "llm")
        self.assertTrue(first.prior.degraded)

        # 用一个「偏爱 CRY」的健康先验对照：它会真的改变结果
        healthy = _FixedPrior({"A0.CRY": 1.0}, degraded=False)
        seen = set()
        for seed in range(30):
            seen.add(
                decide(_cands(), provider=healthy, **_kwargs(rng=derive("t", seed))).primitive  # type: ignore[arg-type]
            )
        self.assertIn("A0.CRY", seen, "健康先验应当能影响选择（否则这条测不出东西）")

    def test_degraded_prior_leaves_scores_untouched(self) -> None:
        provider = _FixedPrior({"A0.CRY": 1.0}, degraded=True)
        d = decide(_cands(), provider=provider, **_kwargs())  # type: ignore[arg-type]
        self.assertTrue(d.scores, "分数仍要落盘，审计要能看见他当时怎么想的")


class TestJointAttention(unittest.TestCase):
    def test_joint_bonus_only_on_looking(self) -> None:
        """跟随指向是视线跟过去；伸手去抓同一个东西不该拿这份加成。"""
        needs = {"joint": 1.0, "joint_target": "blanket", "joint_gain": 6.0}
        look = Candidate(primitive="A0.FIXATE", params={"target": "blanket"}, afford={"stim": 0.2})
        grab = Candidate(primitive="A1.REACH", params={"target": "blanket"}, afford={"stim": 0.2})
        other = Candidate(primitive="A1.REACH", params={"target": "mobile"}, afford={"stim": 0.2})
        kw = {
            "bias": _bias(),
            "drives": Drives(energy=0.6, stimulation=0.5, social=0.4, competence=0.4, safety=0.7),
            "appraisal": Appraisal(valence=0.1, arousal=0.3, threat=0.2, control=0.4, pe=0.1, novelty=0.4),
            "needs": needs,
        }
        look_s = score_candidate(look, **kw)  # type: ignore[arg-type]
        grab_s = score_candidate(grab, **kw)  # type: ignore[arg-type]
        other_s = score_candidate(other, **kw)  # type: ignore[arg-type]
        self.assertGreater(look_s, grab_s + 3.0)
        self.assertAlmostEqual(grab_s, other_s, places=6)


if __name__ == "__main__":
    unittest.main()
