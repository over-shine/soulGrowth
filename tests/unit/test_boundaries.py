"""负能力：这个阶段**不存在**的东西，编译器不能凭空造出来。

这是 docs/08 的五道防线里最便宜也最有效的一道（D5）：不测「能做到什么」，
只测「做不到什么」。它与里程碑无关，任何一次改动都不该让它变绿。
"""

from __future__ import annotations

import unittest

from persona.body.compiler import Compiler, FALLBACK_CHAINS
from persona.body.primitives import default_registry
from persona.types import STAGES

#: S0 的 7 个动作（docs/03 S0 解锁表）。一个都不能多，一个都不能少。
S0_ACTS = {
    "A0.FIXATE",
    "A0.TURN_HEAD",
    "A0.GRASP",
    "A0.SUCK",
    "A0.CRY",
    "A0.STARTLE",
    "A0.SLEEP",
}


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = default_registry()

    def test_s0_exactly_seven(self) -> None:
        ids = {p.id for p in self.registry.for_stage("S0") if p.adapter != "none"}
        self.assertEqual(ids, S0_ACTS)

    def test_no_speech_primitive_anywhere(self) -> None:
        # 「说话」不是一个原语，是一层层能力长出来的：S0/S1 不存在 SPEAK/SAY
        for prim in self.registry.all():
            self.assertNotIn(prim.id.split(".")[-1], {"SPEAK", "SAY", "TALK", "SPEECH"})

    def test_stage_order(self) -> None:
        order = []
        for stage in STAGES:
            order.append(len(self.registry.for_stage(stage)))
        self.assertEqual(order, sorted(order), "解锁集合必须随阶段单调变大")

    def test_fallback_chains_terminate(self) -> None:
        # 降级链必须单向前进且终止于 A0.NONE，否则一次拒绝就能把进程拖死在循环里
        for start, chain in FALLBACK_CHAINS.items():
            seen = [start]
            for nxt in chain:
                if nxt in seen:
                    self.fail(f"{start} 的降级链出现环：{seen} → {nxt}")
                seen.append(nxt)
            self.assertEqual(seen[-1], "A0.NONE", f"{start} 的降级链没有终止于 A0.NONE")


class CompileGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = default_registry()
        self.compiler = Compiler(self.registry)

    def test_locked_primitive_never_compiles(self) -> None:
        # S3 的 SENTENCE 在 S0 必须降级成 S0 真有的东西（这里是哭），绝不能原样返回
        compiled = self.compiler.compile(
            "A3.SENTENCE",
            {},
            stage="S0",
            body=_body(),
            world=_world(),
            lexicon=_lexicon(),
        )
        self.assertNotEqual(compiled.primitive, "A3.SENTENCE")
        self.assertIn(compiled.primitive, S0_ACTS)
        self.assertTrue(compiled.degraded)
        self.assertIn("STAGE_LOCKED", compiled.denials)

    def test_cry_always_has_a_reason(self) -> None:
        compiled = self.compiler.compile(
            "A0.CRY",
            {"intensity": 2},
            stage="S0",
            body=_body(hunger=0.9),
            world=_world(),
            lexicon=_lexicon(),
        )
        self.assertEqual(compiled.primitive, "A0.CRY")
        self.assertEqual(compiled.params.get("reason"), "hunger")


def _body(*, hunger: float = 0.3):
    from persona.body.state import BodyState

    body = BodyState()
    body.hunger = hunger
    return body


def _world():
    from persona.world.generator import create_world

    return create_world(individual_id="soul-test", birth_iso="2026-01-01T06:00:00+08:00")


def _lexicon():
    from persona.language.lexicon import Lexicon

    return Lexicon()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
