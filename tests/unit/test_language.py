"""语言习得：词必须是从听见的话里长出来的，不是配置里写死的。

对应 docs/06 的四条硬约束：schema 里没有自由文本；出口词必须是 productive；
词形必须是一句真实话语的连续子串；说出来的词数按「词」算而不是按「字」算。
"""

from __future__ import annotations

import unittest

from persona.language import syntax
from persona.language.lexicon import COMPREHENSION_MIN, PRODUCTION_THRESHOLD, Lexicon
from persona.stages import milestones


def _feed(lex: Lexicon, *, times: int = 1) -> None:
    """喂一段「照料者语料」：两个词各自出现在多种说法、多个情境里。"""
    corpus = [
        ("妈妈抱抱", "soothe"),
        ("宝宝看妈妈", "play"),
        ("妈妈在这里", "greet"),
        ("宝宝醒啦", "greet"),
        ("宝宝饿了吗", "feed"),
        ("宝宝不怕", "soothe"),
    ]
    for _ in range(times):
        for text, ctx in corpus:
            lex.hear_speech(text, day=30, context=ctx)


class TestWordDiscovery(unittest.TestCase):
    """一条话音里的片段要长成词，得反复出现、出现在不同情境里。"""

    def test_repeated_word_becomes_entry(self) -> None:
        lex = Lexicon(stage="S1")
        _feed(lex)
        self.assertIsNotNone(lex.by_form("妈妈"))
        self.assertIsNotNone(lex.by_form("宝宝"))

    def test_single_context_is_not_enough(self) -> None:
        """只在一种情境里反复出现，不算词（可能只是某句固定话的一部分）。"""
        lex = Lexicon(stage="S1")
        for _ in range(10):
            lex.hear_speech("妈妈抱抱", day=30, context="soothe")
        self.assertIsNone(lex.by_form("妈妈抱抱"))
        # 「妈妈」在这句里出现 10 次，但只有 1 个情境 → 仍是片段，不是词
        self.assertIsNone(lex.by_form("妈妈"))

    def test_fragment_of_longer_word_is_not_a_word(self) -> None:
        lex = Lexicon(stage="S1")
        _feed(lex, times=3)
        # 「宝宝」是词，那么「宝宝」的左邻片段不该同时成为词
        self.assertIsNotNone(lex.by_form("宝宝"))
        self.assertIsNone(lex.by_form("宝宝看"))

    def test_every_form_is_a_substring_of_speech(self) -> None:
        """不变量：词形必须是一句真听过的话的连续子串。"""
        lex = Lexicon(stage="S1")
        _feed(lex, times=2)
        corpus = "妈妈抱抱 宝宝看妈妈 妈妈在这里 宝宝醒啦 宝宝饿了吗 宝宝不怕"
        for entry in lex.entries.values():
            self.assertTrue(
                any(entry.form in line.replace(" ", "") for line in corpus.split(" ")),
                f"{entry.form} 不在任何一句真话里",
            )

    def test_exposures_and_comprehension_grow(self) -> None:
        lex = Lexicon(stage="S1")
        _feed(lex)
        first = lex.by_form("妈妈")
        assert first is not None
        before = first.exposures
        _feed(lex, times=2)
        after = lex.by_form("妈妈")
        assert after is not None
        self.assertGreater(after.exposures, before)
        self.assertLessEqual(after.comprehension_confidence, 1.0)


class TestProduction(unittest.TestCase):
    """说得出来，靠的是练够了次数，而不是「词库里有」。"""

    def test_practice_then_produce_promotes(self) -> None:
        lex = Lexicon(stage="S1")
        _feed(lex)
        target = lex.practice_form()
        self.assertIsNotNone(target)
        assert target is not None
        threshold = PRODUCTION_THRESHOLD["S1"]
        self.assertEqual(target.status, "heard_only")
        for day in range(threshold):
            lex.produce(target.form, day=30 + day)
        promoted = lex.by_form(target.form)
        assert promoted is not None
        self.assertEqual(promoted.status, "productive")
        # 练习算「说过」，但不算「第一次说出一个词」——那是 A1.WORD 才有的记录
        self.assertIsNone(promoted.first_produced_day)
        lex.produce(target.form, day=42, as_word=True)
        self.assertEqual(promoted.first_produced_day, 42)

    def test_unpracticed_word_cannot_be_said(self) -> None:
        lex = Lexicon(stage="S1")
        _feed(lex)
        self.assertEqual(lex.productive_forms(), set())
        entry = lex.by_form("宝宝")
        assert entry is not None
        self.assertLess(entry.comprehension_confidence, 1.0)

    def test_thresholds_shrink_with_stage(self) -> None:
        self.assertGreater(PRODUCTION_THRESHOLD["S1"], PRODUCTION_THRESHOLD["S3"])
        self.assertLess(COMPREHENSION_MIN, 1.0)


class TestTokenCount(unittest.TestCase):
    """「宝宝」是一个词，不是两个字——按字计会让 S1 一张嘴就超长。"""

    def test_chinese_word_counts_as_one_token(self) -> None:
        lex = Lexicon(stage="S1")
        _feed(lex)
        for entry in lex.entries.values():
            for _ in range(PRODUCTION_THRESHOLD["S1"]):
                lex.produce(entry.form, day=40)
        self.assertEqual(syntax.count_tokens("宝宝", lex), 1)
        self.assertEqual(syntax.max_tokens("S1"), 1)
        problems = syntax.check_utterance("宝宝", level="L1", stage="S1", lexicon=lex)
        self.assertEqual(problems, [])

    def test_unknown_chars_still_count(self) -> None:
        lex = Lexicon(stage="S1")
        _feed(lex)
        entry = lex.by_form("宝宝")
        assert entry is not None
        for _ in range(8):
            lex.produce("宝宝", day=40)
        # 「宝宝水」里的「水」没学过 → 两个词
        self.assertEqual(syntax.count_tokens("宝宝水", lex), 2)


class TestMilestoneEvidence(unittest.TestCase):
    """里程碑只认 episode 里的证据，不认自述。"""

    @staticmethod
    def _ep(eid: str, ts: str, prim: str, params: dict, utterance: str = "") -> dict:
        return {
            "id": eid,
            "ts": ts,
            "stage": "S1",
            "action": {"primitive": prim, "params": params},
            "language": {"mode": "verbal", "utterance": utterance or None, "syntax_level": "L1"},
        }

    def test_m1_8_needs_two_consecutive_weeks(self) -> None:
        eps = [
            self._ep("ep-1", "2026-02-02T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
            self._ep("ep-2", "2026-02-03T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
            self._ep("ep-3", "2026-02-04T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
            self._ep("ep-4", "2026-02-09T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
            self._ep("ep-5", "2026-02-10T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
            self._ep("ep-6", "2026-02-11T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
        ]
        ev = milestones.eval_m1_8(eps)
        self.assertEqual(ev.status, milestones.PASSED)
        self.assertEqual(ev.value, 2.0)

    def test_m1_8_ignores_weeks_with_few_words(self) -> None:
        eps = [
            self._ep("ep-1", "2026-02-02T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
            self._ep("ep-2", "2026-02-09T09:00:00+08:00", "A1.WORD", {"lexeme": "妈妈"}, "妈妈"),
        ]
        ev = milestones.eval_m1_8(eps)
        self.assertEqual(ev.status, milestones.FAILED)

    def test_m1_6_counts_babble_days(self) -> None:
        eps = []
        for day in range(1, 8):
            for k in range(10):
                eps.append(
                    self._ep(f"ep-{day}-{k}", f"2026-02-{day:02d}T0{k % 9 + 1}:00:00+08:00",
                             "A1.BABBLE", {"syllables": "ba", "reps": 3})
                )
        ev = milestones.eval_m1_6(eps)
        self.assertEqual(ev.status, milestones.PASSED)
        self.assertGreaterEqual(ev.value or 0, 5.0)


if __name__ == "__main__":
    unittest.main()
