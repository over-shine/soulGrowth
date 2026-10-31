"""动作编译器：把「想做什么」翻译成「身体此刻真能做到什么」。

这条链路是「不越界」的技术保证（docs/08）：

    意图 → 能力表（按阶段） → 前置条件 → 参数校验 → 话语实现 → 保真度 → 降级链 → 再校验

失败不是异常，是**结果**：`Compiled.denials` 里记下拒绝码，episode 照常落盘。
同一 episode 连续 3 次 SANDBOX_DENIED 或一天 ≥20 次拒绝会触发安全事件。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from persona.body.perception import Bundle
from persona.body.primitives import Primitive, Registry
from persona.body.state import BodyState
from persona.errors import CompileError
from persona.language import syntax
from persona.language.lexicon import Lexicon
from persona.types import Intent
from persona.world.objects import WorldState

#: 拒绝码（docs/08）
STAGE_LOCKED = "STAGE_LOCKED"
PRECONDITION_FAILED = "PRECONDITION_FAILED"
ENERGY_LOW = "ENERGY_LOW"
SANDBOX_DENIED = "SANDBOX_DENIED"
SCHEMA_INVALID = "SCHEMA_INVALID"
ADAPTER_MISSING = "ADAPTER_MISSING"

DENIAL_CODES = (STAGE_LOCKED, PRECONDITION_FAILED, ENERGY_LOW, SANDBOX_DENIED, SCHEMA_INVALID, ADAPTER_MISSING)

#: 意图 → 原语的保真度（docs/08 示例「要杯子」的一般化）。
#: 数值表示「用这个原语能把这个意图表达得多完整」，用于在阶段受限时挑最优降级。
FIDELITY: dict[str, float] = {
    "A4.COMPOUND": 0.95,
    "A3.SENTENCE": 0.95,
    "A2.TELEGRAPH": 0.85,
    "A2.TWO_WORD": 0.80,
    "A2.SHOW": 0.60,
    "A2.PRETEND": 0.55,
    "A2.CHOOSE": 0.50,
    "A2.WAIT": 0.50,
    "A1.WORD": 0.65,
    "A1.POINT": 0.50,
    "A1.REACH": 0.40,
    "A1.GIVE": 0.35,
    "A1.BABBLE": 0.20,
    "A0.CRY": 0.25,
    "A0.SUCK": 0.15,
    "A0.FIXATE": 0.10,
    "A0.NONE": 0.00,
}

#: 单向降级链（docs/08）。到了尽头就静默——**必须终止**。
FALLBACK_CHAINS: dict[str, tuple[str, ...]] = {
    "A4.COMPOUND": ("A3.SENTENCE", "A2.TWO_WORD", "A1.WORD", "A0.NONE"),
    "A3.SENTENCE": ("A2.TWO_WORD", "A1.WORD", "A0.NONE"),
    "A2.TWO_WORD": ("A1.WORD", "A1.POINT", "A1.REACH", "A0.CRY", "A0.NONE"),
    "A2.TELEGRAPH": ("A2.TWO_WORD", "A1.WORD", "A0.NONE"),
    "A1.WORD": ("A1.POINT", "A1.REACH", "A1.BABBLE", "A0.CRY", "A0.NONE"),
    "A1.POINT": ("A1.REACH", "A0.CRY", "A0.NONE"),
    "A1.REACH": ("A0.CRY", "A0.NONE"),
    "A0.CRY": ("A0.SUCK", "A0.NONE"),
}

MIN_FIDELITY = 0.35

#: 双词句里的「谓词」候选：孩子说的两词句，一半是东西，一半是对它做的事/给的指向。
#: 只有在词库里已经会说的那几个才用得上（说不出就是不会说，不拿别的词顶替）。
PREDICATE_FORMS: tuple[str, ...] = ("抱", "给", "要", "看", "走", "吃", "喝", "拿", "开", "来")


@dataclass
class Candidate:
    """一个「此刻做得到」的动作选项。打分在 cognition/policy，这里只列事实。"""

    primitive: str
    params: dict[str, Any] = field(default_factory=dict)
    afford: dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    fidelity: float = 1.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "params": dict(self.params),
            "cost": round(self.cost, 6),
            "fidelity": round(self.fidelity, 6),
        }


@dataclass
class Compiled:
    """编译结果。`denials` 非空时 `primitive` 为 A0.NONE。"""

    primitive: str = "A0.NONE"
    params: dict[str, Any] = field(default_factory=dict)
    intent: Intent | None = None
    utterance: str | None = None
    mode: str = "pre_verbal"
    level: str = "L0"
    fidelity: float = 0.0
    degraded: bool = False
    chain: tuple[str, ...] = ()
    denials: tuple[str, ...] = ()

    def as_action_kwargs(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "params": dict(self.params),
            "intent": self.intent,
            "fidelity": self.fidelity,
            "degraded": self.degraded,
            "chain": self.chain,
        }


class Compiler:
    """编译入口。无状态，可复用。"""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    # -- 候选 ---------------------------------------------------------------
    def candidates(
        self,
        bundle: Bundle,
        body: BodyState,
        *,
        stage: str,
        world: WorldState,
        lexicon: Lexicon,
        limit: int = 12,
        social: float = 0.0,
    ) -> list[Candidate]:
        """当前阶段「存在且可用」的原语。未解锁的根本不在列表里（D1）。"""
        out: list[Candidate] = []
        for prim in self.registry.for_stage(stage):
            if prim.adapter == "none":
                continue
            if prim.id == "A2.CHOOSE":
                # 选项要摆在桌上才有得挑：由 _with_choices 按卡片逐个生成，
                # 不能在这里用「注意力里最显眼那个」凑一个候选（那就成了替他挑）。
                continue
            ok, _ = self._preconditions(prim, world=world, bundle=bundle, body=body, lexicon=lexicon)
            if not ok and prim.id == "A0.GRASP" and self._touch_target(world):
                # 抓握反射由触觉引发：手边有东西就算达成前置
                ok = True
            if not ok:
                continue
            params = self._suggest_params(
                prim, bundle=bundle, body=body, world=world, lexicon=lexicon, social=social
            )
            if self.registry.validate_params(prim, params):
                continue
            out.append(
                Candidate(
                    primitive=prim.id,
                    params=params,
                    afford=dict(prim.afford),
                    cost=prim.cost,
                    fidelity=FIDELITY.get(prim.id, 0.3),
                )
            )
        out.sort(key=lambda c: (c.fidelity, -c.cost), reverse=True)
        out = out[:limit]
        out = self._with_focus(out, bundle=bundle, stage=stage, world=world)
        out = self._with_walk(out, stage=stage)
        out = self._with_choices(out, stage=stage, world=world)
        return out

    def _with_focus(
        self,
        out: list[Candidate],
        *,
        bundle: Bundle,
        stage: str,
        world: WorldState,
    ) -> list[Candidate]:
        """大人指了某样东西时，即使它不够显眼，也要有一个「看向它」的候选。

        跟随指向的意思就是：别人的注视**能让**一个原本不显眼的东西进入我的注意。
        注意力上限只有 2–3 个对象，光靠显著度排序，被指的那个常常挤不进来。
        """
        focus = str(world.focus_target or "")
        if not focus or world.obj(focus) is None:
            return out
        if any(c.primitive == "A0.FIXATE" and str(c.params.get("target") or "") == focus for c in out):
            return out
        if "A0.FIXATE" not in {p.id for p in self.registry.for_stage(stage)}:
            return out
        prim = self.registry.get("A0.FIXATE")
        params = prim.default_params()
        params["target"] = focus
        if self.registry.validate_params(prim, params):
            return out
        return out + [
            Candidate(
                primitive=prim.id,
                params=params,
                afford=dict(prim.afford),
                cost=prim.cost,
                fidelity=FIDELITY.get(prim.id, 0.3),
            )
        ]

    def _with_walk(
        self,
        out: list[Candidate],
        *,
        stage: str,
    ) -> list[Candidate]:
        """会走之后，「走过去」永远是眼前可做的事之一。

        它不是被某个目标临时召唤出来的动作：房间那头有东西没东西，他都走。
        缺了这条，走路会在自选单里排到十几名开外，一个学步儿整年不挪窝。
        """
        if stage not in ("S2", "S3", "S4", "S5", "S6"):
            return out
        if any(c.primitive == "A2.WALK" for c in out):
            return out
        if "A2.WALK" not in {p.id for p in self.registry.for_stage(stage)}:
            return out
        prim = self.registry.get("A2.WALK")
        params = prim.default_params()
        params.setdefault("dir", "toward")
        params.setdefault("speed", 1)
        params.setdefault("dist_m", 1.0)
        if self.registry.validate_params(prim, params):
            return out
        return out + [
            Candidate(
                primitive=prim.id,
                params=params,
                afford=dict(prim.afford),
                cost=prim.cost,
                fidelity=FIDELITY.get(prim.id, 0.3),
            )
        ]

    def _with_choices(
        self,
        out: list[Candidate],
        *,
        stage: str,
        world: WorldState,
    ) -> list[Candidate]:
        """「挑一个」必须真的把每个选项摆出来。

        只给一个候选等于替他挑好了：那样测出来的不是偏好，是我的默认值。
        """
        if "A2.CHOOSE" not in {p.id for p in self.registry.for_stage(stage)}:
            return out
        if not getattr(world, "shown_cards", False):
            # 卡片没举起来的时候，「挑」不是一个可做的动作。
            return out
        prim = self.registry.get("A2.CHOOSE")
        extra: list[Candidate] = []
        for obj in world.visible_objects():
            if "picture" not in obj.tags:
                continue
            if not _reachable(obj, world):
                # 摆在桌上够不着的卡片，挑不了：那是看见，不是选择。
                continue
            params = prim.default_params()
            params["target"] = obj.id
            if self.registry.validate_params(prim, params):
                continue
            extra.append(
                Candidate(
                    primitive=prim.id,
                    params=params,
                    afford=dict(prim.afford),
                    cost=prim.cost,
                    fidelity=FIDELITY.get(prim.id, 0.3),
                )
            )
        if len(extra) < 2:
            return out
        have = {str(c.params.get("target") or "") for c in out if c.primitive == "A2.CHOOSE"}
        extra = [c for c in extra if str(c.params.get("target") or "") not in have]
        return out + extra[:3]

    def _suggest_params(
        self,
        prim: Primitive,
        *,
        bundle: Bundle,
        body: BodyState,
        world: WorldState,
        lexicon: Lexicon | None = None,
        social: float = 0.0,
    ) -> dict[str, Any]:
        """给出**合法**的默认参数。粗但诚实：不会凭空编造对象 id。"""
        params: dict[str, Any] = prim.default_params()
        salient = bundle.salient_ids()
        for key, spec in prim.params.items():
            t = spec.get("type")
            if t == "object_id" and key not in params:
                if prim.id == "A1.PEEK":
                    # 去找的东西此刻**看不见**（被盖住了），所以不能从 salient 里挑，
                    # 只能从「世界记得它被盖住」里挑。这正是客体永久性的意思：
                    # 对象不在感知里，但还在他心里。
                    hidden = [oid for oid, covered in sorted(world.occluded.items()) if covered]
                    if hidden:
                        params[key] = hidden[0]
                    continue
                if prim.id == "A0.FIXATE" and world.focus_target in salient:
                    # 共同注意：刚指过的东西，看它才有「跟过去」这回事。
                    params[key] = str(world.focus_target)
                    continue
                if (
                    world.stranger_present
                    and world.caregiver_present
                    and "mother" in salient
                    and prim.id in ("A0.FIXATE", "A1.CRAWL")
                ):
                    # 社交参照：不认识的人在的时候，看/靠近的是照料者，而不是那个陌生人。
                    # 这是注意的偏置，不是「他喜欢谁」——陌生面孔本身更显眼，
                    # 没有这条偏置，他反而会一直盯着陌生人（M1.4 就无从测起）。
                    params[key] = "mother"
                    continue
                pick = self._pick_target(salient, world, reachable_only="target_reachable" in prim.pre)
                if pick:
                    params[key] = pick
            elif t == "entity" and key not in params:
                mother = world.obj("mother")
                if mother is not None and world.caregiver_present:
                    params[key] = "mother"
            elif t == "syllable_string" and key not in params:
                # 咿呀练的是他**真听见过**的那个词的前两个音节，不是随便一个 "ba"
                practice = lexicon.practice_form() if lexicon is not None else None
                params[key] = practice.form[:2] if practice is not None else "ba"
            elif t == "lexeme" and key not in params:
                # 会说的词轮流用：总说同一个词，词汇量就永远停在 1（M2.2 要两周五十个）
                words = _productive_by_exposure(lexicon)
                if words:
                    k = int(world.count("words_said")) if world is not None else 0
                    params[key] = words[k % len(words)]
                else:
                    params[key] = ""
            elif t == "primitive_id" and key not in params:
                params[key] = "A0.FIXATE"
            elif t == "string" and key not in params:
                params[key] = "explore"
            elif t == "short_text" and key not in params:
                params[key] = "…"
            elif t == "wh_question" and key not in params:
                params[key] = "what"
            elif t == "clause_list" and key not in params:
                params[key] = ["a"]
        # 哭必须有理由：从驱力缺口里挑最急的
        if prim.id == "A0.CRY" and "reason" not in params:
            params["reason"] = self._cry_reason(body, world=world, social=social)
        # 两个词得是**两个不同的词**，而且都得会说。
        # 而且不能是随便两个词：孩子说的是「球球滚」「妈妈抱」，
        # 前一个是眼前的东西，后一个是对它做的事。轮着念词表不算说话。
        if prim.id == "A2.TWO_WORD":
            words = _productive_by_exposure(lexicon)
            a, b = "", ""
            focused = str(getattr(world, "goal_focus", "") or "") if world is not None else ""
            if focused and lexicon is not None:
                entry = lexicon.by_form(focused)
                if entry is not None and entry.status == "productive":
                    a = entry.form
            if not a and words:
                k = int(world.count("two_word_actions")) if world is not None else 0
                a = words[k % len(words)]
            preds = [w for w in PREDICATE_FORMS if lexicon is not None and w in words]
            if preds:
                b = preds[int(world.count("two_word_actions")) % len(preds)] if world is not None else preds[0]
            if not b or b == a:
                rest = [w for w in words if w != a]
                if rest:
                    k = int(world.count("two_word_actions")) if world is not None else 0
                    b = rest[k % len(rest)]
            if a and b and a != b:
                params["a"], params["b"] = a, b
            else:
                params.pop("a", None)
                params.pop("b", None)
        # 摆弄：同一个玩具能做的事不一样，别拿「转」对付积木
        if prim.id == "A2.MANIPULATE":
            op = str(params.get("op") or "")
            target = str(params.get("target") or "")
            if not op or op == "push":
                params["op"] = {
                    "blocks": "stack",
                    "cup": "insert",
                    "ball": "push",
                    "mirror": "turn",
                    "book": "turn",
                }.get(target, "push")
        # 模仿：她刚在做什么，就照着做什么。S2 的模仿不需要理解意图，只需要动作对应
        if prim.id == "A2.IMITATE":
            params.setdefault("agent", "mother")
            params.setdefault("action", "A1.POINT")
        # 展示：拿给她看，看的人只能是照护者
        if prim.id == "A2.SHOW":
            params.setdefault("to", "mother")
        # 假装：拿一个玩具当真的用（喂娃娃、哄娃娃睡）。道具得在手边。
        if prim.id == "A2.PRETEND":
            if "target" not in params:
                toy = self._toy_target(bundle, world)
                if toy:
                    params["target"] = toy
            scripts = ("喂饭", "哄睡", "做饭", "打电话")
            k = int(world.count("pretend_actions")) if world is not None else 0
            params["script"] = scripts[k % len(scripts)]
        # 等：想要的东西被大人收着，先不抢——这就是延迟满足那个动作
        if prim.id == "A2.WAIT":
            held = str(getattr(world, "withheld", "") or "") if world is not None else ""
            if held:
                params["target"] = held
            else:
                params.pop("target", None)
        # 抓握是**反射**：手掌被碰到就会握，不需要那东西先进注意力
        if prim.id == "A0.GRASP" and "target" not in params:
            touched = self._touch_target(world)
            if touched:
                params["target"] = touched
        return params

    @staticmethod
    def _touch_target(world: WorldState) -> str | None:
        """在手边（0.5 米内）且可触及的东西。反射的触发条件，不是注意力。"""
        best: tuple[float, str] | None = None
        for obj in world.objects.values():
            if not obj.reachable:
                continue
            dist = (obj.pos[0] ** 2 + (obj.pos[1] + 0.18) ** 2 + (obj.pos[2] - 0.3) ** 2) ** 0.5
            if dist <= 0.5 and (best is None or dist < best[0]):
                best = (dist, obj.id)
        return best[1] if best else None

    def _toy_target(self, bundle: Bundle, world: WorldState) -> str | None:
        """手边可以拿来当道具的东西：先看在注意力里的，再看够得着的。"""
        for oid in bundle.salient_ids():
            obj = world.obj(oid)
            if obj is not None and ("toy" in obj.tags or obj.kind == "toy"):
                return oid
        cands = [
            obj
            for obj in world.objects.values()
            if ("toy" in obj.tags or obj.kind == "toy") and _reachable(obj, world)
        ]
        if not cands:
            return None
        cands.sort(key=lambda o: o.id)
        return cands[0].id

    @staticmethod
    def _pick_target(
        salient: list[str], world: WorldState, reachable_only: bool = False
    ) -> str | None:
        """看东西不需要够得着；够东西才需要。salient 已按显著度降序。"""
        live = [oid for oid in salient if world.obj(oid) is not None]
        if not reachable_only:
            return live[0] if live else None
        for oid in live:
            obj = world.obj(oid)
            if obj is not None and obj.reachable:
                return oid
        return live[0] if live else None

    @staticmethod
    def _cry_reason(body: BodyState, *, world: WorldState | None = None, social: float = 0.0) -> str:
        """哭声的**表达**理由：从当前缺口里挑最急的那个。

        孤独不是「她不在」这个事实，而是社交缺口大到盖过其他缺口时才会被哭出来：
        所以这里看的是驱力，不是在场与否。饿着的时候，哭出来的仍然是饿。
        """
        if social >= 0.70 and body.hunger < 0.65 and (world is None or not world.caregiver_present):
            return "lonely"
        if body.hunger >= 0.65:
            return "hunger"
        if body.fatigue >= 0.7:
            return "sleepy"
        if body.integrity < 0.95:
            return "discomfort"
        return "discomfort"

    def _preconditions(
        self,
        prim: Primitive,
        *,
        world: WorldState,
        bundle: Bundle | None,
        body: BodyState,
        lexicon: Lexicon,
    ) -> tuple[bool, str]:
        for code in prim.pre:
            if code == "target_reachable":
                ids = bundle.salient_ids() if bundle is not None else [o.id for o in world.objects]
                if not any(
                    (lambda o: o is not None and _reachable(o, world))(world.obj(oid))
                    for oid in ids
                ):
                    return False, PRECONDITION_FAILED
            elif code == "caregiver_present":
                if not world.caregiver_present:
                    return False, PRECONDITION_FAILED
            elif code == "lexicon:productive":
                if not lexicon.productive_forms():
                    return False, PRECONDITION_FAILED
            else:
                return False, PRECONDITION_FAILED
        if prim.id == "A0.SLEEP" and not body.tired:
            return False, ENERGY_LOW
        return True, ""

    # -- 编译 ---------------------------------------------------------------
    def compile(
        self,
        primitive: str,
        params: dict[str, Any],
        *,
        stage: str,
        body: BodyState,
        world: WorldState,
        lexicon: Lexicon,
        intent: Intent | None = None,
        bundle: Bundle | None = None,
        social: float = 0.0,
    ) -> Compiled:
        """校验 → 实现话语 → 降级。返回的 primitive 一定「该阶段存在」。"""
        chain: list[str] = []
        current = primitive
        denials: list[str] = []

        for _ in range(8):  # 链再长也必须终止
            prim = self.registry._items.get(current)  # noqa: SLF001 - 需区分「不存在」与「未解锁」
            if prim is None:
                denials.append(STAGE_LOCKED)
                nxt = self._next_in_chain(current, tried=chain)
                if nxt is None:
                    break
                chain.append(current)
                current = nxt
                continue
            if prim.stage_index() > _stage_index(stage):
                denials.append(STAGE_LOCKED)
                nxt = self._next_in_chain(current, tried=chain)
                if nxt is None:
                    break
                chain.append(current)
                current = nxt
                continue

            p = dict(params)
            if prim.id == "A0.CRY" and "reason" not in p:
                # 哭声的「理由」是表达层的事，由驱力现算，不由调用方随口指定
                p["reason"] = self._cry_reason(body, world=world, social=social)
            problems = self.registry.validate_params(prim, p)
            if problems:
                denials.append(SCHEMA_INVALID)
                nxt = self._next_in_chain(current, tried=chain)
                if nxt is None:
                    break
                chain.append(current)
                current = nxt
                continue

            ok, code = self._preconditions(prim, world=world, bundle=bundle, body=body, lexicon=lexicon)
            if not ok and prim.id == "A0.GRASP" and p.get("target") and world.obj(str(p["target"])) is not None:
                ok = True  # 反射链：手边的东西不需要先进入注意力
            if not ok and code == ENERGY_LOW:
                denials.append(ENERGY_LOW)
                nxt = self._next_in_chain(current, tried=chain)
                chain.append(current)
                if nxt is None:
                    return Compiled(primitive="A0.NONE", denials=tuple(denials), degraded=True, chain=tuple(chain))
                current = nxt
                continue

            if not ok:
                # 前置不满足：这是「拒绝」，不是「可执行」。降级链继续走（docs/08 I4）
                denials.append(code or PRECONDITION_FAILED)
                nxt = self._next_in_chain(current, tried=chain)
                chain.append(current)
                if nxt is None:
                    break
                current = nxt
                continue

            utterance, mode, level, sproblems = self._render(prim, p, intent=intent, stage=stage, lexicon=lexicon)
            if sproblems:
                denials.append(SCHEMA_INVALID)
                nxt = self._next_in_chain(current, tried=chain)
                if nxt is None:
                    break
                chain.append(current)
                current = nxt
                continue

            fidelity = FIDELITY.get(prim.id, 0.3)
            return Compiled(
                primitive=prim.id,
                params=p,
                intent=intent,
                utterance=utterance,
                mode=mode,
                level=level,
                fidelity=fidelity,
                degraded=bool(chain),
                chain=tuple(chain),
                denials=tuple(denials),
            )

        return Compiled(
            primitive="A0.NONE",
            denials=tuple(denials) or (SCHEMA_INVALID,),
            degraded=bool(chain),
            chain=tuple(chain),
        )

    def _next_in_chain(self, primitive: str, *, tried: Iterable[str] = ()) -> str | None:
        """沿单向链找下一个**还没试过**的原语。链的尽头返回 None。"""
        seen = set(tried)
        for nxt in FALLBACK_CHAINS.get(primitive, ()):
            if nxt not in seen:
                return nxt
        return None

    def _render(
        self,
        prim: Primitive,
        params: dict[str, Any],
        *,
        intent: Intent | None,
        stage: str,
        lexicon: Lexicon,
    ) -> tuple[str | None, str, str, list[str]]:
        """把动作的言语部分实现出来。S0 一律无词。"""
        if stage == "S0":
            mode = "cry" if prim.id == "A0.CRY" else "pre_verbal"
            return None, mode, "L0", []
        if prim.id == "A1.WORD":
            # 说哪个词由他自己决定（params.lexeme），不是由「意图里有什么」猜出来的
            form = str(params.get("lexeme") or "")
            entry = lexicon.by_form(form)
            if entry is None or entry.status != "productive":
                return None, "verbal", "L0", [f"{form!r} 还不会说"]
            return form, "verbal", "L1", []
        if prim.id == "A2.TWO_WORD":
            # 双词不是「把两个词拼起来」那么随便：两个词都得真的会说，
            # 而且第二个词是**新信息**（孩子说的是「宝宝 水」，不是「宝宝 宝宝」）。
            a = str(params.get("a") or "")
            b = str(params.get("b") or "")
            if not a or not b or a == b:
                return None, "verbal", "L0", ["还凑不出两个词"]
            for form in (a, b):
                entry = lexicon.by_form(form)
                if entry is None or entry.status != "productive":
                    return None, "verbal", "L0", [f"{form!r} 还不会说"]
            return f"{a}{b}", "verbal", "L2", []
        if prim.id in ("A2.TELEGRAPH", "A3.SENTENCE", "A4.COMPOUND", "A3.ASK"):
            level = {"A1.WORD": "L1", "A2.TWO_WORD": "L2", "A2.TELEGRAPH": "L2"}.get(prim.id, "L3")
            if intent is None:
                return None, "verbal", "L0", ["缺少结构化意图"]
            try:
                realized = syntax.realize(intent, level=level, stage=stage, lexicon=lexicon)
            except CompileError as exc:
                return None, "verbal", "L0", [str(exc)]
            return realized.utterance, realized.mode, realized.level, []
        if prim.id == "A1.BABBLE":
            return None, "babble", "L0", []
        return None, "pre_verbal", "L0", []


def _stage_index(stage: str) -> int:
    from persona.types import STAGES

    return STAGES.index(stage)


def _reachable(obj: Any, world: WorldState) -> bool:
    """这个对象此刻够不够得着。

    「够得着」是**身体和东西之间的距离**，不是东西的属性：球一直躺在
    1.3 米外的地板上，S1 的他够不着，会走的他走过去就够得着了。
    """
    if getattr(world, "withheld", "") == getattr(obj, "id", None):
        # 东西被她收在手里：不是「够不着」，是「现在不给」。要拿只能等。
        return False
    if obj.reachable:
        return True
    bx, by, bz = world.baby_pos
    ox, oy, oz = obj.pos
    return math.dist((bx, by, bz), (ox, oy, oz)) <= 0.5


def _productive_by_exposure(lexicon: Lexicon | None) -> list[str]:
    """会说的词，按他听过的次数排序——先会说的一定是最熟的那个。"""
    if lexicon is None:
        return []
    pool = lexicon.productive()
    pool.sort(key=lambda e: (e.exposures, e.produced_count), reverse=True)
    return [e.form for e in pool]
