"""动作选择。

打分公式（docs/04 通道 2）：

    score(a) = Σ_d drive_d · afford_d(a)
             + β_E·social_gain(a) + β_O·novelty(a) + β_A·other_benefit(a)
             − β_N·risk(a)

特质只从 `Bias` 进来，而 `Bias` 每 episode 由当前 T 现算。
LLM 只提供**先验**（一个分布），真正选谁由这个公式决定；
没有 key 时先验来自 `MockProvider` 的确定性伪分布，`degraded=true`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from persona.body.compiler import Candidate
from persona.cognition.appraisal import History
from persona.cognition.traits import Bias
from persona.types import Appraisal, Drives, Intent, clip01, r6

#: 打分权重（相对大小即偏好强度）
W_DRIVE = 0.9
SOCIAL_GAIN = 1.1
NOVELTY_GAIN = 0.9
OTHER_BENEFIT = 0.5
RISK_AVERSION = 1.3

#: 一天里大约各做多少次（超出就明显变钝）。这不是「配额」，是一天只有 32 格清醒时间
#: 这件事的另一种写法：不会走八小时，也不会咿呀八小时。
DAILY_TARGET: dict[str, int] = {
    "A0.FIXATE": 8,
    "A0.CRY": 3,
    "A0.SUCK": 6,
    "A0.SLEEP": 2,
    "A1.CRAWL": 3,
    "A1.PEEK": 12,
    "A1.POINT": 6,
    "A1.GIVE": 2,
    "A1.TAKE": 2,
    "A1.SIT": 2,
    "A1.BABBLE": 20,
    "A1.WORD": 14,
    "A2.WALK": 24,
    "A2.MANIPULATE": 5,
    "A2.IMITATE": 3,
    "A2.TWO_WORD": 9,
    "A2.TELEGRAPH": 2,
    "A2.SHOW": 5,
    "A2.REFUSE": 1,
    "A2.PRETEND": 6,
    "A2.WAIT": 4,
    "A2.CHOOSE": 6,
}


class PriorProvider(Protocol):
    def prior(self, *, observation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> "Prior": ...


@dataclass
class Prior:
    """LLM 给出的先验分布 + 用量。`degraded` 必须带原因（I8）。"""

    probs: dict[str, float] = field(default_factory=dict)
    model: str = "none"
    tokens_in: int = 0
    tokens_out: int = 0
    degraded: bool = False
    degraded_reason: str | None = None


@dataclass
class Decision:
    primitive: str
    params: dict[str, Any]
    source: str
    scores: dict[str, float]
    shortlist: list[str]
    prior: Prior
    intent: Intent | None = None
    temperature: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "params": dict(self.params),
            "source": self.source,
            "shortlist": list(self.shortlist),
            "scores": {k: r6(v) for k, v in self.scores.items()},
            "prior_degraded": self.prior.degraded,
        }


def score_candidate(
    c: Candidate,
    *,
    bias: Bias,
    drives: Drives,
    appraisal: Appraisal,
    needs: Mapping[str, float] | None = None,
) -> float:
    """单个候选的分数。所有项都由当前状态算出，不含任何常量人格。"""
    afford = c.afford or {}
    drive_term = (
        W_DRIVE
        * (
            drives.energy * float(afford.get("energy", 0.0))
            + drives.stimulation * float(afford.get("stim", 0.0))
            + drives.social * float(afford.get("social", 0.0))
            + drives.competence * float(afford.get("comp", 0.0))
            + drives.safety * float(afford.get("safety", 0.0))
        )
    )
    social_gain = SOCIAL_GAIN * bias.social * float(afford.get("social", 0.0))
    novelty_gain = NOVELTY_GAIN * bias.explore * float(afford.get("stim", 0.0)) * (
        0.4 + 0.6 * appraisal.novelty
    )
    other_benefit = OTHER_BENEFIT * bias.prosocial * float(afford.get("comp", 0.0)) * 0.5
    risk = RISK_AVERSION * bias.risk * appraisal.threat * (1.0 - 0.6 * float(afford.get("safety", 0.0)))

    # 稳态紧迫：不给这些动作一个「需求项」，婴儿就会饿到 1.0 也不哭也不吸。
    # 这是反射层，不是人格层——它与 T 无关，只与身体缺口有关。
    homeo = 0.0
    if needs:
        hunger = clip01(float(needs.get("hunger", 0.0)))
        fatigue = clip01(float(needs.get("fatigue", 0.0)))
        discomfort = clip01(float(needs.get("discomfort", 0.0)))
        # 发声的两个闸门：有没有可练的词、有没有会说的词。放在最前面，
        # 因为后面几种"说话类"原语都要用（早先写在分支后面，S2 一进来就 UnboundLocalError）。
        vocal = clip01(float(needs.get("vocal", 0.0)))
        words = clip01(float(needs.get("vocal_word", 0.0)))
        # 定向反射：有东西刚动起来时，看向它是反射，不是「好奇心」。所以放在这一层，
        # 与 T 无关；否则 30 分钟粒度的世界里它永远抢不过哭声和吮吸。
        orient = clip01(float(needs.get("orient", 0.0)))
        if c.primitive == "A0.CRY":
            homeo = 1.1 * hunger + 0.8 * discomfort
        elif c.primitive == "A0.SUCK":
            # 吮吸会「用旧」：连着吸也吸不出奶的时候，它的安抚力就掉下去，
            # 于是饿着的小孩最终哭出来。没有这一条，吮吸永远是最高分，
            # 哭声就再也不会被别的缺口（比如「她不在」）推动。
            streak = max(0.0, float(needs.get("suck_streak", 0.0)))
            homeo = 1.0 * hunger / (1.0 + 0.6 * streak)
        elif c.primitive == "A0.SLEEP":
            homeo = 1.3 * fatigue
        elif c.primitive == "A0.FIXATE":
            homeo = 3.0 * orient
        elif c.primitive == "A0.TURN_HEAD":
            homeo = 1.0 * orient
        elif c.primitive == "A1.REACH":
            homeo = 0.2 * (hunger + discomfort)
        elif c.primitive == "A1.POINT":
            # 指不是「够不着」，是「给你看」：照料者在、眼前有东西，才想指。
            share = clip01(float(needs.get("share", 0.0)))
            homeo = 0.2 * (hunger + discomfort) + 1.2 * share
        elif c.primitive == "A1.CRAWL":
            # 会爬之后，推力来自「想过去」：醒着、有力气、眼前有新奇的东西。
            # 没有这一项，他就整天坐在原地看——1 岁的孩子不是这样。
            homeo = (
                0.2 * (hunger + discomfort)
                + 0.7 * clip01(drives.stimulation)
                + 0.5 * clip01(appraisal.novelty)
            )
        elif c.primitive == "A2.WALK":
            # 走和爬的动机不一样：爬是「够到那个东西」，走是「想到处去」。
            # 但学步期的孩子不是漫无目的地走：看见一个够不着的东西，
            # 他会朝它过去——「过去」本身就是 S2 最主要的动作（M2.1）。
            homeo = (
                0.2 * (hunger + discomfort)
                + 0.9 * clip01(drives.stimulation)
                + 0.6 * clip01(appraisal.novelty)
                + 2.0 * clip01(float(needs.get("far_target", 0.0)))
                # 学步儿本来就走个不停：走路不需要理由，停下来才需要。
                + 1.0
            )
        elif c.primitive == "A2.MANIPULATE":
            # 摆弄是「我能把它怎么样」：胜任缺口越大越想动它，眼前的东西越新越想试
            homeo = 1.1 * clip01(drives.competence) + 0.5 * clip01(appraisal.novelty)
        elif c.primitive == "A2.IMITATE":
            # 模仿的推力是社会性的：有人在，而且那个动作他刚刚看见过
            homeo = 0.9 * clip01(drives.social) + 0.3 * clip01(float(needs.get("share", 0.0)))
        elif c.primitive == "A2.TWO_WORD":
            # 会说一个词之后，把两个词拼起来是下一步。词库里会说两个以上，才有这股劲。
            homeo = max(homeo, 2.0 * words)
        elif c.primitive == "A2.SHOW":
            # 展示：拿给她看。有东西在手、她在旁边，才会举起来。
            share = clip01(float(needs.get("share", 0.0)))
            homeo = 1.0 * share
        elif c.primitive == "A2.TELEGRAPH":
            homeo = 0.6 * clip01(drives.social)
        elif c.primitive == "A2.REFUSE":
            # 拒绝是「我不要」，不是「我不敢」：安全缺口大的时候不会硬顶
            homeo = 0.5 * clip01(drives.competence) - 0.6 * clip01(drives.safety)
        elif c.primitive == "A2.PRETEND":
            # 假装：手里有玩具、旁边有人看着，才会「喂娃娃吃饭」。
            # 推力来自胜任与刺激——假装是「我能让它变成别的」这件事本身好玩。
            drive = clip01(float(needs.get("pretend_drive", 0.0)))
            homeo = drive * (
                0.9 * clip01(drives.competence) + 0.6 * clip01(drives.stimulation) + 0.4
            )
        elif c.primitive == "A2.WAIT":
            # 等：东西被大人收着。等不是乖，是「抢不到而且抢会被制止」之后学会的
            # ——所以这里给安全缺口一点权重：越觉得安全，越等得住。
            drive = clip01(float(needs.get("wait_drive", 0.0)))
            homeo = drive * (4.2 + 0.8 * clip01(drives.safety))
        elif c.primitive == "A2.CHOOSE":
            # 挑：卡片摆在眼前。偏好是选出来的——选过的会再被选（这种惯性就是
            # docs/03 的 M2.9 要测的「形象选择的稳定性」）。
            # 用**占比**而不是次数：次数会正反馈到爆，占比有上限。
            drive = clip01(float(needs.get("choose_drive", 0.0)))
            target = str(c.params.get("target") or "")
            picked = float(needs.get(f"chose:{target}", 0.0))
            total = max(1.0, float(needs.get("choose_total", 0.0)))
            # 大人把三张卡片举到他面前等着的那一刻，是全天最明确的「要我挑」信号
            homeo = drive * (3.6 + 2.0 * clip01(picked / total))
        # 发声：词库里有没练熟的词就练，有会说的词就说。这不是「学会说话的奖励」，
        # 而是他自己就有这股劲（词库空时这一项恒为 0，所以 S0 不会凭空咿呀）。
        if c.primitive == "A1.BABBLE":
            # 咿呀是练习驱力：有可练的词形时，它会压过看和抓（docs/03 的 M1.6 要求
            # 一天十次以上的重复音节串，靠「顺便练一下」到不了）。
            # 词汇爆发只在 S2 之后发动：积压再多，一岁的孩子也不会一天练三十个词。
            backlog = clip01(float(needs.get("practice_backlog", 0.0)))
            boom = clip01((float(needs.get("age_days", 0.0)) - 330.0) / 30.0)
            boost = 1.0 + boom * 2.6 * backlog
            # 一岁那年音节练习是主线，会说的词还少，不能让它被「说词」挤掉。
            babble_gain = 2.8 if float(needs.get("age_days", 0.0)) < 330.0 else 2.4
            homeo = max(homeo, babble_gain * vocal * boost)
        elif c.primitive == "A1.WORD":
            # 会说的词一多，说的机会就要多：两岁那年一天只说两句，词汇量是长不出来的。
            word_gain = 2.8 if float(needs.get("age_days", 0.0)) >= 330.0 else 1.2
            homeo = max(homeo, word_gain * words)
        # 客体永久性：东西不见了还想去找。这不是「好奇」，是那东西在他心里还活着；
        # 没有这一项，被盖住的摇铃就只是「从世界里消失了」，永远不会有搜索行为。
        occluded = clip01(float(needs.get("occluded", 0.0)))
        if c.primitive == "A1.PEEK":
            # 被盖住的东西会拉着他的手：这不是好奇，是「它还在那儿」。
            # 权重压在咿呀之上——搜索是目标驱动的，咿呀是背景性的。
            homeo = max(homeo, 3.0 * occluded)
        # 陌生人焦虑：不认识的人在，就往照料者那边靠（看她、爬向她）。
        stranger = clip01(float(needs.get("stranger", 0.0)))
        if stranger > 0.0 and str(c.params.get("target") or "") == "mother":
            homeo += 1.5 * stranger
        # 共同注意：刚指过的东西，目光要跟过去。指了不看，就只是挥手。
        # 强度随月龄长起来——跟随指向 9–12 个月才稳定，不是一开始就 100%。
        # 只加在「看」上：跟随是视线跟过去，不是伸手去抓同一个东西。
        # （早先加在所有 target 相同的候选上，结果 REACH/GIVE/TAKE 借光抢走了注视。）
        joint = clip01(float(needs.get("joint", 0.0)))
        if (
            joint > 0.0
            and c.primitive == "A0.FIXATE"
            and str(c.params.get("target") or "") == str(needs.get("joint_target") or "")
        ):
            homeo += float(needs.get("joint_gain", 0.0)) * joint
    score = drive_term + social_gain + novelty_gain + other_benefit + homeo - risk - 0.15 * c.cost
    # 一天的总量是有限的：做什么都有「今天已经做过多少」这一层。
    # 没有它，谁的基础分最高谁就吃掉整天——走路能走八小时，咿呀能咿呀八小时。
    target = DAILY_TARGET.get(c.primitive)
    if target:
        done = max(0.0, float(needs.get(f"today:{c.primitive}", 0.0)))
        score = score / (1.0 + 2.0 * min(1.0, done / float(target)))
    return score


def _softmax(values: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not values:
        return {}
    t = max(0.05, float(temperature))
    m = max(values.values())
    exps = {k: math.exp((v - m) / t) for k, v in values.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def decide(
    candidates: Sequence[Candidate],
    *,
    bias: Bias,
    drives: Drives,
    appraisal: Appraisal,
    provider: PriorProvider | None = None,
    llm_weight: float = 0.45,
    temperature: float = 0.7,
    shortlist_size: int = 5,
    rng: Any = None,
    observation: Mapping[str, Any] | None = None,
    needs: Mapping[str, float] | None = None,
) -> Decision:
    """先打分、再取短名单、可选混入 LLM 先验、最后按温度采样。"""
    if not candidates:
        return Decision(
            primitive="A0.NONE",
            params={},
            source="fallback",
            scores={},
            shortlist=[],
            prior=Prior(degraded=True, degraded_reason="无可用候选动作"),
            temperature=temperature,
        )

    scores = {
        c.primitive: score_candidate(c, bias=bias, drives=drives, appraisal=appraisal, needs=needs)
        for c in candidates
    }
    order = sorted(scores, key=lambda k: scores[k], reverse=True)
    shortlist = order[: max(1, shortlist_size)]
    by_prim = {c.primitive: c for c in candidates}

    probs = _softmax({k: scores[k] for k in shortlist}, temperature)
    prior = Prior(model="none")
    if provider is not None:
        payload = [{"primitive": by_prim[k].primitive, "params": dict(by_prim[k].params)} for k in shortlist]
        prior = provider.prior(observation=observation or {}, candidates=payload)
        # 降级的先验不许参与决策。mock 的「先验」是一条平坦的伪分布：
        # 把它按 0.45 掺进来，等于每条决策都有 45% 是掷骰子——
        # 行为会被整体拉平，任何「学会了的本事」都到不了 100%（M1.7 曾卡在 0.58）。
        if prior.probs and not prior.degraded:
            llm = _softmax({k: float(prior.probs.get(k, 0.0)) for k in shortlist}, 1.0)
            w = clip01(llm_weight)
            probs = {k: (1.0 - w) * probs.get(k, 0.0) + w * llm.get(k, 0.0) for k in shortlist}
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}

    chosen = _sample(probs, rng)
    c = by_prim[chosen]
    return Decision(
        primitive=c.primitive,
        params=dict(c.params),
        source="llm" if (provider is not None and prior.probs and not prior.degraded) else "heuristic",
        scores={k: r6(v) for k, v in scores.items()},
        shortlist=shortlist,
        prior=prior,
        intent=make_intent(c, drives=drives, appraisal=appraisal),
        temperature=temperature,
    )


def _sample(probs: Mapping[str, float], rng: Any) -> str:
    keys = sorted(probs)
    if rng is None:
        return max(keys, key=lambda k: probs[k])
    r = rng.random()
    acc = 0.0
    for k in keys:
        acc += probs[k]
        if r <= acc:
            return k
    return keys[-1]


def make_intent(c: Candidate, *, drives: Drives, appraisal: Appraisal) -> Intent | None:
    """把动作反推成结构化意图。S0 的意图没有词，但结构仍在（这是防泄漏的要点）。"""
    goal = _dominant_drive(drives)
    speech_act = "express" if c.primitive == "A0.CRY" else "act"
    predicate = goal
    if c.primitive == "A0.CRY":
        predicate = str(c.params.get("reason", "discomfort"))
        speech_act = "express"
    referent = c.params.get("target") or c.params.get("entity")
    return Intent(
        goal=goal,
        speech_act=speech_act,
        referent_id=str(referent) if referent else None,
        predicate_type=predicate,
        urgency=r6(clip01(max(drives.energy, drives.safety, drives.social))),
    )


def _dominant_drive(d: Drives) -> str:
    pairs = [
        ("satiety", d.energy),
        ("stimulation", d.stimulation),
        ("affiliation", d.social),
        ("competence", d.competence),
        ("safety", d.safety),
    ]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[0][0]
