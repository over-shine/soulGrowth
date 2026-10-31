"""里程碑判定。

原则（docs/10）：只认**可数的行为**和**可追溯的证据**，不认模型自述。

每个 evaluator 都在 `evaluator` 字段里写明它是「直接指标」还是「代理指标」
（v0 的代理指标一律标 `proxy:` 前缀，宁可疑也不装作精确）。
证据不足就返回 `insufficient`——**不许**用「看起来像」蒙混过关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from persona.types import r6

#: 评估状态
PASSED = "passed"
FAILED = "failed"
INSUFFICIENT = "insufficient"

#: 社会性微笑从第 6 周（42 天）起才计入——docs/03 说的是 6–8 周
SMILE_FIRST_DAY = 42
#: 跟随指向只在中后段算「稳定」：真月龄是 9–12 个月，S1 共 300 天，
#: 这里取出生后 60 天（约 S1 的十分之一）作为门槛，再早的偶尔跟上是运气。
FOLLOW_MIN_AGE = 60


@dataclass(frozen=True)
class Milestone:
    id: str
    stage: str
    name: str
    description: str
    threshold: float
    unit: str


@dataclass
class Evaluation:
    milestone_id: str
    stage: str
    status: str
    value: float | None
    threshold: float
    evidence: list[str] = field(default_factory=list)
    evaluator: str = "direct"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.milestone_id,
            "stage": self.stage,
            "status": self.status,
            "value": None if self.value is None else r6(self.value),
            "threshold": r6(self.threshold),
            "evidence": list(self.evidence),
            "evaluator": self.evaluator,
        }


#: S0 的七个里程碑（docs/03）
S0_MILESTONES: tuple[Milestone, ...] = (
    Milestone("M0.1", "S0", "视觉定向", "2 秒内把注视点移到新出现的目标上", 0.60, "比例"),
    Milestone("M0.2", "S0", "视觉追踪", "以约 10°/s 横穿视野时跟随 ≥50% 行程", 0.50, "行程比例"),
    Milestone("M0.3", "S0", "习惯化", "第 5 次注视时长 ≤ 第 1 次的 60%", 0.60, "时长比"),
    Milestone("M0.4", "S0", "定向反射", "突然的声音/光亮后 2 秒内动作或唤醒变化", 0.60, "比例"),
    Milestone("M0.5", "S0", "抓握反射", "被放入手心后握持 ≥2 秒", 2.00, "秒"),
    Milestone("M0.6", "S0", "哭声分化", "饥饿/不适/困倦三类各 5 次且效应量 d≥0.8", 5.00, "次数"),
    Milestone("M0.7", "S0", "认照料者", "第 3 周起对 mother 声音的反应比陌生人强 ≥1σ", 1.00, "σ"),
)

S1_MILESTONES: tuple[Milestone, ...] = (
    Milestone("M1.1", "S1", "社会性微笑", "6–8 周起每天 ≥5 次对人笑", 5.00, "次/天"),
    Milestone("M1.2", "S1", "客体永久性", "被遮住后仍去寻找，成功率 ≥60%", 0.60, "比例"),
    # 30 分钟一格的世界测不到「停下 2 秒」，测得到的是「那天在床沿停住了」。
    # 判据改成天数，单位写清楚，不拿假秒数糊弄。
    Milestone("M1.3", "S1", "深度知觉", "在床沿停住（至少两天）", 2.00, "天"),
    Milestone("M1.4", "S1", "陌生人焦虑", "陌生人靠近时向照料者移动 ≥基线 +2σ", 2.00, "σ"),
    # 一格就是 30 分钟，所以 docs 里「负性行为 >60 秒」在这里等价于「连续两格」。
    Milestone("M1.5", "S1", "分离焦虑", "母亲离开后负性行为连续 ≥2 格（30 分钟粒度）", 2.00, "格"),
    Milestone("M1.6", "S1", "咿呀", "重复音节串 ≥10 次/天，连续 5 天", 5.00, "天"),
    Milestone("M1.7", "S1", "共同注意", "跟随指向看同一物体 ≥70%", 0.70, "比例"),
    Milestone("M1.8", "S1", "第一个词", "同一词形 ≥3 次/周，连续 2 周", 2.00, "周"),
)

S2_MILESTONES: tuple[Milestone, ...] = (
    # 「走 10 步」在一格 30 分钟里是连续位移；这里数的是「这一格走了 ≥10 步」
    # 的天数——步数是行为计数，不是他自述「我会走了」。
    Milestone("M2.1", "S2", "独立行走", "连走 ≥10 步（连续三天）", 3.00, "天"),
    Milestone("M2.2", "S2", "词汇爆发", "两周内新增 ≥50 个会说出的词", 50.00, "词"),
    Milestone("M2.3", "S2", "双词组合", "≥20 种不同的双词组合", 20.00, "种"),
    Milestone("M2.4", "S2", "自我识别", "在镜子前认出自己（触碰镜像 ≥5 次）", 5.00, "次"),
    Milestone("M2.5", "S2", "假装游戏", "假装动作 ≥3 次", 3.00, "次"),
    Milestone("M2.6", "S2", "延迟满足", "等待 30 秒以上的比例 ≥60%", 0.60, "比例"),
    Milestone("M2.7", "S2", "自我安抚", "2 分钟内自平复 ≥40%", 0.40, "比例"),
    Milestone("M2.8", "S2", "代词我", "说出「我」≥10 次", 10.00, "次"),
    Milestone("M2.9", "S2", "首次形象选择", "自选形象 top1 ≥60%", 0.60, "比例"),
)

STAGE_MILESTONES: dict[str, tuple[Milestone, ...]] = {
    "S0": S0_MILESTONES,
    "S1": S1_MILESTONES,
    "S2": S2_MILESTONES,
}


def milestones_of(stage: str) -> tuple[Milestone, ...]:
    return STAGE_MILESTONES.get(stage, ())


# -- 评估器 ----------------------------------------------------------------
def _episodes(eps: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [e for e in eps if isinstance(e, Mapping)]


def _first_target_episodes(eps: Sequence[Mapping[str, Any]]) -> tuple[int, list[str]]:
    hits = 0
    evidence: list[str] = []
    for e in eps:
        act = e.get("action") or {}
        if act.get("primitive") == "A0.FIXATE" and (act.get("params") or {}).get("target"):
            hits += 1
            evidence.append(str(e.get("id")))
    return hits, evidence


def _prim(e: Mapping[str, Any]) -> str:
    return str((e.get("action") or {}).get("primitive", ""))


def _params(e: Mapping[str, Any]) -> Mapping[str, Any]:
    return (e.get("action") or {}).get("params") or {}


def _salient(e: Mapping[str, Any]) -> list[str]:
    obs = e.get("observation") or {}
    return [str(x) for x in (obs.get("salient") or [])]


def _novelty(e: Mapping[str, Any]) -> dict[str, float]:
    obs = e.get("observation") or {}
    raw = obs.get("novelty") or {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _onsets(e: Mapping[str, Any]) -> list[str]:
    obs = e.get("observation") or {}
    return [str(x) for x in (obs.get("motion_onsets") or [])]


def eval_m0_1(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """视觉定向：凡是「看向某物」的 episode，看的是不是真的在视野里的东西。

    这是精度指标而不是频率指标——婴儿大部分时间在吮吸和睡觉，要求 60% 的
    episode 都在看东西是不合理的；「看的时候看对了」才是这个里程碑的意思。
    """
    eps = _episodes(eps)
    fix = [e for e in eps if _prim(e) == "A0.FIXATE"]
    if len(fix) < 20:
        return Evaluation("M0.1", "S0", INSUFFICIENT, None, 0.60, [], "proxy:注视精度")
    hits = [e for e in fix if str(_params(e).get("target", "")) in _salient(e)]
    value = len(hits) / len(fix)
    status = PASSED if value >= 0.60 else FAILED
    return Evaluation(
        "M0.1", "S0", status, value, 0.60, [str(e.get("id")) for e in hits[:20]], "proxy:注视精度"
    )


def _passes(eps: Sequence[Mapping[str, Any]], target: str, min_len: int = 2) -> list[list[Mapping[str, Any]]]:
    """把「目标连续可见」的相邻 episode 切成若干段（模拟一次横穿视野）。"""
    out: list[list[Mapping[str, Any]]] = []
    run: list[Mapping[str, Any]] = []
    for e in eps:
        if target in _salient(e):
            run.append(e)
        else:
            if len(run) >= min_len:
                out.append(run)
            run = []
    if len(run) >= min_len:
        out.append(run)
    return out


def eval_m0_2(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """视觉追踪：东西动的时候比不动的时候更常被看——而且这个差距是后来才长出来的。

    30 分钟一条的粒度测不到「10°/s 横穿视野」，能测的是它的行为痕迹：床铃是
    发条玩具，动 30 分钟、停 60 分钟。于是可以问一个**对比**问题——同样是床铃，
    「它正在动」的那些帧里被注视的比例，比「它停着」的帧高多少？

    用对比而不是绝对比例，是因为绝对比例会被这一天哭了多久、吮吸了多少次
    整体压低或抬高；对比衡量的是「有没有专门冲着动的东西去看」。

    docs/03 说追踪是 8–14 天才稳定的，所以还要求这个差距**后来比一开始大**
    （晚期 ≥14 天 vs 第 1 周，z≥1），且晚期差距 ≥0.30。
    """
    frames = _episodes(eps)

    def group(lo: int, hi: int | None) -> list[Mapping[str, Any]]:
        return [
            e
            for e in frames
            if int(e.get("age_days", 0)) >= lo and (hi is None or int(e.get("age_days", 0)) <= hi)
        ]

    def lift(group_: Sequence[Mapping[str, Any]]) -> tuple[float, float] | None:
        # (差值, 方差)
        on = [e for e in group_ if "mobile" in _onsets(e)]
        off = [e for e in group_ if "mobile" not in _onsets(e)]
        if len(on) < 20 or len(off) < 20:
            return None

        def hit(e: Mapping[str, Any]) -> bool:
            return _prim(e) == "A0.FIXATE" and str(_params(e).get("target", "")) == "mobile"

        p_on = sum(1 for e in on if hit(e)) / len(on)
        p_off = sum(1 for e in off if hit(e)) / len(off)
        var = p_on * (1 - p_on) / len(on) + p_off * (1 - p_off) / len(off)
        return p_on - p_off, var

    early = lift(group(0, 6))
    late = lift(group(14, None))
    if early is None or late is None:
        return Evaluation("M0.2", "S0", INSUFFICIENT, None, 0.30, [], "proxy:动/不动的注视对比")

    se = (early[1] + late[1]) ** 0.5
    z = 0.0 if se <= 0 else (late[0] - early[0]) / se
    ok = late[0] >= 0.30 and z >= 1.0
    return Evaluation(
        "M0.2",
        "S0",
        PASSED if ok else FAILED,
        late[0],
        0.30,
        [str(e.get("id")) for e in group(14, None) if "mobile" in _onsets(e)][:20],
        f"proxy:动/不动注视差 第1周{early[0]:+.3f} 第3周起{late[0]:+.3f} z={z:.2f}",
    )


def eval_m0_3(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """习惯化：同一个目标第 5 次被看时的新异度 ≤ 第 1 次的 60%。"""
    eps = sorted(_episodes(eps), key=lambda e: str(e.get("ts", "")))
    seq: dict[str, list[float]] = {}
    for e in eps:
        if _prim(e) != "A0.FIXATE":
            continue
        tgt = str(_params(e).get("target", ""))
        nov = _novelty(e).get(tgt)
        if nov is not None:
            seq.setdefault(tgt, []).append(nov)
    best: float | None = None
    for tgt, vals in seq.items():
        if len(vals) < 5:
            continue
        ratio = vals[4] / vals[0] if vals[0] > 0 else 1.0
        if best is None or ratio < best:
            best = ratio
    if best is None:
        return Evaluation("M0.3", "S0", INSUFFICIENT, None, 0.60, [], "proxy:第5次/第1次新异度")
    status = PASSED if best <= 0.60 else FAILED
    return Evaluation("M0.3", "S0", status, best, 0.60, [], "proxy:第5次/第1次新异度")


def eval_m0_4(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """定向反射：有新对象进入视野后，唤醒度随之改变。

    「2 秒内」在 30 分钟一条的粒度上测不出来，这里退一步测**共变**：
    东西刚开始动的那一条，唤醒度相对上一条升了 ≥0.10。
    """
    eps = sorted(_episodes(eps), key=lambda e: str(e.get("ts", "")))
    if len(eps) < 10:
        return Evaluation("M0.4", "S0", INSUFFICIENT, None, 0.60, [], "proxy:运动起始帧的唤醒变化")
    trials = 0
    hits: list[str] = []
    for prev, cur in zip(eps, eps[1:]):
        novel = bool(_onsets(cur))
        if not novel:
            continue
        trials += 1
        # 有符号的变化：定向反射是唤醒**升上去**。取绝对值会把紧跟着的回落帧
        # 也算成「有反应」，而那只是恢复，不是定向。
        da = float((cur.get("appraisal") or {}).get("arousal", 0.0)) - float(
            (prev.get("appraisal") or {}).get("arousal", 0.0)
        )
        if da >= 0.10:
            hits.append(str(cur.get("id")))
    if trials < 5:
        return Evaluation("M0.4", "S0", INSUFFICIENT, None, 0.60, [], "proxy:运动起始帧的唤醒变化")
    value = len(hits) / trials
    status = PASSED if value >= 0.60 else FAILED
    return Evaluation("M0.4", "S0", status, value, 0.60, hits[:20], "proxy:运动起始帧的唤醒变化")


def _hour_of(e: Mapping[str, Any]) -> int:
    ts = str(e.get("ts", ""))
    try:
        return int(ts[11:13])
    except (ValueError, IndexError):
        return -1


def eval_m0_7(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """认照料者：第 3 周起，「她不在」这件事本身开始让她哭（孤独哭出现并增多）。

    原始指标是「对 mother 声音的反应比陌生人强 1σ」，但 S0 的世界里没有陌生人；
    注意力上限只有 2，母亲在不在都占得住注意（实测 97%，天花板效应，量不出生长）。
    所以看哭声的**表达理由**：饥饿/困/不适之外，S0 的哭声里还有一类是 Lonely——
    社交缺口大到盖过别的缺口时才会那么哭（compiler._cry_reason），而社交缺口
    里有一项是「熟人不在」（drives.compute 的 familiarity 项）。

    判据：这类哭在早期（3–7 天）几乎不出现，晚期（≥21 天）在「她不在」的帧里
    占到 ≥12%，两段比例的 z ≥1；且全月不少于 5 次（与 M0.6 同一量级）。
    """
    eps = _episodes(eps)

    def absent(e: Mapping[str, Any]) -> bool:
        others = [str(x) for x in ((e.get("context") or {}).get("others") or [])]
        return "mother" not in others

    def lonely(e: Mapping[str, Any]) -> bool:
        return _prim(e) == "A0.CRY" and str(_params(e).get("reason", "")) == "lonely"

    lonely_all = [e for e in eps if lonely(e)]
    early = [e for e in eps if absent(e) and 3 <= int(e.get("age_days", 0)) <= 7]
    late = [e for e in eps if absent(e) and int(e.get("age_days", 0)) >= 21]
    if len(early) < 12 or len(late) < 12 or len(lonely_all) < 5:
        return Evaluation(
            "M0.7",
            "S0",
            INSUFFICIENT,
            None,
            1.00,
            [str(e.get("id")) for e in lonely_all[:20]],
            f"proxy:孤独哭 {len(lonely_all)} 次",
        )

    pa = sum(1 for e in early if lonely(e)) / len(early)
    pb = sum(1 for e in late if lonely(e)) / len(late)
    p = (pa * len(early) + pb * len(late)) / (len(early) + len(late))
    se = ((p * (1 - p)) * (1 / len(early) + 1 / len(late))) ** 0.5
    z = 0.0 if se <= 0 else (pb - pa) / se
    ok = z >= 1.0 and pb >= 0.12 and len(lonely_all) >= 5
    return Evaluation(
        "M0.7",
        "S0",
        PASSED if ok else FAILED,
        z,
        1.00,
        [str(e.get("id")) for e in lonely_all[:20]],
        f"proxy:孤独哭 早期{pa:.3f}({len(early)}帧) 晚期{pb:.3f}({len(late)}帧) 共{len(lonely_all)}次",
    )


def eval_m0_5(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    eps = _episodes(eps)
    best = 0.0
    evidence: list[str] = []
    for e in eps:
        act = e.get("action") or {}
        if act.get("primitive") == "A0.GRASP":
            hold = float((act.get("params") or {}).get("hold_s", 0.0) or 0.0)
            if hold >= best:
                best = hold
            if hold >= 2.0:
                evidence.append(str(e.get("id")))
    if not eps:
        return Evaluation("M0.5", "S0", INSUFFICIENT, None, 2.0)
    return Evaluation("M0.5", "S0", PASSED if best >= 2.0 else FAILED, best, 2.0, evidence[:20], "direct:握持秒数")


def eval_m0_6(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    eps = _episodes(eps)
    counts: dict[str, int] = {}
    evidence: list[str] = []
    for e in eps:
        act = e.get("action") or {}
        if act.get("primitive") != "A0.CRY":
            continue
        reason = str((act.get("params") or {}).get("reason", ""))
        counts[reason] = counts.get(reason, 0) + 1
        evidence.append(str(e.get("id")))
    need = ("hunger", "discomfort", "sleepy")
    if not eps:
        return Evaluation("M0.6", "S0", INSUFFICIENT, None, 5.0)
    ok = all(counts.get(r, 0) >= 5 for r in need)
    value = float(min(counts.get(r, 0) for r in need)) if counts else 0.0
    return Evaluation(
        "M0.6",
        "S0",
        PASSED if ok else FAILED,
        value,
        5.0,
        evidence[:20],
        "proxy:按原因计次（尚未做强度效应量 d）",
    )


def _unsupported(m: Milestone) -> Evaluation:
    return Evaluation(m.id, m.stage, INSUFFICIENT, None, m.threshold, [], "not_implemented")


# -- S1：语言与共同注意 ------------------------------------------------------
def _monday(ts: str) -> date | None:
    """这条 episode 所属**周一**（ISO 周）。

    注意 `date.fromisocalendar(y, w, weekday)` 的第三个参数是「星期几」，传当天的
    weekday 会原样返回当天；要拿周一必须传 1。
    """
    try:
        iso = _iso_date(ts).isocalendar()
        return date.fromisocalendar(iso[0], iso[1], 1)
    except (ValueError, TypeError):
        return None


def _iso_date(ts: str) -> date:
    return date.fromisoformat(str(ts)[:10])


def eval_m1_6(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """咿呀：带重复的音节串，一天 ≥10 次，连续 5 天。

    只看 `A1.BABBLE` 且 `reps ≥ 2`（单次发声不算「重复音节串」）。
    """
    per_day: dict[str, int] = {}
    for e in _episodes(eps):
        if _prim(e) != "A1.BABBLE":
            continue
        try:
            reps = int(_params(e).get("reps", 0) or 0)
        except (TypeError, ValueError):
            reps = 0
        if reps < 2:
            continue
        day = str(e.get("ts", ""))[:10]
        per_day[day] = per_day.get(day, 0) + 1

    if not per_day:
        return Evaluation("M1.6", "S1", INSUFFICIENT, None, 5.0, [], "proxy:重复音节串天数")

    best = run = 0
    prev: date | None = None
    for key in sorted(per_day):
        try:
            cur = _iso_date(key)
        except ValueError:
            continue
        ok = per_day[key] >= 10
        if ok and prev is not None and (cur - prev).days == 1:
            run += 1
        elif ok:
            run = 1
        else:
            run = 0
        best = max(best, run)
        prev = cur
    status = PASSED if best >= 5 else FAILED
    return Evaluation("M1.6", "S1", status, float(best), 5.0, sorted(per_day)[:20], "proxy:重复音节串天数")


def eval_m1_8(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """第一个词：同一个词形，一周里 ≥3 次，并且连续两周。

    只认 `A1.WORD` 真说出来的那个词形——哭声、咿呀都不算词。
    """
    per_week: dict[tuple[str, date], list[str]] = {}
    for e in _episodes(eps):
        if _prim(e) != "A1.WORD":
            continue
        utt = str((e.get("language") or {}).get("utterance") or "").strip()
        if not utt:
            continue
        monday = _monday(str(e.get("ts", "")))
        if monday is None:
            continue
        per_week.setdefault((utt, monday), []).append(str(e.get("id")))

    if not per_week:
        return Evaluation("M1.8", "S1", INSUFFICIENT, None, 2.0, [], "proxy:同一词形连续两周")

    # 每个词形：哪些周达到了 ≥3 次
    forms: dict[str, list[date]] = {}
    for (form, monday), ids in per_week.items():
        if len(ids) >= 3:
            forms.setdefault(form, []).append(monday)

    best = 0
    best_form = ""
    for form, weeks in forms.items():
        weeks.sort()
        run = 0
        prev: date | None = None
        for monday in weeks:
            run = run + 1 if prev is not None and (monday - prev).days == 7 else 1
            if run > best:
                best, best_form = run, form
            prev = monday
    status = PASSED if best >= 2 else FAILED
    evidence = [i for (form, _), ids in sorted(per_week.items()) if form == best_form for i in ids][:20]
    return Evaluation(
        "M1.8", "S1", status, float(best), 2.0, evidence, f"proxy:同一词形连续两周（{best_form}）"
    )


def _events(e: Mapping[str, Any]) -> list[str]:
    """这一格真的发生了什么。里程碑只认事件，不认自述（docs/10）。"""
    raw = (e.get("outcome") or {}).get("events") or []
    return [str(x) for x in raw]


def _age(e: Mapping[str, Any]) -> int:
    try:
        return int(e.get("age_days", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _others(e: Mapping[str, Any]) -> list[str]:
    raw = (e.get("context") or {}).get("others") or []
    return [str(x) for x in raw]


def _toward_caregiver(e: Mapping[str, Any]) -> bool:
    """这一格的行为是不是朝着照料者去的（看她、爬向她、指给她）。"""
    if _prim(e) == "A0.FIXATE":
        return str(_params(e).get("target") or "") == "mother"
    if _prim(e) in ("A1.CRAWL", "A1.REACH", "A1.POINT", "A1.GIVE"):
        return str(_params(e).get("target") or "") == "mother"
    return False


def eval_m1_1(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """社会性微笑：第 6 周（42 天）起，对熟悉的面孔微笑，一天 ≥5 次。

    笑是**自动表情**（身体侧的输出），不是模型决定要笑；这里数的是事件。
    """
    per_day: dict[str, int] = {}
    for e in _episodes(eps):
        if _age(e) < SMILE_FIRST_DAY:
            continue
        n = sum(1 for x in _events(e) if x == "smiled")
        if n:
            key = str(e.get("ts", ""))[:10]
            per_day[key] = per_day.get(key, 0) + n
    if not per_day:
        return Evaluation(
            "M1.1", "S1", INSUFFICIENT, None, 5.0, [], "proxy:对熟悉面孔的笑（第 6 周起）"
        )
    days_ok = sorted(d for d, v in per_day.items() if v >= 5)
    best = max(per_day.values())
    status = PASSED if best >= 5 and len(days_ok) >= 3 else FAILED
    return Evaluation(
        "M1.1", "S1", status, float(best), 5.0, days_ok[:20], "proxy:对熟悉面孔的笑（第 6 周起）"
    )


def eval_m1_2(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """客体永久性：东西被盖住之后，他还会去找。

    机会 = 摇铃被盖住的那一格；成功 = 接下来两格里他去找了它（掀开/伸手/翻找）。
    看不见就不存在的话，他一次都不会去找。
    """
    rows = _episodes(eps)
    trials = 0
    hits = 0
    evidence: list[str] = []
    covered_before = False
    for i, e in enumerate(rows):
        covered_now = "covered:rattle" in _events(e)
        if not covered_now:
            covered_before = False
            continue
        # 一次「藏起来」算一次机会：藏着的每一格不再重复计机会，
        # 否则一次六格的遮挡会被数成六次尝试，找过一次就把比例抬到 1。
        if covered_before:
            continue
        covered_before = True
        trials += 1
        # 一格三十分钟：给他两个小时去找，仍然算「记得它还在」
        for nxt in rows[i + 1 : i + 5]:
            ev2 = _events(nxt)
            searched = "revealed:rattle" in ev2 or (
                _prim(nxt) == "A1.PEEK" and str(_params(nxt).get("target") or "") == "rattle"
            )
            if searched:
                hits += 1
                evidence.append(str(nxt.get("id")))
                break
    if trials < 10:
        return Evaluation(
            "M1.2", "S1", INSUFFICIENT, None, 0.60, [], f"proxy:被盖住后去找（只有 {trials} 次机会）"
        )
    value = hits / trials
    status = PASSED if value >= 0.60 else FAILED
    return Evaluation("M1.2", "S1", status, round(value, 6), 0.60, evidence[:20], "proxy:被盖住后去找")


def eval_m1_3(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """深度知觉：爬到床沿会停住。

    不让婴儿真的掉下去（那会毁掉后面的所有数据），所以判据是「在边缘停住的那一天」。
    """
    days: list[str] = []
    evidence: list[str] = []
    for e in _episodes(eps):
        if "edge_stopped" not in _events(e):
            continue
        key = str(e.get("ts", ""))[:10]
        if key not in days:
            days.append(key)
        evidence.append(str(e.get("id")))
    if not evidence:
        return Evaluation("M1.3", "S1", INSUFFICIENT, None, 2.0, [], "proxy:床沿停住的天数")
    value = float(len(days))
    status = PASSED if value >= 2.0 else FAILED
    return Evaluation("M1.3", "S1", status, value, 2.0, sorted(days)[:20], "proxy:床沿停住的天数")


def eval_m1_4(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """陌生人焦虑：陌生人在的时候，他更靠近照料者（看她/爬向她）。

    用两比例 z 检验。空口说「他怕生」不算，得看行为比例真的抬起来。
    """
    rows = _episodes(eps)
    with_s: list[float] = []
    without: list[float] = []
    for e in rows:
        hit = 1.0 if _toward_caregiver(e) else 0.0
        if "stranger" in _others(e):
            with_s.append(hit)
        else:
            without.append(hit)
    if len(with_s) < 10 or len(without) < 20:
        return Evaluation(
            "M1.4",
            "S1",
            INSUFFICIENT,
            None,
            2.0,
            [],
            f"proxy:陌生人时向照料者（{len(with_s)} 帧陌生人 / {len(without)} 帧对照）",
        )
    p1 = sum(with_s) / len(with_s)
    p0 = sum(without) / len(without)
    pooled = (sum(with_s) + sum(without)) / (len(with_s) + len(without))
    se = math.sqrt(max(1e-9, pooled * (1.0 - pooled) * (1.0 / len(with_s) + 1.0 / len(without))))
    z = (p1 - p0) / se
    status = PASSED if z >= 2.0 else FAILED
    return Evaluation(
        "M1.4",
        "S1",
        status,
        round(z, 6),
        2.0,
        [f"陌生人 {p1:.2f}", f"平常 {p0:.2f}"],
        "proxy:陌生人时向照料者的比例 z 检验",
    )


def eval_m1_5(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """分离焦虑：照料者不在的时候，负性行为能连着烧几格。"""
    rows = _episodes(eps)
    best = 0
    run = 0
    evidence: list[str] = []
    for e in rows:
        absent = "mother" not in _others(e)
        valence = float((e.get("appraisal") or {}).get("valence", 0.0) or 0.0)
        negative = valence < -0.05 or any(x in ("no_response", "lonely") for x in _events(e)) or _prim(e) == "A0.CRY"
        if absent and negative:
            run += 1
            if run > best:
                best = run
                evidence = [str(e.get("id"))]
            elif run == best and evidence:
                evidence.append(str(e.get("id")))
        else:
            run = 0
    if not rows:
        return Evaluation("M1.5", "S1", INSUFFICIENT, None, 2.0, [], "proxy:分离时连续负性格数")
    status = PASSED if best >= 2 else FAILED
    return Evaluation(
        "M1.5", "S1", status, float(best), 2.0, evidence[:20], "proxy:分离时连续负性格数"
    )


def eval_m1_7(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """共同注意：大人指了某样东西，他的目光会不会跟过去。

    机会 = 一次大人的指向（`caregiver_pointed:<id>`）；成功 = 接下来两格里
    他注视的正是那个被指的东西。他自己指不算——指是「给你看」，跟才是「一起看」。
    """
    rows = _episodes(eps)
    trials = 0
    hits = 0
    evidence: list[str] = []
    for i, e in enumerate(rows):
        # 9 个月以后的跟随才算「稳定跟随」：更早的偶尔跟随是运气，不是能力。
        if _age(e) < FOLLOW_MIN_AGE:
            continue
        target = ""
        for x in _events(e):
            if str(x).startswith("caregiver_pointed:"):
                target = str(x).split(":", 1)[1]
        if not target or i + 1 >= len(rows):
            continue
        trials += 1
        for nxt in rows[i + 1 : i + 3]:
            if _prim(nxt) == "A0.FIXATE" and str(_params(nxt).get("target") or "") == target:
                hits += 1
                evidence.append(str(nxt.get("id")))
                break
    if trials < 10:
        return Evaluation(
            "M1.7", "S1", INSUFFICIENT, None, 0.70, [], f"proxy:跟随指向（只有 {trials} 次指向）"
        )
    value = hits / trials
    status = PASSED if value >= 0.70 else FAILED
    return Evaluation("M1.7", "S1", status, round(value, 6), 0.70, evidence[:20], "proxy:跟随指向")


def _walk_days(eps: Sequence[Mapping[str, Any]]) -> dict[date, int]:
    """每天走了多少步。一步 = 一个 A2.WALK（停在原地的那些不算）。"""
    per_day: dict[date, int] = {}
    for e in _episodes(eps):
        if _prim(e) != "A2.WALK":
            continue
        events = [str(x) for x in (e.get("outcome") or {}).get("events") or []]
        if any(x.startswith("already_there") for x in events):
            continue
        try:
            d = _iso_date(str(e.get("ts", "")))
        except ValueError:
            continue
        per_day[d] = per_day.get(d, 0) + 1
    return per_day


def _longest_consecutive(days: Sequence[date]) -> int:
    best = run = 0
    prev: date | None = None
    for d in sorted(days):
        run = run + 1 if prev is not None and (d - prev).days == 1 else 1
        best = max(best, run)
        prev = d
    return best


def eval_m2_1(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """独立行走：一天里自己走了 ≥10 步，而且连着三天。

    走是位移，不是被抱着。所以只数 A2.WALK——S2 之前这个原语根本不存在。
    """
    per_day = _walk_days(eps)
    walks = sum(1 for e in _episodes(eps) if _prim(e) == "A2.WALK")
    if not per_day:
        # 走了，但一步没挪（原地踏步）——这是失败，不是没数据
        if walks:
            return Evaluation("M2.1", "S2", FAILED, 0.0, 3.0, [], "proxy:会走的天数")
        return Evaluation("M2.1", "S2", INSUFFICIENT, None, 3.0, [], "proxy:会走的天数")
    good = [d for d, n in per_day.items() if n >= 10]
    best = _longest_consecutive(good)
    status = PASSED if best >= 3 else FAILED
    evidence = [d.isoformat() for d in sorted(good)][:20]
    return Evaluation("M2.1", "S2", status, float(best), 3.0, evidence, "proxy:会走的天数")


def eval_m2_2(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """词汇爆发：任意连续 14 天里，他说出口的不同词形 ≥50 个。

    数的是**说出来的**词，不是听过的词——词汇爆发是产出侧的事。
    """
    per_day: dict[date, set[str]] = {}
    for e in _episodes(eps):
        prim = _prim(e)
        forms: list[str] = []
        if prim == "A1.WORD":
            utt = str((e.get("language") or {}).get("utterance") or "").strip()
            if utt:
                forms.append(utt)
        elif prim == "A2.TWO_WORD":
            params = _params(e)
            for key in ("a", "b"):
                form = str(params.get(key) or "").strip()
                if form:
                    forms.append(form)
        if not forms:
            continue
        try:
            d = _iso_date(str(e.get("ts", "")))
        except ValueError:
            continue
        per_day.setdefault(d, set()).update(forms)

    if not per_day:
        return Evaluation("M2.2", "S2", INSUFFICIENT, None, 50.0, [], "proxy:两周词汇量")

    days = sorted(per_day)
    best = 0
    for start in days:
        window: set[str] = set()
        for d in days:
            delta = (d - start).days
            if delta < 0:
                continue
            if delta >= 14:
                break
            window |= per_day[d]
        best = max(best, len(window))
    status = PASSED if best >= 50 else FAILED
    return Evaluation("M2.2", "S2", status, float(best), 50.0, [], f"proxy:两周词汇量（{best} 个）")


def eval_m2_3(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """双词组合：他说出来过的双词，有 ≥20 种不同的搭配。"""
    kinds: set[str] = set()
    evidence: list[str] = []
    for e in _episodes(eps):
        if _prim(e) != "A2.TWO_WORD":
            continue
        utt = str((e.get("language") or {}).get("utterance") or "").strip()
        if utt:
            kinds.add(utt)
            if len(evidence) < 20:
                evidence.append(str(e.get("id")))
    value = float(len(kinds))
    status = PASSED if value >= 20 else (INSUFFICIENT if not kinds else FAILED)
    return Evaluation("M2.3", "S2", status, value, 20.0, evidence, "proxy:双词种类")


def eval_m2_4(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """自我识别：在镜子前反复伸手——镜子里那张脸，他认出来是自己。"""
    hits = 0
    evidence: list[str] = []
    for e in _episodes(eps):
        events = [str(x) for x in (e.get("outcome") or {}).get("events") or []]
        if any(x == "manipulated:turn:mirror" or x.startswith("looked_at_mirror") for x in events):
            hits += 1
            if len(evidence) < 20:
                evidence.append(str(e.get("id")))
    status = PASSED if hits >= 5 else (INSUFFICIENT if hits == 0 else FAILED)
    return Evaluation("M2.4", "S2", status, float(hits), 5.0, evidence, "proxy:镜子前的触碰")


def eval_m2_8(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """代词「我」：说出 ≥10 次。

    这是一个词，不是一种能力——但一个会说「我」的孩子，已经在用语言指自己了。
    """
    hits = 0
    evidence: list[str] = []
    for e in _episodes(eps):
        utt = str((e.get("language") or {}).get("utterance") or "").strip()
        if "我" in utt:
            hits += 1
            if len(evidence) < 20:
                evidence.append(str(e.get("id")))
    if hits == 0:
        return Evaluation("M2.8", "S2", INSUFFICIENT, 0.0, 10.0, [], "proxy:说出「我」的次数")
    status = PASSED if hits >= 10 else FAILED
    return Evaluation("M2.8", "S2", status, float(hits), 10.0, evidence, "proxy:说出「我」的次数")


def eval_m2_7(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """自平复：难受之后，不用人抱也能自己缓过来。

    真月龄的判据是「2 分钟内平复」，30 分钟一格量不到；这里量的是**下一小时**：
    一次难过（效价 ≤ -0.10 或哭了）之后，四格之内效价回到 -0.05 以上，
    且中间没有人抱他、没有人哄他——被哄好的不算自己好的。
    """
    items = _episodes(eps)
    total = 0
    recovered = 0
    evidence: list[str] = []
    for i, e in enumerate(items):
        app = e.get("appraisal") or {}
        valence = float(app.get("valence") or 0.0)
        prim = str((e.get("action") or {}).get("primitive") or "")
        if valence > -0.10 and prim != "A0.CRY":
            continue
        total += 1
        for nxt in items[i + 1 : i + 5]:
            events = [str(x) for x in (nxt.get("outcome") or {}).get("events") or ()]
            if any(x.startswith("held") or x.startswith("caregiver_hold") for x in events):
                break
            if float((nxt.get("appraisal") or {}).get("valence") or 0.0) >= -0.05:
                recovered += 1
                if len(evidence) < 20:
                    evidence.append(str(e.get("id")))
                break
    if total < 6:
        return Evaluation(
            "M2.7", "S2", INSUFFICIENT, None, 0.40, evidence, "proxy:没人哄也自己缓过来"
        )
    rate = recovered / total
    status = PASSED if rate >= 0.40 else FAILED
    return Evaluation(
        "M2.7", "S2", status, r6(rate), 0.40, evidence, f"proxy:没人哄也自己缓过来 {recovered}/{total}"
    )


def eval_m2_5(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """假装游戏：拿一个东西当别的用，至少三次，而且不是同一出。

    判据只看行为事件：`pretended:<剧本>`。剧本种类 ≥2 才算「假装」而不是
    「喜欢那个动作」——同一出戏重复三遍，可能只是在玩那只杯子。
    """
    items = _episodes(eps)
    total = 0
    scripts: set[str] = set()
    evidence: list[str] = []
    for e in items:
        for ev in (e.get("outcome") or {}).get("events") or ():
            text = str(ev)
            if not text.startswith("pretended:"):
                continue
            total += 1
            scripts.add(text.split(":", 1)[1])
            if len(evidence) < 20:
                evidence.append(str(e.get("id")))
    if total == 0:
        return Evaluation("M2.5", "S2", INSUFFICIENT, 0.0, 3.0, [], "proxy:假装动作的次数")
    status = PASSED if total >= 3 and len(scripts) >= 2 else FAILED
    return Evaluation(
        "M2.5", "S2", status, float(total), 3.0, evidence, f"proxy:假装 {total} 次，{len(scripts)} 种剧本"
    )


def eval_m2_6(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """延迟满足：东西被收着的时候，等得住。

    真实验里是 30 秒；30 分钟一格量不到，这里把「这一格没有去抢」当成等住了
    （比 30 秒严格得多，宁可判严）。
    """
    items = _episodes(eps)
    chances = 0
    waits = 0
    evidence: list[str] = []
    for e in items:
        events = [str(x) for x in (e.get("outcome") or {}).get("events") or ()]
        prim = str((e.get("action") or {}).get("primitive") or "")
        if not (any(x.startswith("withheld:") for x in events) or prim == "A2.WAIT"):
            continue
        chances += 1
        if prim == "A2.WAIT":
            waits += 1
            if len(evidence) < 20:
                evidence.append(str(e.get("id")))
    if chances == 0:
        return Evaluation(
            "M2.6", "S2", INSUFFICIENT, None, 0.60, evidence, "proxy:收着东西时没去抢"
        )
    rate = min(1.0, waits / chances)
    status = PASSED if rate >= 0.60 else FAILED
    return Evaluation(
        "M2.6", "S2", status, r6(rate), 0.60, evidence, f"proxy:等了 {waits} 次 / 收了 {chances} 次"
    )


def eval_m2_9(eps: Sequence[Mapping[str, Any]]) -> Evaluation:
    """形象选择：几张卡片摆在眼前，他挑的那张是不是稳定的。

    测的是**偏好的一致性**（top1 占比），不是「挑对了」。他挑了哪张不重要，
    每次都挑同一张才说明他有偏好；top1 = 60% 就是明显高于随便挑。
    """
    items = _episodes(eps)
    picks: dict[str, int] = {}
    evidence: list[str] = []
    for e in items:
        prim = str((e.get("action") or {}).get("primitive") or "")
        if prim != "A2.CHOOSE":
            continue
        target = str((e.get("action") or {}).get("target") or "")
        if not target:
            continue
        picks[target] = picks.get(target, 0) + 1
        if len(evidence) < 20:
            evidence.append(str(e.get("id")))
    total = sum(picks.values())
    if total < 10:
        return Evaluation("M2.9", "S2", INSUFFICIENT, None, 0.60, evidence, "proxy:形象选择 top1")
    top = max(picks.values())
    share = top / total
    status = PASSED if share >= 0.60 else FAILED
    return Evaluation(
        "M2.9", "S2", status, r6(share), 0.60, evidence, f"proxy:形象选择 top1 {share:.2f}（{total} 次）"
    )


EVALUATORS: dict[str, Callable[[Sequence[Mapping[str, Any]]], Evaluation]] = {
    "M0.1": eval_m0_1,
    "M0.2": eval_m0_2,
    "M0.3": eval_m0_3,
    "M0.4": eval_m0_4,
    "M0.5": eval_m0_5,
    "M0.6": eval_m0_6,
    "M0.7": eval_m0_7,
    "M1.1": eval_m1_1,
    "M1.2": eval_m1_2,
    "M1.3": eval_m1_3,
    "M1.4": eval_m1_4,
    "M1.5": eval_m1_5,
    "M1.6": eval_m1_6,
    "M1.7": eval_m1_7,
    "M1.8": eval_m1_8,
    "M2.1": eval_m2_1,
    "M2.2": eval_m2_2,
    "M2.3": eval_m2_3,
    "M2.4": eval_m2_4,
    "M2.5": eval_m2_5,
    "M2.6": eval_m2_6,
    "M2.7": eval_m2_7,
    "M2.8": eval_m2_8,
    "M2.9": eval_m2_9,
}


def evaluate(stage: str, episodes: Sequence[Mapping[str, Any]]) -> list[Evaluation]:
    """评估该阶段全部里程碑。未实现的给 `insufficient`，不伪造通过。"""
    out: list[Evaluation] = []
    for m in milestones_of(stage):
        fn = EVALUATORS.get(m.id)
        out.append(fn(episodes) if fn else _unsupported(m))
    return out


def all_passed(evals: Sequence[Evaluation]) -> bool:
    return bool(evals) and all(e.status == PASSED for e in evals)
