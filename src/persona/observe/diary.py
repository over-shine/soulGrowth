"""把一天写成一段人话。

这份日记是**观察者**写的，不是他自己写的：S0 没有语言，让他写日记就等于
替他说话（docs/06 的四层反作弊要防的第一件事）。第一人称日记要等到 L5，
那时它才有词可用。

所有句子都由 episode 记录拼出来，模板不承担判断，只负责排版；因此同样的库
一定生成同样的日记（确定性），可以进 git、可以 diff。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from persona.types import r6

#: 动作的白话名。没有语言的人也有行为，只是行为没有名字——这些名字是我们给的。
NAMES = {
    "A0.FIXATE": "注视",
    "A0.TURN_HEAD": "转头",
    "A0.GRASP": "抓住",
    "A0.SUCK": "吮吸",
    "A0.CRY": "哭",
    "A0.STARTLE": "惊跳",
    "A0.SLEEP": "睡",
    "A0.NONE": "什么也没做",
    "A1.REACH": "伸手",
    "A1.CRAWL": "爬",
    "A1.SIT": "坐起来",
    "A1.POINT": "指",
    "A1.BABBLE": "咿呀",
    "A1.WORD": "说一个词",
    "A1.GIVE": "递东西",
    "A1.TAKE": "拿东西",
    "A1.PEEK": "找东西",
    "A2.WALK": "走",
    "A2.MANIPULATE": "摆弄",
    "A2.IMITATE": "模仿",
    "A2.TWO_WORD": "说两个词",
    "A2.TELEGRAPH": "预告",
    "A2.SHOW": "展示",
    "A2.REFUSE": "拒绝",
    "A3.PRETEND": "假装",
    "A3.COOPERATE": "一起玩",
    "A3.ASK": "问",
    "A3.SENTENCE": "说一句话",
    "A3.HELP": "帮忙",
    "A3.SNATCH": "抢",
    "A4.PLAN": "计划",
    "A4.TOOL": "用工具",
    "A4.RULE_GAME": "玩规则游戏",
    "A4.READ": "读",
    "A4.WRITE": "写",
    "A4.COMPOUND": "说复合句",
    "A4.MEASURE": "量",
}

CRY_REASONS = {"hunger": "饿", "discomfort": "不舒服", "sleepy": "困", "lonely": "孤单"}

MILESTONE_NAMES = {
    "M0.1": "视觉定向",
    "M0.2": "视觉追踪",
    "M0.3": "习惯化",
    "M0.4": "定向反射",
    "M0.5": "抓握反射",
    "M0.6": "哭声分化",
    "M0.7": "认照料者",
    "M1.1": "社会性微笑",
    "M1.2": "客体永久性",
    "M1.3": "深度知觉",
    "M1.4": "陌生人焦虑",
    "M1.5": "分离焦虑",
    "M1.6": "咿呀",
    "M1.7": "共同注意",
    "M1.8": "第一个词",
}

#: 一天的三个时段。没有更细的划分，是因为 S0 的节律粗到只有「醒着/睡着」。
BANDS = (("上午", 6, 12), ("下午", 12, 18), ("傍晚", 18, 23))


def _prim(ep: Mapping[str, Any]) -> str:
    if ep.get("kind") == "sleep":
        return "A0.SLEEP"
    return str(ep.get("action", {}).get("primitive") or "A0.NONE")


def _cn(prim: str) -> str:
    return NAMES.get(prim, prim)


def _clock(ts: str) -> str:
    return ts[11:16] if len(ts) >= 16 else ts


def _hour(ts: str) -> int:
    try:
        return int(str(ts)[11:13])
    except (ValueError, IndexError):
        return 0


def _targets(eps: Iterable[Mapping[str, Any]]) -> Counter:
    c: Counter = Counter()
    for ep in eps:
        act = ep.get("action", {})
        if act.get("primitive") in ("A0.FIXATE", "A0.GRASP", "A0.TURN_HEAD"):
            t = act.get("params", {}).get("target") or act.get("target")
            if t:
                c[str(t)] += 1
    return c


def _band_line(name: str, eps: Sequence[Mapping[str, Any]]) -> str | None:
    if not eps:
        return None
    counts = Counter(_cn(_prim(e)) for e in eps)
    parts = "、".join(f"{k} {v} 次" for k, v in counts.most_common(3))
    return f"- {name}：{parts}"


def _highlights(eps: Sequence[Mapping[str, Any]]) -> list[str]:
    """值得单独写一行的事：某动作这一天第一次出现、连着哭。"""
    out: list[str] = []
    seen: set[str] = set()
    for ep in eps:
        prim = _prim(ep)
        if prim in seen:
            continue
        seen.add(prim)
        out.append(f"- {_clock(str(ep.get('ts', '')))} 今天第一次{_cn(prim)}")
    # 连续两次以上的哭
    run = 0
    start_ts = ""
    for ep in eps:
        if _prim(ep) == "A0.CRY":
            if run == 0:
                start_ts = _clock(str(ep.get("ts", "")))
            run += 1
        else:
            if run >= 2:
                out.append(f"- {start_ts} 连着哭了 {run} 次")
            run = 0
    if run >= 2:
        out.append(f"- {start_ts} 连着哭了 {run} 次")
    return out


def render_day(
    *,
    day: int,
    date: str,
    stage: str,
    episodes: Sequence[Mapping[str, Any]],
    milestones: Sequence[Mapping[str, Any]] = (),
    traits: Mapping[str, Any] | None = None,
    memories: Sequence[Mapping[str, Any]] = (),
    name: str = "小满",
) -> str:
    awake = [e for e in episodes if e.get("kind") != "sleep"]
    sleep = [e for e in episodes if e.get("kind") == "sleep"]
    lines: list[str] = []
    lines.append(f"# 第 {day} 天 · {date}")
    lines.append("")
    lines.append(f"{name}，{stage}。这一天有 {len(awake)} 段清醒、{len(sleep)} 段睡眠。")
    lines.append("")

    lines.append("## 一天的样子")
    lines.append("")
    for band, lo, hi in BANDS:
        chunk = [e for e in episodes if lo <= _hour(str(e.get("ts", ""))) < hi]
        line = _band_line(band, chunk)
        if line:
            lines.append(line)
    lines.append("")

    marks = _highlights(episodes)
    if marks:
        lines.append("## 值得记一笔")
        lines.append("")
        lines.extend(marks)
        lines.append("")

    cries = [e for e in awake if _prim(e) == "A0.CRY"]
    if cries:
        reasons = Counter(CRY_REASONS.get(str(e["action"].get("params", {}).get("reason")), "说不清") for e in cries)
        detail = "、".join(f"{k} {v} 次" for k, v in reasons.most_common())
        lines.append("## 哭")
        lines.append("")
        lines.append(f"哭了 {len(cries)} 次：{detail}。")
        lines.append("")

    said = [
        str((e.get("language") or {}).get("utterance") or "")
        for e in awake
        if _prim(e) == "A1.WORD" and (e.get("language") or {}).get("utterance")
    ]
    if said:
        words = Counter(said)
        first = _clock(str(next(e["ts"] for e in awake if _prim(e) == "A1.WORD")))
        lines.append("## 说话")
        lines.append("")
        detail = "、".join(f"「{w}」{n} 次" for w, n in words.most_common())
        lines.append(f"说了 {len(said)} 次：{detail}。第一次是 {first} 的「{said[0]}」。")
        lines.append("")

    tg = _targets(awake)
    if tg:
        lines.append("## 看得最多")
        lines.append("")
        lines.append("、".join(f"{k} {v} 次" for k, v in tg.most_common(4)) + "。")
        lines.append("")

    if milestones:
        lines.append("## 里程碑")
        lines.append("")
        for m in milestones:
            status = {"passed": "通过", "failed": "未通过", "insufficient": "样本不足"}.get(
                str(m.get("status")), str(m.get("status"))
            )
            value = m.get("value")
            vtxt = "" if value is None else f"，值 {r6(float(value)):.2f}"
            lines.append(f"- {m.get('id')} {MILESTONE_NAMES.get(str(m.get('id')), '')}：{status}{vtxt}")
        lines.append("")

    if episodes:
        body = (awake or episodes)[-1].get("drives") or {}
        if body:
            lines.append("## 驱力")
            lines.append("")
            lines.append(
                f"能量 {float(body.get('energy', 0)):.2f}、"
                f"刺激 {float(body.get('stimulation', 0)):.2f}、"
                f"社交 {float(body.get('social', 0)):.2f}、"
                f"胜任 {float(body.get('competence', 0)):.2f}、"
                f"安全 {float(body.get('safety', 0)):.2f}。"
            )
            lines.append("")

    if traits:
        t = traits.get("T") or {}
        if t:
            lines.append("## 此刻的样子")
            lines.append("")
            lines.append(
                "开放 {O:+.2f}、尽责 {C:+.2f}、外向 {E:+.2f}、宜人 {A:+.2f}、情绪稳定 {N:+.2f}".format(
                    **{k: float(t.get(k, 0.0)) for k in ("O", "C", "E", "A", "N")}
                )
            )
            lines.append("")

    if memories:
        lines.append("## 他记住的")
        lines.append("")
        for m in memories:
            stamp = _clock(str(m.get("created", ""))) or "——"
            content = str(m.get("content", "")).split("；随后")[0]
            lines.append(f"- {stamp} {content}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_day(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out
