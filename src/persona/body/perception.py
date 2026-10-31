"""感知：把世界投影成一份「婴儿此刻能感到的东西」。

三条硬约束（docs/08）：
* **未解锁的通道根本不存在**——S0 的 bundle 里没有 face 字段，不是填 null；
* 注意力有上限（S0 只装得下 2 个对象），**没被注意到的对象不会被写进 bundle**；
* 同输入 + 同 seed → 同一份 bundle，噪声来自 `derive(day, episode_id, channel)`。

注意 S0 的一个真实细节：婴儿能听见、看见、被摸到，但**听不懂**。语言通道在 S1 才出现。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from persona.types import STAGES

STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}
from persona.body.state import BodyState
from persona.rng import derive
from persona.world.objects import CRIB_HALF_X, CRIB_HALF_Y, WorldObject, WorldState, edge_distance

#: 通道表：id / 单位成本 / 解锁阶段（docs/08 感知通道）
CHANNELS: dict[str, tuple[str, int, str]] = {
    "V0": ("V0.luminance", 12, "S0"),
    "V1": ("V1.vision", 28, "S0"),
    "AUD": ("AUD.audition", 24, "S0"),
    "TAC": ("TAC.touch", 16, "S0"),
    "INT": ("INT.interoception", 18, "S0"),
    "FAC": ("FAC.face", 30, "S1"),
    "TRK": ("TRK.motion", 20, "S1"),
    "DEP": ("DEP.depth", 18, "S2"),
    "ATT": ("ATT.joint_attention", 14, "S2"),
    "MND": ("MND.mental_state", 26, "S3"),
    "RUL": ("RUL.symbolic", 40, "S4"),
    "SOC": ("SOC.social_eval", 30, "S5"),
}

#: 每阶段的注意力带宽（docs/08）
ATTENTION_MAX: dict[str, int] = {"S0": 2, "S1": 3, "S2": 4, "S3": 5, "S4": 6, "S5": 8, "S6": 8}
#: 每阶段 bundle 的字符预算（约 token ×2.2，中文按 token 计的保守估计）
BUNDLE_BUDGET: dict[str, int] = {
    "S0": 840,
    "S1": 1020,
    "S2": 1240,
    "S3": 1500,
    "S4": 1760,
    "S5": 1980,
    "S6": 1980,
}

CHANNEL_UNLOCK = {k: v[2] for k, v in CHANNELS.items()}


def unlocked_channels(stage: str) -> list[str]:
    idx = STAGES.index(stage)
    return [k for k, (_, _, min_stage) in CHANNELS.items() if STAGES.index(min_stage) <= idx]


@dataclass
class Stimulus:
    """一个被注意到的对象，以及它在各通道上的读数。"""

    id: str
    label: str
    kind: str
    salience: float
    novelty: float = 0.0
    channels: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "salience": round(self.salience, 6),
            "novelty": round(self.novelty, 6),
            "channels": self.channels,
        }


@dataclass
class Bundle:
    """感知结果。它同时是「给内核看的输入」和「给日志留的证据」。"""

    stage: str
    modality: list[str]
    stimuli: list[Stimulus]
    luminance: float
    ambient_db: float
    interoception: dict[str, float]
    attention_max: int
    dropped: list[str] = field(default_factory=list)
    truncated: bool = False
    budget_chars: int = 0
    # 这一帧里「刚刚开始动」的对象：定向反射与视觉追踪的证据来源
    onsets: list[str] = field(default_factory=list)
    # 这一帧里在动的对象（发条玩具的「转动时段」）
    moving: list[str] = field(default_factory=list)

    def salient_ids(self) -> list[str]:
        return [s.id for s in self.stimuli]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "modality": list(self.modality),
            "luminance": round(self.luminance, 6),
            "ambient_db": round(self.ambient_db, 6),
            "interoception": {k: round(v, 6) for k, v in self.interoception.items()},
            "stimuli": [s.to_dict() for s in self.stimuli],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def chars(self) -> int:
        return len(self.to_json())

    # -- 给 episode 的观察摘要（人读，非 LLM 生成） -------------------------
    def summary(self) -> str:
        if not self.stimuli:
            return "没有看到任何东西"
        parts = []
        for s in self.stimuli:
            bits = []
            if "V1" in s.channels:
                bits.append(f"亮度{s.channels['V1']['luminance']:.1f}")
                if s.channels["V1"].get("moving"):
                    bits.append("在动")
            if "AUD" in s.channels:
                bits.append(f"声音{s.channels['AUD']['loudness']:.2f}")
            if "TAC" in s.channels:
                bits.append(s.channels["TAC"]["texture"])
            parts.append(f"{s.label}（{'/'.join(bits)}）" if bits else s.label)
        return "感觉到：" + "、".join(parts)

    def observation(self) -> dict[str, Any]:
        """`episode.observation` 三件套：modality / summary / salient。"""
        return {
            "modality": list(self.modality),
            "summary": self.summary(),
            "salient": self.salient_ids(),
            "interoception": {k: round(v, 6) for k, v in self.interoception.items()},
            # 每个显著对象的新异度：习惯化里程碑要靠这条时间序列才判得动
            "novelty": {s.id: round(s.novelty, 6) for s in self.stimuli},
            # 这一帧里刚动起来的对象（定向反射/追踪的判定依据）
            "motion_onsets": list(self.onsets),
            "moving": list(self.moving),
        }


class Perception:
    """按阶段投影。**所有**随机性都从这里派生，保证同一天重跑一致。"""

    def __init__(self, *, noise: float = 0.05) -> None:
        self.noise = float(noise)

    # -- 主流程 -------------------------------------------------------------
    def bundle(
        self,
        world: WorldState,
        body: BodyState,
        *,
        stage: str,
        day: int,
        episode_id: str,
        clock_hours: float,
        goal_focus: str | None = None,
    ) -> Bundle:
        channels = unlocked_channels(stage)
        max_att = ATTENTION_MAX[stage]
        budget = BUNDLE_BUDGET[stage]
        rng = derive("perception", day, episode_id)

        def noise(scale: float = 1.0) -> float:
            return rng.gauss(0.0, self.noise * scale)

        # 运动起始：上一次看到它时不动、这次在动 → 记一次 onset（去习惯化从这里开始）
        onsets: list[str] = []
        for obj in world.visible_objects():
            on = obj.moving(clock_hours)
            was = world.motion_seen.get(obj.id, False)
            if on and not was:
                world.motion_onset[obj.id] = world.abs_hours
                onsets.append(obj.id)
            world.motion_seen[obj.id] = on

        candidates = self._candidates(
        world, body, goal_focus=goal_focus, stage=stage, t=clock_hours
    )
        ranked = sorted(candidates, key=lambda t: t[1], reverse=True)
        attended = ranked[:max_att]
        dropped = [s.id for s, _ in ranked[max_att:]]

        stimuli: list[Stimulus] = []
        for obj, sal in attended:
            stimuli.append(
                self._project(
                    obj,
                    sal,
                    world=world,
                    body=body,
                    stage=stage,
                    channels=channels,
                    t=clock_hours,
                    noise=noise,
                )
            )

        # 习惯化记在「看进来的东西」上，而不是记在「伸手去抓的东西」上：
        # 目光扫过就算看过一眼。记在 FIXATE 上的话，从来没被抓住的东西
        # 新异度永远停在 1.0（母亲就是这样一直不变的）。
        for obj, _ in attended:
            world.fixation_counts[obj.id] = world.fixation_counts.get(obj.id, 0) + 1

        bundle = Bundle(
            stage=stage,
            modality=[CHANNELS[c][0] for c in channels],
            stimuli=stimuli,
            luminance=max(0.05, min(1.0, world.light + noise(0.6))),
            ambient_db=max(0.0, min(1.0, world.ambient_db + noise(0.4))),
            interoception={
                "energy": body.energy / 100.0,
                "fatigue": body.fatigue,
                "hunger": body.hunger,
                "temperature": body.temperature,
                "integrity": body.integrity,
            },
            attention_max=max_att,
            dropped=dropped,
            budget_chars=budget,
            onsets=[o for o in onsets if o in {s.id for s in stimuli}],
            moving=[o.id for o, _ in attended if o.moving(clock_hours)],
        )

        # 预算超了就砍最不显著的对象（保留 id 与显著度，删细节）
        while bundle.chars() > budget and len(bundle.stimuli) > 1:
            bundle.stimuli.pop()
            bundle.truncated = True
        if bundle.truncated:
            bundle.dropped = bundle.dropped + ["(budget)"]
        return bundle

    # -- 内部 ---------------------------------------------------------------
    def _novelty(self, world: WorldState, obj: WorldObject) -> float:
        """新异度 = 1/(1+看过几次)。刚刚开始动的东西例外：它重新是新的。

        这是「习惯化—去习惯化」的最小实现：看着不动的东西会腻，一动起来又新鲜。
        """
        if world.motion_onset_now(obj.id):
            return 1.0
        return 1.0 / (1.0 + world.fixation_counts.get(obj.id, 0))

    def _candidates(
        self,
        world: WorldState,
        body: BodyState,
        *,
        goal_focus: str | None,
        stage: str = "S0",
        t: float | None = None,
    ) -> list[tuple[WorldObject, float]]:
        out: list[tuple[WorldObject, float]] = []
        # 会走之前，地板上那些东西不在他的世界里：不是「看不见」，是**还不存在**。
        # 这正是 docs/08 的 I1：未解锁能力对应的对象，由世界适配器按阶段实例化。
        s2 = STAGE_INDEX.get(stage, 0) >= STAGE_INDEX.get("S2", 2)
        for obj in world.visible_objects():
            if "appears_at_S2" in obj.tags and not s2:
                continue
            novelty = self._novelty(world, obj)
            # 距离从他**站的地方**算，不是从床中央算——会走之后这两者不是一回事
            dist = _distance(obj.pos, world.baby_pos)
            proximity = max(0.0, 1.0 - dist / 2.0)
            motion = 1.0 if obj.moving(t) else 0.0
            relevance = 0.0
            if goal_focus and (goal_focus in obj.tags or goal_focus == obj.id):
                relevance = 1.0
            elif body.hunger > 0.6 and "social" in obj.tags:
                relevance = 0.6
            elif body.tired and obj.kind == "fabric":
                relevance = 0.4
            sal = 0.4 * novelty + 0.2 * proximity + 0.2 * motion + 0.2 * relevance
            # 熟悉度：被照料得越多，那个人越容易进入注意——这是联结的雏形，
            # 不是「S0 就认脸」（认脸要 S1 的 FAC 通道），只是注意分配的改变。
            if "social" in obj.tags:
                care_n = sum(v for k, v in world.counters.items() if k.startswith("caregiver_"))
                sal += 0.06 * (1.0 - math.exp(-care_n / 200.0))
            out.append((obj, min(1.0, max(0.0, sal))))
        return out

    def _project(
        self,
        obj: WorldObject,
        salience: float,
        *,
        world: WorldState,
        body: BodyState,
        stage: str,
        channels: list[str],
        t: float,
        noise: Any,
    ) -> Stimulus:
        stim = Stimulus(
            id=obj.id,
            label=obj.label,
            kind=obj.kind,
            salience=salience,
            novelty=self._novelty(world, obj),
        )
        pos = obj.offset(t)

        if "V1" in channels:
            x, y = _project_screen(pos)
            stim.channels["V1"] = {
                "x": round(min(1.0, max(0.0, x + noise(0.3))), 4),
                "y": round(min(1.0, max(0.0, y + noise(0.3))), 4),
                "size": round(max(0.02, min(1.0, obj.size[0] + noise(0.2))), 4),
                "luminance": round(
                    max(0.0, min(1.0, obj.luminance * (0.55 + 0.45 * world.light) + noise(0.5))),
                    4,
                ),
            }
            if "TRK" in channels:
                stim.channels["V1"]["moving"] = obj.moving(t)

        if "AUD" in channels and obj.loudness > 0.05:
            stim.channels["AUD"] = {
                "loudness": round(max(0.0, min(1.0, obj.loudness + noise(0.3))), 4),
                "kind": "voice" if obj.kind == "person" else "object",
            }

        if "TAC" in channels and (obj.reachable or _distance(pos, (0.0, 0.0, 0.15)) < 0.5):
            stim.channels["TAC"] = {
                "texture": obj.texture,
                "temperature": round(max(0.0, min(1.0, obj.temperature + noise(0.3))), 4),
            }

        if "FAC" in channels and obj.kind == "person" and stage >= obj.no_face_until:
            stim.channels["FAC"] = {
                "present": True,
                # S1 的面孔通道不只说「有张脸」，还说「这张脸我熟不熟」。
                # 熟悉/陌生是感知层给的读数，不是评价层的猜测（docs/08）。
                "familiar": bool(obj.extra.get("familiar", obj.id == "mother")),
                "expression": round(max(-1.0, min(1.0, world.caregiver_mood * 2 - 1)), 4),
            }

        if "TRK" in channels and obj.moving(t) and "V1" not in stim.channels:
            stim.channels["TRK"] = {"moving": True}

        if "DEP" in channels:
            stim.channels["DEP"] = {"distance_m": round(_distance(pos, (0.0, 0.0, 0.15)), 4)}

        # 深度知觉要一个"边缘"的信号：床沿在视野里是一条陡的亮度断崖。
        # S1 还没有 DEP 通道（那是 S2），但他已经会绕着床沿停住——
        # 所以这里给的是 V1 上的一个几何读数，不是三维距离。
        if "V1" in channels:
            stim.channels["V1"]["edge_m"] = round(edge_distance(pos), 4)

        return stim


def _edge_distance(pos: tuple[float, float, float]) -> float:
    """对象到床沿的最近距离（米）。对象在床外就是 0。"""
    x, y, _ = pos
    dx = CRIB_HALF_X - abs(x)
    dy = CRIB_HALF_Y - abs(y)
    return max(0.0, min(dx, dy))


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _project_screen(pos: tuple[float, float, float]) -> tuple[float, float]:
    """极简透视：把 3D 位置压到 0–1 的视野坐标（婴儿仰卧，正上方是中心）。"""
    x = 0.5 + pos[0] * 0.6
    y = 0.5 - (pos[2] - 0.2) * 0.9 - pos[1] * 0.2
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
