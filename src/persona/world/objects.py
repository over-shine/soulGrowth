"""世界对象模型。

世界是**唯一有记忆的物理层**：它保存物体位置、照料者状态、以及一个用于「习惯化」的注视计数。
婴儿不知道这些；婴儿只拿到 projections（见 body/perception.py）。

一条纪律：物体身上**没有**任何「给 LLM 看的描述字段**。所有可感知量都是数值的，
描述由身体层按阶段投影出来——否则 D4（适配器按阶段实例化）形同虚设。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persona.types import canonical_json

Vec3 = tuple[float, float, float]

#: 婴儿床的半边长（米）。越过这条线就是「床沿」——视觉悬崖实验里那道看不见的落差。
CRIB_HALF_X = 0.36
CRIB_HALF_Y = 0.62
# 会走之后，活动范围不再限于婴儿床。地板是连续的：床沿那件事只在 S0/S1 成立。
ROOM_HALF_X = 2.0
ROOM_HALF_Y = 2.4


def edge_distance(pos: Vec3) -> float:
    """到床沿的最近距离（米）。已经越过边缘就是 0。"""
    x, y = pos[0], pos[1]
    return max(0.0, min(CRIB_HALF_X - abs(x), CRIB_HALF_Y - abs(y)))


def room_distance(pos: Vec3) -> float:
    """到房间墙的最近距离。会走之后，「走得远不远」由它决定。"""
    x, y = pos[0], pos[1]
    return max(0.0, min(ROOM_HALF_X - abs(x), ROOM_HALF_Y - abs(y)))


@dataclass
class WorldObject:
    """一个可被感知/可被作用的物体。"""

    id: str
    kind: str  # furniture | toy | person | body_part | light | fabric
    label: str
    pos: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (0.3, 0.3, 0.3)
    # 感知量（都不是"文案"）
    luminance: float = 0.5
    loudness: float = 0.0
    texture: str = "smooth"
    temperature: float = 0.5  # 0 冷 1 热
    # 行为
    movement: str = "still"  # still | oscillate | rotate | breathe | approach
    phase: float = 0.0
    # 发条玩具式的运动：每 period_min 分钟里只动 duty_min 分钟。
    # 没有 duty_min 就是一直在动（呼吸、心跳那种）。
    period_min: float = 15.0
    duty_min: float | None = None
    reachable: bool = False
    # 阶段闸门：`min_stage` 之前，这个物体的**某些通道**在适配器里根本不存在
    no_face_until: str = "S1"
    no_motion_until: str = "S1"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def moving(self, t: float | None = None) -> bool:
        """此刻是否在动。``t`` 是当天的小时数；传 None 表示不看时间（"会动的东西"）。"""
        if self.movement == "still":
            return False
        if self.duty_min is None or t is None:
            return True
        cycle = (float(t) * 60.0 + self.phase * 60.0) % max(1e-6, self.period_min)
        return cycle < self.duty_min

    def offset(self, t: float) -> Vec3:
        """随时间的小幅位移，让「运动」是真的（追踪类里程碑要靠它）。"""
        if self.movement == "oscillate":
            dx = 0.12 * math.sin(t * 1.1 + self.phase)
            return (self.pos[0] + dx, self.pos[1], self.pos[2])
        if self.movement == "rotate":
            return self.pos
        if self.movement == "breathe":
            dz = 0.004 * math.sin(t * 0.9 + self.phase)
            return (self.pos[0], self.pos[1], self.pos[2] + dz)
        if self.movement == "approach":
            return self.pos
        return self.pos

    def to_dict(self, *, precise: bool = False) -> dict[str, Any]:
        def r6(x: float) -> float:
            # 存档用的快照把浮点收成 6 位，人读着舒服；
            # 但要交给另一具身体（transport）时不能收——舍入会改变显著度的比较结果，
            # 于是「本地跑」和「远端跑」同一天会产出不同的 episode。
            return float(x) if precise else round(float(x), 6)

        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "pos": list(self.pos),
            "size": list(self.size),
            "luminance": r6(self.luminance),
            "loudness": r6(self.loudness),
            "texture": self.texture,
            "temperature": r6(self.temperature),
            "movement": self.movement,
            "phase": r6(self.phase),
            "period_min": r6(self.period_min),
            "duty_min": None if self.duty_min is None else r6(self.duty_min),
            "reachable": self.reachable,
            "no_face_until": self.no_face_until,
            "no_motion_until": self.no_motion_until,
            "tags": list(self.tags),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldObject:
        return cls(
            id=str(d["id"]),
            kind=str(d["kind"]),
            label=str(d["label"]),
            pos=tuple(d.get("pos", (0, 0, 0))),  # type: ignore[arg-type]
            size=tuple(d.get("size", (0.3, 0.3, 0.3))),  # type: ignore[arg-type]
            luminance=float(d.get("luminance", 0.5)),
            loudness=float(d.get("loudness", 0.0)),
            texture=str(d.get("texture", "smooth")),
            temperature=float(d.get("temperature", 0.5)),
            movement=str(d.get("movement", "still")),
            phase=float(d.get("phase", 0.0)),
            period_min=float(d.get("period_min", 15.0)),
            duty_min=(None if d.get("duty_min") is None else float(d["duty_min"])),
            reachable=bool(d.get("reachable", False)),
            no_face_until=str(d.get("no_face_until", "S1")),
            no_motion_until=str(d.get("no_motion_until", "S1")),
            tags=list(d.get("tags", [])),
            extra=dict(d.get("extra", {})),
        )


@dataclass
class WorldState:
    """世界的可序列化快照。崩溃恢复时从它 + 最后一条 episode 续跑。"""

    day: int = 0
    clock_hours: float = 6.0  # 一天内的绝对小时（0–24，跨天由 day 进位）
    light: float = 0.5
    objects: dict[str, WorldObject] = field(default_factory=dict)
    # 环境音
    ambient_db: float = 0.15
    # 习惯化计数：object_id -> 连续注视次数（docs/03 M0.3 用第 5 次与第 1 次比）
    fixation_counts: dict[str, int] = field(default_factory=dict)
    # 运动状态：上一次看到它时是否在动，以及这次动是从哪一刻开始的（绝对小时）
    motion_seen: dict[str, bool] = field(default_factory=dict)
    motion_onset: dict[str, float] = field(default_factory=dict)
    # 这一格他关注的目标（刚指过的东西）：编译动作时优先给它一个候选
    focus_target: str | None = None
    # 照料者
    caregiver_present: bool = True
    caregiver_mood: float = 0.6
    caregiver_last_response_min: float | None = None
    # S1 起才存在的东西：被盖住的玩具（客体永久性）、偶尔来访的陌生人（陌生人焦虑）
    occluded: dict[str, bool] = field(default_factory=dict)
    stranger_present: bool = False
    # 婴儿自己在床上的位置（爬行会改变它；床沿就是「视觉悬崖」）
    baby_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 上一格指过的东西：指完之后目光要跟过去（共同注意）
    joint_target: str | None = None
    # 大人收着、暂时不给的东西（延迟满足的那个「等」）
    withheld: str = ""
    # 三张卡片是否正被举在眼前（没有举起来就没有「挑」这件事）
    shown_cards: bool = False
    # 统计（里程碑证据来源，全部是**行为计数**而非模型自述）
    counters: dict[str, int] = field(default_factory=dict)

    # -- 查询 ---------------------------------------------------------------
    def obj(self, oid: str) -> WorldObject | None:
        return self.objects.get(oid)

    def visible_objects(self) -> list[WorldObject]:
        """此刻他真的能看见的东西。

        被盖住的不在视野里（这正是客体永久性要测的东西）；陌生人只有来访时才有；
        自己的身体部件不算「视野里的对象」。
        """
        out: list[WorldObject] = []
        for o in self.objects.values():
            if o.kind == "body_part":
                continue
            if self.occluded.get(o.id):
                continue
            if o.id == "stranger" and not self.stranger_present:
                continue
            out.append(o)
        return out

    def bump(self, key: str, n: int = 1) -> int:
        self.counters[key] = self.counters.get(key, 0) + n
        return self.counters[key]

    def count(self, key: str) -> int:
        return self.counters.get(key, 0)

    @property
    def abs_hours(self) -> float:
        """从出生起算的绝对小时数。跨天比较时间差要用它，不要用 clock_hours。"""
        return self.day * 24.0 + self.clock_hours

    # 运动起始后的「新异窗」：这段时间里习惯了的东西重新变成新的。
    # 必须**短于**采样间隔（30 分钟），否则每一帧都在窗内，去习惯化就永远看不见
    # （表现是唤醒度一条直线，M0.4 恒为 0）。
    ONSET_WINDOW_HOURS = 0.25

    def motion_onset_now(self, oid: str) -> bool:
        """这个对象是不是**刚刚**动起来的（用于定向反射/追踪）。"""
        since = self.motion_onset.get(oid)
        if since is None:
            return False
        return (self.abs_hours - since) <= self.ONSET_WINDOW_HOURS

    # -- 时间推进 -----------------------------------------------------------
    def advance(self, hours: float) -> None:
        self.clock_hours += float(hours)
        while self.clock_hours >= 24.0:
            self.clock_hours -= 24.0
            self.day += 1
        self.light = daylight(self.clock_hours)

    # -- 序列化 -------------------------------------------------------------
    def to_dict(self, *, precise: bool = False) -> dict[str, Any]:
        def r6(x: float) -> float:
            return float(x) if precise else round(float(x), 6)

        return {
            "day": self.day,
            "clock_hours": r6(self.clock_hours),
            "light": r6(self.light),
            "objects": {k: v.to_dict(precise=precise) for k, v in sorted(self.objects.items())},
            "ambient_db": r6(self.ambient_db),
            "fixation_counts": dict(sorted(self.fixation_counts.items())),
            "motion_seen": dict(sorted(self.motion_seen.items())),
            "motion_onset": {k: r6(v) for k, v in sorted(self.motion_onset.items())},
            "caregiver_present": self.caregiver_present,
            "caregiver_mood": r6(self.caregiver_mood),
            "caregiver_last_response_min": self.caregiver_last_response_min,
            "occluded": dict(sorted(self.occluded.items())),
            "stranger_present": self.stranger_present,
            "baby_pos": [r6(x) for x in self.baby_pos],
            "joint_target": self.joint_target,
            "withheld": self.withheld,
            "shown_cards": self.shown_cards,
            "counters": dict(sorted(self.counters.items())),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldState:
        return cls(
            day=int(d.get("day", 0)),
            clock_hours=float(d.get("clock_hours", 6.0)),
            light=float(d.get("light", 0.5)),
            objects={k: WorldObject.from_dict(v) for k, v in (d.get("objects") or {}).items()},
            ambient_db=float(d.get("ambient_db", 0.15)),
            fixation_counts=dict(d.get("fixation_counts") or {}),
            motion_seen={k: bool(v) for k, v in (d.get("motion_seen") or {}).items()},
            motion_onset={k: float(v) for k, v in (d.get("motion_onset") or {}).items()},
            caregiver_present=bool(d.get("caregiver_present", True)),
            caregiver_mood=float(d.get("caregiver_mood", 0.6)),
            caregiver_last_response_min=d.get("caregiver_last_response_min"),
            occluded={k: bool(v) for k, v in (d.get("occluded") or {}).items()},
            stranger_present=bool(d.get("stranger_present", False)),
            baby_pos=tuple(d.get("baby_pos") or (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            joint_target=(None if d.get("joint_target") is None else str(d["joint_target"])),
            withheld=str(d.get("withheld", "")),
            shown_cards=bool(d.get("shown_cards", False)),
            counters=dict(d.get("counters") or {}),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(canonical_json(self.to_dict()), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> WorldState | None:
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


def daylight(hour: float) -> float:
    """一天里的「照度」。简化成正弦：6 点起亮，18 点最暗。"""
    # 6:00 = 0.15，12:00 = 1.0，18:00 = 0.15，0:00 = 0.05
    phase = (hour - 6.0) / 24.0 * 2 * math.pi
    base = (math.sin(phase) + 1.0) / 2.0
    return max(0.05, 0.05 + 0.95 * base)
