"""世界的出生现场：一间育婴房。

刻意小：S0 的婴儿只有光、声、触、本体觉。房间里的东西必须少到「每个都能被看见」，
否则早期里程碑（定向/追踪/习惯化）会被噪声淹没，日志也无法人工判读。
"""

from __future__ import annotations

from persona.rng import derive, uniform
from persona.world.objects import WorldObject, WorldState


def create_world(*, individual_id: str, birth_iso: str, seed_salt: str = "v1") -> WorldState:
    """确定性生成出生世界。同一个体同一天永远得到同一个房间。"""
    rng = derive(seed_salt, individual_id, "birth", birth_iso)

    def jitter(base: float, spread: float = 0.06) -> float:
        return round(min(1.0, max(0.0, base + uniform(rng, -spread, spread))), 6)

    objects: dict[str, WorldObject] = {}

    def add(o: WorldObject) -> None:
        objects[o.id] = o

    add(
        WorldObject(
            id="crib",
            kind="furniture",
            label="婴儿床",
            pos=(0.0, 0.0, 0.0),
            size=(0.7, 1.2, 0.35),
            luminance=jitter(0.35),
            texture="wood",
            temperature=0.45,
            reachable=False,
            tags=["container", "stable"],
        )
    )
    add(
        WorldObject(
            id="mobile",
            kind="toy",
            label="床铃",
            pos=(0.0, -0.45, 0.55),
            size=(0.18, 0.18, 0.18),
            luminance=jitter(0.72),
            loudness=jitter(0.08, 0.03),
            texture="plastic",
            movement="oscillate",
            phase=round(uniform(rng, 0.0, 1.57), 6),
            # 发条玩具：上满一次发条动 30 分钟，然后停 60 分钟。
            # 周期取 90 分钟（= 三个 30 分钟采样节拍），这样无论 phase 是多少，
            # 每三个采样里总有且只有一个落在「动」上——「开始动」的时刻每天都有，
            # 不会因为相位刚好错开就整天测不到（20/5 分钟的周期会踩这个坑）。
            period_min=90.0,
            duty_min=30.0,
            reachable=False,
            tags=["moving", "bright", "pendant"],
        )
    )
    add(
        WorldObject(
            id="window",
            kind="light",
            label="窗",
            pos=(1.2, 0.2, 1.4),
            size=(0.9, 0.05, 0.7),
            luminance=jitter(0.9, 0.04),
            movement="still",
            reachable=False,
            tags=["bright", "far"],
        )
    )
    add(
        WorldObject(
            id="blanket",
            kind="fabric",
            label="毯子",
            pos=(0.0, 0.25, 0.02),
            size=(0.5, 0.6, 0.02),
            luminance=jitter(0.55),
            texture="soft",
            temperature=0.6,
            reachable=True,
            tags=["contact", "near"],
        )
    )
    add(
        WorldObject(
            id="rattle",
            kind="toy",
            label="摇铃",
            pos=(0.16, 0.10, 0.06),
            size=(0.07, 0.07, 0.12),
            luminance=jitter(0.68),
            loudness=jitter(0.10, 0.03),
            texture="plastic",
            temperature=0.55,
            reachable=True,
            # 这只摇铃会被毯子盖住：客体永久性要测的是「看不见了还去找」，
            # 所以它必须真的会从视野里消失，而不是一直摆在那里。
            tags=["near", "toy", "graspable", "occludable"],
        )
    )
    add(
        WorldObject(
            id="stranger",
            kind="person",
            label="陌生人",
            pos=(0.70, -0.55, 0.38),
            size=(0.40, 0.30, 0.60),
            luminance=jitter(0.55),
            loudness=jitter(0.18, 0.04),
            texture="skin",
            temperature=0.62,
            movement="breathe",
            reachable=False,
            no_face_until="S1",
            # 偶尔来一次的陌生人。她在场时，婴儿会更靠近照料者（陌生人焦虑）。
            tags=["social", "unknown", "visitor"],
        )
    )
    add(
        WorldObject(
            id="mother",
            kind="person",
            label="照料者",
            pos=(0.0, -0.35, 0.35),
            size=(0.4, 0.3, 0.6),
            luminance=jitter(0.62),
            loudness=jitter(0.22, 0.05),
            texture="skin",
            temperature=0.68,
            movement="breathe",
            reachable=False,
            no_face_until="S1",
            tags=["social", "warm", "voice"],
        )
    )
    add(
        WorldObject(
            id="my_hand",
            kind="body_part",
            label="自己的手",
            pos=(0.08, 0.02, 0.12),
            size=(0.05, 0.05, 0.02),
            luminance=jitter(0.6),
            texture="skin",
            temperature=0.66,
            reachable=True,
            tags=["self", "proprioceptive"],
        )
    )

    # -- 会走之后才用得上的东西 ------------------------------------------------
    # S2 的解锁不是「多几个标签」，而是房间里真的多了可以走过去、可以摆弄、
    # 可以照着做的东西。它们从第一天就在，只是 S0/S1 的身体够不着、也不会用。
    add(
        WorldObject(
            id="ball",
            kind="toy",
            label="球",
            pos=(1.30, -0.90, 0.06),
            size=(0.12, 0.12, 0.12),
            luminance=jitter(0.66),
            texture="rubber",
            movement="still",
            reachable=False,  # 走得到，但现在够不着
            tags=["floor", "toy", "rolls", "appears_at_S2"],
        )
    )
    add(
        WorldObject(
            id="blocks",
            kind="toy",
            label="积木",
            pos=(-1.10, 0.80, 0.05),
            size=(0.20, 0.20, 0.06),
            luminance=jitter(0.58),
            texture="wood",
            reachable=False,
            tags=["floor", "toy", "stackable", "appears_at_S2"],
        )
    )
    add(
        WorldObject(
            id="cup",
            kind="tool",
            label="杯子",
            pos=(0.90, 0.95, 0.07),
            size=(0.08, 0.08, 0.10),
            luminance=jitter(0.60),
            texture="plastic",
            reachable=False,
            tags=["floor", "tool", "holds_water", "appears_at_S2"],
        )
    )
    add(
        WorldObject(
            id="mirror",
            kind="furniture",
            label="镜子",
            pos=(-1.60, -1.20, 0.90),
            size=(0.50, 0.04, 0.80),
            luminance=jitter(0.82, 0.05),
            texture="glass",
            reachable=False,
            # 镜子里那张脸是他自己的。S2 之前他分不出来（docs/03 的 M2.4）。
            tags=["wall", "reflective", "appears_at_S2"],
        )
    )
    add(
        WorldObject(
            id="book",
            kind="toy",
            label="图画书",
            pos=(0.40, 1.30, 0.03),
            size=(0.24, 0.18, 0.02),
            luminance=jitter(0.70),
            texture="paper",
            reachable=False,
            tags=["floor", "toy", "pictures", "appears_at_S2"],
        )
    )
    # 三张卡片：照护者举着让人挑。挑哪张是稳定的偏好，不是随机数。
    add(
        WorldObject(
            id="pic_self",
            kind="picture",
            label="自己的照片",
            pos=(0.20, 0.24, 0.10),
            size=(0.12, 0.12, 0.01),
            luminance=jitter(0.72),
            texture="paper",
            reachable=True,
            tags=["near", "picture", "appears_at_S2"],
        )
    )
    add(
        WorldObject(
            id="pic_mother",
            kind="picture",
            label="妈妈的照片",
            pos=(0.24, 0.20, 0.10),
            size=(0.12, 0.12, 0.01),
            luminance=jitter(0.70),
            texture="paper",
            reachable=True,
            tags=["near", "picture", "appears_at_S2"],
        )
    )
    add(
        WorldObject(
            id="pic_duck",
            kind="picture",
            label="鸭子的照片",
            pos=(0.16, 0.20, 0.10),
            size=(0.12, 0.12, 0.01),
            luminance=jitter(0.68),
            texture="paper",
            reachable=True,
            tags=["near", "picture", "appears_at_S2"],
        )
    )

    world = WorldState(
        day=0,
        clock_hours=6.0,
        objects=objects,
        ambient_db=jitter(0.15, 0.03),
        caregiver_present=True,
        caregiver_mood=round(uniform(rng, 0.5, 0.75), 6),
    )
    return world


def reset_day(world: WorldState, *, day: int) -> None:
    """跨天时重置：照料者在场、照度按新一天算。物体位置保留（世界是连续的）。"""
    world.day = day
    world.caregiver_present = True
    world.stranger_present = False
    world.occluded = {}
    world.joint_target = None
    world.withheld = ""
    world.shown_cards = False
    # 「今天说过多少句」是按天累计的：新的一天，话又有了分量
    world.counters.pop("said_today", None)
    world.counters.pop("pretend_today", None)
    world.counters.pop("babble_today", None)
    world.counters.pop("walk_today", None)
    # 「今天各做过几次」跨天清零（否则第二天早上他什么也不想做了）
    for key in [k for k in world.counters if k.startswith("done_today:")]:
        world.counters.pop(key, None)
