"""身体服务：把感知与编译放到另一个进程里。

这里刻意只做三件事——看、列得出做得到什么、把一个意图编译成一次具体动作。
它不评价、不选择、不记忆、不写库：那些是 core 的事（docs/02 的层次：L1 body
在 L2 kernel 之下）。所以服务本身是**无状态**的：每个请求带齐世界快照与身体快照。

唯一的例外是习惯化计数。感知会改 ``fixation_counts`` / ``motion_seen`` /
``motion_onset``——那些是「这具身体看过什么」，属于身体。所以应答里带一个
``perception`` 补丁，core 把它合并回自己的世界状态。这样两个进程跑出来的
episode 才能和单进程逐字节一致（test_transport 盯着这件事）。
"""

from __future__ import annotations

from typing import Any

from persona.body.compiler import Compiler
from persona.body.perception import Perception
from persona.body.primitives import default_registry
from persona.body.state import BodyState
from persona.language.lexicon import Lexicon
from persona.transport import protocol
from persona.world.objects import WorldState


class BodyService:
    """一条 WS 连接背后的大脑：无状态，可并发。"""

    def __init__(self, *, noise: float = 0.05, registry: Any | None = None) -> None:
        self.registry = registry or default_registry()
        self.perception = Perception(noise=noise)
        self.compiler = Compiler(self.registry)

    # -- 分派 ----------------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        seq = message.get("seq")
        kind = str(message.get("type") or "")
        if message.get("version") not in (None, protocol.VERSION):
            return protocol.error("版本不匹配", seq=seq)
        try:
            if kind == "hello":
                return self._hello()
            if kind == "observe":
                return self._observe(message)
            if kind == "afford":
                return self._afford(message)
            if kind == "compile":
                return self._compile(message)
            if kind == "ping":
                return {"type": "pong", "seq": seq, "version": protocol.VERSION}
            return protocol.error(f"不认识的请求：{kind}", kind="unknown", seq=seq)
        except Exception as exc:  # 身体不该把异常吐给 core 的调用栈，要吐拒绝码
            return protocol.error(f"{type(exc).__name__}: {exc}", kind="exception", seq=seq)

    # -- 各方法 --------------------------------------------------------------

    def _hello(self) -> dict[str, Any]:
        return {
            "version": protocol.VERSION,
            "type": "hello",
            "protocol": protocol.PROTOCOL,
            # 注册表把 hash 存成属性（不是方法）：它算一次就不该变
            "registry_hash": self.registry.hash,
            "primitives": len(self.registry.all()),
        }

    def _observe(self, message: dict[str, Any]) -> dict[str, Any]:
        seq = message.get("seq")
        world = WorldState.from_dict(message["world"])
        body = BodyState.from_dict(message["body"])
        bundle = self.perception.bundle(
            world,
            body,
            stage=str(message["stage"]),
            day=int(message["day"]),
            episode_id=str(message["episode_id"]),
            clock_hours=float(message["clock_hours"]),
            goal_focus=message.get("goal_focus"),
        )
        return {
            "version": protocol.VERSION,
            "type": "bundle",
            "seq": seq,
            "bundle": protocol.bundle_to_wire(bundle),
            # 习惯化是身体的记忆，core 没有权利自己编
            "perception": {
                "fixation_counts": dict(world.fixation_counts),
                "motion_seen": dict(world.motion_seen),
                "motion_onset": dict(world.motion_onset),
            },
        }

    def _afford(self, message: dict[str, Any]) -> dict[str, Any]:
        seq = message.get("seq")
        world = WorldState.from_dict(message["world"])
        body = BodyState.from_dict(message["body"])
        bundle = protocol.bundle_from_wire(message["bundle"])
        lexicon = Lexicon.from_dict(message.get("lexicon") or {})
        candidates = self.compiler.candidates(
            bundle,
            body,
            stage=str(message["stage"]),
            world=world,
            lexicon=lexicon,
            limit=int(message.get("limit", 12)),
            social=float(message.get("social", 0.0)),
        )
        return {
            "version": protocol.VERSION,
            "type": "candidates",
            "seq": seq,
            "candidates": [protocol.candidate_to_wire(c) for c in candidates],
        }

    def _compile(self, message: dict[str, Any]) -> dict[str, Any]:
        seq = message.get("seq")
        world = WorldState.from_dict(message["world"])
        body = BodyState.from_dict(message["body"])
        bundle = protocol.bundle_from_wire(message["bundle"])
        lexicon = Lexicon.from_dict(message.get("lexicon") or {})
        compiled = self.compiler.compile(
            str(message["primitive"]),
            dict(message.get("params") or {}),
            stage=str(message["stage"]),
            body=body,
            world=world,
            lexicon=lexicon,
            intent=protocol.intent_from_wire(message.get("intent")),
            bundle=bundle,
            social=float(message.get("social", 0.0)),
        )
        return {
            "version": protocol.VERSION,
            "type": "compiled",
            "seq": seq,
            "compiled": protocol.compiled_to_wire(compiled),
        }
