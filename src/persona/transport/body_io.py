"""身体的两副面孔：进程内的，和网线那头的。

runner 只认这三件事——看、列候选、编译。至于这具身体在北京还是在 localhost，
它不需要知道。``LocalBody`` 就是原来那几个对象；``RemoteBody`` 把同样的调用
变成一次 WS 往返。
"""

from __future__ import annotations

from typing import Any

from persona.body.compiler import Candidate, Compiled, Compiler
from persona.body.perception import Bundle, Perception
from persona.errors import TransportError
from persona.types import Intent
from persona.world.objects import WorldState


class LocalBody:
    """同一进程里直接调用。行为与加 transport 之前完全一致。"""

    def __init__(self, perception: Perception, compiler: Compiler) -> None:
        self.perception = perception
        self.compiler = compiler

    def bundle(
        self,
        world: WorldState,
        body: Any,
        *,
        stage: str,
        day: int,
        episode_id: str,
        clock_hours: float,
        goal_focus: str | None = None,
    ) -> Bundle:
        return self.perception.bundle(
            world,
            body,
            stage=stage,
            day=day,
            episode_id=episode_id,
            clock_hours=clock_hours,
            goal_focus=goal_focus,
        )

    def candidates(
        self,
        bundle: Bundle,
        body: Any,
        *,
        stage: str,
        world: WorldState,
        lexicon: Any,
        limit: int = 12,
        social: float = 0.0,
    ) -> list[Candidate]:
        return self.compiler.candidates(
            bundle,
            body,
            stage=stage,
            world=world,
            lexicon=lexicon,
            limit=limit,
            social=social,
        )

    def compile(
        self,
        primitive: str,
        params: dict[str, Any],
        *,
        stage: str,
        body: Any,
        world: WorldState,
        lexicon: Any,
        intent: Intent | None = None,
        bundle: Bundle | None = None,
        social: float = 0.0,
    ) -> Compiled:
        return self.compiler.compile(
            primitive,
            params,
            stage=stage,
            body=body,
            world=world,
            lexicon=lexicon,
            intent=intent,
            bundle=bundle,
            social=social,
        )

    def close(self) -> None:
        return None


class RemoteBody:
    """身体在另一个进程里：每次调用一次 WS 往返（本地回环，亚毫秒级）。"""

    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        from persona.transport.client import BodyClient

        self.client = BodyClient(url, timeout=timeout)
        self.client.connect()
        self.hello = self.client.hello()

    # -- 三次往返 ------------------------------------------------------------

    def bundle(
        self,
        world: WorldState,
        body: Any,
        *,
        stage: str,
        day: int,
        episode_id: str,
        clock_hours: float,
        goal_focus: str | None = None,
    ) -> Bundle:
        from persona.transport import protocol

        reply = self.client.request(
            protocol.observe(
                self.client.next_seq(),
                stage=stage,
                day=day,
                episode_id=episode_id,
                clock_hours=clock_hours,
                goal_focus=goal_focus,
                world=world.to_dict(precise=True),
                body=body.to_dict(precise=True),
            )
        )
        if reply.get("type") != "bundle":
            raise TransportError(f"期望 bundle，收到 {reply.get('type')}")
        # 习惯化是身体记的：把它算出来的计数合并回 core 的世界状态
        patch = reply.get("perception") or {}
        world.fixation_counts = {k: int(v) for k, v in (patch.get("fixation_counts") or {}).items()}
        world.motion_seen = {k: bool(v) for k, v in (patch.get("motion_seen") or {}).items()}
        world.motion_onset = {k: float(v) for k, v in (patch.get("motion_onset") or {}).items()}
        return protocol.bundle_from_wire(reply["bundle"])

    def candidates(
        self,
        bundle: Bundle,
        body: Any,
        *,
        stage: str,
        world: WorldState,
        lexicon: Any,
        limit: int = 12,
        social: float = 0.0,
    ) -> list[Candidate]:
        from persona.transport import protocol

        reply = self.client.request(
            protocol.afford(
                self.client.next_seq(),
                stage=stage,
                world=world.to_dict(precise=True),
                body=body.to_dict(precise=True),
                lexicon=lexicon.to_dict(),
                social=social,
                limit=limit,
                bundle=protocol.bundle_to_wire(bundle),
            )
        )
        if reply.get("type") != "candidates":
            raise TransportError(f"期望 candidates，收到 {reply.get('type')}")
        return [protocol.candidate_from_wire(c) for c in reply.get("candidates") or []]

    def compile(
        self,
        primitive: str,
        params: dict[str, Any],
        *,
        stage: str,
        body: Any,
        world: WorldState,
        lexicon: Any,
        intent: Intent | None = None,
        bundle: Bundle | None = None,
        social: float = 0.0,
    ) -> Compiled:
        from persona.transport import protocol

        if bundle is None:
            raise TransportError("远端编译必须带 bundle（前端条件在 bundle 上）")
        reply = self.client.request(
            protocol.compile_action(
                self.client.next_seq(),
                stage=stage,
                world=world.to_dict(precise=True),
                body=body.to_dict(precise=True),
                lexicon=lexicon.to_dict(),
                social=social,
                primitive=primitive,
                params=dict(params),
                intent=protocol.intent_to_wire(intent),
                bundle=protocol.bundle_to_wire(bundle),
            )
        )
        if reply.get("type") != "compiled":
            raise TransportError(f"期望 compiled，收到 {reply.get('type')}")
        return protocol.compiled_from_wire(reply["compiled"])

    def close(self) -> None:
        self.client.close()
