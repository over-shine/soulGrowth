"""core 这一侧的连接：一次请求、一次应答，同步阻塞。

刻意不做异步、不做重连池：一个 core 进程只有一具身体，一问一答就够了。
连接断了就抛 ``TransportError``，由上层决定是降级还是退出——身体不在，
core 不该假装还活着。
"""

from __future__ import annotations

from urllib.parse import urlparse

from persona.errors import TransportError
from persona.transport import protocol, ws


class BodyClient:
    def __init__(self, url: str, *, timeout: float = 10.0) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "http", ""):
            raise TransportError(f"只支持 ws:// 地址，收到 {url}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = int(parsed.port or 8767)
        self.timeout = timeout
        self.sock: ws.WebSocket | None = None
        self._seq = 0
        self._hello: dict | None = None

    # -- 生命周期 ------------------------------------------------------------

    def connect(self) -> None:
        try:
            self.sock = ws.WebSocket.connect(self.host, self.port, timeout=self.timeout)
        except OSError as exc:
            raise TransportError(f"连不上身体服务 {self.host}:{self.port}：{exc}") from exc

    def hello(self) -> dict:
        if self._hello is None:
            self._hello = self.request(protocol.hello("core"), type_expected="hello")
        return self._hello

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    # -- 一次往返 ------------------------------------------------------------

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def request(self, message: dict, *, type_expected: str | None = None) -> dict:
        if self.sock is None:
            raise TransportError("连接已经关了")
        seq = message.get("seq")
        try:
            self.sock.send_text(protocol.encode(message))
            reply = protocol.decode(self.sock.recv_text())
        except ws.ConnectionClosed as exc:
            self.sock = None
            raise TransportError(f"身体服务断开了：{exc}") from exc
        except (OSError, ws.WebSocketError) as exc:
            self.sock = None
            raise TransportError(f"和身体服务通信失败：{exc}") from exc

        protocol.check(reply, seq=seq)
        if type_expected and reply.get("type") != type_expected:
            raise TransportError(f"期望 {type_expected}，收到 {reply.get('type')}")
        return reply

    def __enter__(self) -> "BodyClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
