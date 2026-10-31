"""身体服务进程：监听一个端口，替 core 看、列、编译。

一条连接 = 一具身体。多个 core 连上来互不影响（每次都从快照重建世界），
因为它们本来就各自持有自己的世界状态——身体是无状态的。
"""

from __future__ import annotations

import socketserver
import threading
from typing import Any

from persona.transport import protocol, ws
from persona.transport.service import BodyService


class _Handler(socketserver.BaseRequestHandler):
    service: BodyService  # 由 BodyServer 注入

    def handle(self) -> None:
        sock = self.request
        try:
            sock.settimeout(30.0)
            rfile = sock.makefile("rb", buffering=0)
            wfile = sock.makefile("wb", buffering=0)
            try:
                ws.server_handshake(rfile, wfile)
            except (ws.WebSocketError, ws.ConnectionClosed, OSError):
                return
            conn = ws.WebSocket(sock, rfile, wfile, mask=False)
            while True:
                try:
                    text = conn.recv_text()
                except (ws.ConnectionClosed, OSError):
                    break
                except ws.WebSocketError as exc:
                    try:
                        conn.send_text(protocol.encode(protocol.error(str(exc))))
                    except OSError:
                        pass
                    break
                try:
                    message = protocol.decode(text)
                except protocol.ProtocolError as exc:
                    conn.send_text(protocol.encode(protocol.error(str(exc))))
                    continue
                reply = self.service.handle(message)
                conn.send_text(protocol.encode(reply))
        finally:
            try:
                sock.close()
            except OSError:
                pass


class BodyServer:
    def __init__(self, service: BodyService | None = None, *, host: str = "127.0.0.1", port: int = 8767) -> None:
        self.service = service or BodyService()
        handler = type("Handler", (_Handler,), {"service": self.service})
        self.httpd = socketserver.ThreadingTCPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.host = host
        self.port = int(self.httpd.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self, *, background: bool = True) -> "BodyServer":
        if not background:
            self.httpd.serve_forever()
            return self
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="body-server", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "BodyServer":
        return self.start(background=True)

    def __exit__(self, *exc: object) -> None:
        self.stop()


def serve(host: str = "127.0.0.1", port: int = 8767, *, noise: float = 0.05) -> BodyServer:
    server = BodyServer(BodyService(noise=noise), host=host, port=port)
    return server


def describe(server: BodyServer) -> str:
    hello: dict[str, Any] = server.service.handle(protocol.hello())
    return (
        f"身体服务 {server.url}  原语 {hello.get('primitives')} 个  "
        f"注册表 {str(hello.get('registry_hash'))[:12]}  协议 {hello.get('protocol')}"
    )
