"""一个够用的 WebSocket（RFC 6455）实现，只用标准库。

为什么自己写：这个项目有一条「零第三方依赖」的硬约束（docs/16）。而 body 与 core
要分成两个进程、走一条双向消息通道，WS 是最省事的选择——两边都是文本 JSON，
不需要分片、不需要扩展、不需要压缩。

支持范围（故意很小，测得完）：
- 握手（客户端/服务端），`Sec-WebSocket-Accept` 校验
- 文本帧、关闭帧、ping/pong
- 长度 125 / 126 / 127 三种形式；单帧上限 ``MAX_FRAME``
- 客户端发出的帧加掩码（协议强制），服务端不加

不支持：分片（continuation）、permessage-deflate、二进制业务消息。收到就用不上。
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import socket
import struct
from typing import BinaryIO

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

MAX_FRAME = 8 * 1024 * 1024


class WebSocketError(RuntimeError):
    """握手或帧不合法。"""


class ConnectionClosed(WebSocketError):
    """对端正常关闭（不是错误，是结束）。"""


def accept_key(key: str) -> str:
    """`Sec-WebSocket-Accept` = base64(sha1(key + GUID))。"""
    digest = hashlib.sha1((key + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


# -- 帧 ----------------------------------------------------------------------


def frame(payload: bytes, opcode: int = OP_TEXT, *, mask: bool = False) -> bytes:
    """把一段负载打成帧。``mask`` 只应由客户端设为 True。"""
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))  # FIN=1，无扩展位
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", length))
    if not mask:
        return bytes(header) + payload
    key = os.urandom(4)
    masked = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return bytes(header) + key + masked


def read_frame(stream: BinaryIO) -> tuple[int, bytes]:
    """读一帧，返回 (opcode, payload)。对端关闭时抛 ConnectionClosed。"""
    head = _read_exact(stream, 2)
    b0, b1 = head[0], head[1]
    fin = bool(b0 & 0x80)
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", _read_exact(stream, 2))
    elif length == 127:
        (length,) = struct.unpack("!Q", _read_exact(stream, 8))
    if length > MAX_FRAME:
        raise WebSocketError(f"单帧太大：{length} 字节")
    if not fin:
        raise WebSocketError("不支持分片帧")
    key = _read_exact(stream, 4) if masked else b""
    payload = _read_exact(stream, length) if length else b""
    if masked:
        payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise ConnectionClosed("连接已关闭")
        buf.extend(chunk)
    return bytes(buf)


# -- 握手 --------------------------------------------------------------------


def _read_headers(stream: BinaryIO) -> tuple[str, dict[str, str]]:
    raw = bytearray()
    while not raw.endswith(b"\r\n\r\n"):
        chunk = stream.read(1)
        if not chunk:
            raise ConnectionClosed("握手期间连接关闭")
        raw.extend(chunk)
        if len(raw) > 64 * 1024:
            raise WebSocketError("握手头太大")
    text = raw.decode("latin-1")
    lines = text.split("\r\n")
    request_line = lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return request_line, headers


def server_handshake(rfile: BinaryIO, wfile: BinaryIO) -> dict[str, str]:
    """在服务端读客户端握手并回 101。返回请求头。"""
    request_line, headers = _read_headers(rfile)
    if "upgrade" not in headers.get("connection", "").lower():
        raise WebSocketError("不是 WebSocket 升级请求")
    if headers.get("upgrade", "").lower() != "websocket":
        raise WebSocketError("Upgrade 头不是 websocket")
    key = headers.get("sec-websocket-key")
    if not key:
        raise WebSocketError("缺 Sec-WebSocket-Key")
    version = headers.get("sec-websocket-version", "")
    if version not in ("13", "8"):
        raise WebSocketError(f"不支持的 WS 版本：{version}")
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key(key)}\r\n"
        "\r\n"
    )
    wfile.write(response.encode("latin-1"))
    wfile.flush()
    return {"request_line": request_line, **headers}


class WebSocket:
    """一端的连接：文本收发、ping/pong、关闭。线程内使用。"""

    def __init__(self, sock: socket.socket, rfile: BinaryIO, wfile: BinaryIO, *, mask: bool) -> None:
        self.sock = sock
        self.rfile = rfile
        self.wfile = wfile
        self.mask = mask
        self.closed = False

    # -- 构造的两端 ----------------------------------------------------------

    @classmethod
    def connect(cls, host: str, port: int, *, timeout: float = 10.0) -> "WebSocket":
        sock = socket.create_connection((host, port), timeout=timeout)
        rfile = sock.makefile("rb", buffering=0)
        wfile = sock.makefile("wb", buffering=0)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        wfile.write(request.encode("latin-1"))
        status_line, headers = _read_headers(rfile)
        if "101" not in status_line:
            raise WebSocketError(f"握手失败：{status_line}")
        expected = accept_key(key)
        if headers.get("sec-websocket-accept") != expected:
            raise WebSocketError("Sec-WebSocket-Accept 不匹配")
        return cls(sock, rfile, wfile, mask=True)

    @classmethod
    def from_accepted_socket(cls, sock: socket.socket) -> "WebSocket":
        """服务端：已经握手完成的连接。"""
        rfile = sock.makefile("rb", buffering=0)
        wfile = sock.makefile("wb", buffering=0)
        return cls(sock, rfile, wfile, mask=False)

    # -- 收发 ----------------------------------------------------------------

    def send_text(self, text: str) -> None:
        self.wfile.write(frame(text.encode("utf-8"), OP_TEXT, mask=self.mask))
        self.wfile.flush()

    def send_ping(self, payload: bytes = b"") -> None:
        self.wfile.write(frame(payload, OP_PING, mask=self.mask))
        self.wfile.flush()

    def recv_text(self, *, max_bytes: int = MAX_FRAME) -> str:
        """收一条文本消息。自动回 pong、跳过不应出现的二进制帧。"""
        while True:
            opcode, payload = read_frame(self.rfile)
            if opcode == OP_TEXT:
                if len(payload) > max_bytes:
                    raise WebSocketError("消息太大")
                return payload.decode("utf-8")
            if opcode == OP_PING:
                self.wfile.write(frame(payload, OP_PONG, mask=self.mask))
                self.wfile.flush()
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                raise ConnectionClosed("对端发送了关闭帧")
            if opcode == OP_BINARY:
                raise WebSocketError("这条通道只走文本 JSON")
            raise WebSocketError(f"未知 opcode {opcode}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.wfile.write(frame(b"", OP_CLOSE, mask=self.mask))
            self.wfile.flush()
        except OSError:
            pass
        for f in (self.rfile, self.wfile):
            try:
                f.close()
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self) -> "WebSocket":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def handshake_bytes(key: str) -> bytes:  # 测试用：手工构造一次握手
    return f"GET / HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n\r\n".encode(
        "latin-1"
    )


def memory_stream(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)
