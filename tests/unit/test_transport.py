"""身体在别的进程里时，core 跑出来的东西必须一模一样。

这是整个 transport 的唯一验收标准：**同样的种子、同样的输入，产出的 episode
逐字节相同**。如果不同，说明序列化丢了东西（新异度、追踪标记、习惯化计数），
那这条通道就是错的，跑得通也不算数。
"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from persona.config import load
from persona.runtime.runner import open_simulation
from persona.transport import protocol, ws
from persona.transport.body_io import RemoteBody
from persona.transport.server import BodyServer
from persona.transport.service import BodyService

TMP_ROOT = Path(__file__).resolve().parents[2] / "data" / "tmp"


def _scratch() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"tp-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def _config(tmp: Path, name: str) -> Path:
    # 每个体一个子目录：世界/身体/词汇表快照是按 store.path 的父目录存的，
    # 两个体挤在一个目录里，第二个会读到第一个的身体状态。
    home = tmp / name
    home.mkdir(parents=True, exist_ok=True)
    cfg = {
        "individual": {"id": "soul-tp", "name": "小传", "birth": "2026-01-01T06:00:00+08:00"},
        "store": {"path": str(home / f"{name}.db"), "events": str(home / f"{name}.jsonl")},
        "clock": {"mode": "mock", "start": "2026-01-01T06:00:00+08:00"},
        "runtime": {"diary": False},
    }
    path = tmp / f"{name}.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return path


class WebSocketTest(unittest.TestCase):
    def test_frame_roundtrip_masked_and_plain(self) -> None:
        for mask in (False, True):
            payload = b'{"hello":"world"}'
            stream = ws.memory_stream(ws.frame(payload, ws.OP_TEXT, mask=mask))
            opcode, got = ws.read_frame(stream)
            self.assertEqual(opcode, ws.OP_TEXT)
            self.assertEqual(got, payload)

    def test_long_frame_uses_126_length(self) -> None:
        payload = b"x" * 300
        stream = ws.memory_stream(ws.frame(payload, ws.OP_TEXT))
        opcode, got = ws.read_frame(stream)
        self.assertEqual(got, payload)
        self.assertEqual(ws.accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_server_accepts_handshake_and_answers_hello(self) -> None:
        with BodyServer(BodyService(), port=0) as server:
            client = ws.WebSocket.connect(server.host, server.port, timeout=5.0)
            try:
                client.send_text(protocol.encode(protocol.hello("core")))
                reply = protocol.decode(client.recv_text())
            finally:
                client.close()
        self.assertEqual(reply["type"], "hello")
        self.assertEqual(reply["protocol"], protocol.PROTOCOL)
        self.assertGreater(reply["primitives"], 40)


class RemoteAgreementTest(unittest.TestCase):
    """本地身体与远端身体跑同一天，逐字节比对。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = _scratch()
        cls.server = BodyServer(BodyService(noise=0.05), port=0).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run_day(self, name: str, *, remote: bool) -> list[dict]:
        cfg = load(["config/default.json", str(_config(self.tmp, name))])
        sim = open_simulation(cfg)
        body = None
        try:
            if remote:
                body = RemoteBody(self.server.url)
                sim.attach_body(body)
            sim.born()
            sim.run_day(0)
            return sim.repo.episodes_of_day("2026-01-01")
        finally:
            if body is not None:
                body.close()
            sim.close()

    def test_same_day_same_bytes(self) -> None:
        local = self._run_day("local", remote=False)
        remote = self._run_day("remote", remote=True)
        self.assertEqual(len(local), len(remote))
        self.assertGreaterEqual(len(local), 30)
        for a, b in zip(local, remote):
            self.assertEqual(a["id"], b["id"])
            self.assertEqual(a["action"]["primitive"], b["action"]["primitive"])
            # 夜里那条是 core 自己补的（身体不在场），没有 novel 字段，所以只比白天。
            if a["kind"] != "sleep":
                self.assertEqual(a["observation"]["novelty"], b["observation"]["novelty"])
                self.assertEqual(a["appraisal"], b["appraisal"])
                self.assertEqual(a["salience"], b["salience"])
            # 最硬的一条：整条 episode 必须逐字段相同。
            self.assertEqual(a, b)

    def test_habituation_survives_the_wire(self) -> None:
        # 习惯化计数由身体算、core 存。跨进程跑完一天后，计数必须真的在长。
        cfg = load(["config/default.json", str(_config(self.tmp, "habit"))])
        sim = open_simulation(cfg)
        body = RemoteBody(self.server.url)
        try:
            sim.attach_body(body)
            sim.born()
            sim.run_day(0)
            counts = dict(sim.world.fixation_counts)
        finally:
            body.close()
            sim.close()
        self.assertTrue(counts, "远端跑完一天，习惯化计数不能是空的")
        self.assertGreater(sum(counts.values()), 0)


if __name__ == "__main__":
    unittest.main()
