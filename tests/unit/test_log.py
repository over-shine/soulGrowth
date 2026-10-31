"""日志必须是可以直接交给任何 UTF-8 读取器的字节流。"""

from __future__ import annotations

import logging
import shutil
import unittest
import uuid
from pathlib import Path

from persona.log import get, log_path_for, setup

#: 系统的临时目录在受限沙箱里不可写，测试自己的临时目录放在工作区内。
TMP_ROOT = Path(__file__).resolve().parents[2] / "data" / "tmp"


def _scratch() -> Path:
    """自己建目录：tempfile.mkdtemp 在 Windows 上会设 0o700 的 ACL，
    受限令牌下反而写不进去。"""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"log-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def _flush() -> None:
    for handler in logging.getLogger("persona").handlers:
        handler.flush()


class LogEncodingTest(unittest.TestCase):
    def tearDown(self) -> None:
        setup(None)
        _flush()

    def test_file_is_utf8_with_lf(self) -> None:
        tmp = _scratch()
        try:
            path = tmp / "logs" / "persona.log"
            setup(path)
            get("test").info("出生：%s，第 %d 天", "小满", 0)
            _flush()

            raw = path.read_bytes()
            text = raw.decode("utf-8")  # 不是 UTF-8 这里就会抛
            self.assertIn("出生：小满，第 0 天", text)
            self.assertNotIn(b"\r\n", raw)
            self.assertTrue(raw.endswith(b"\n"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_two_runs_append_not_truncate(self) -> None:
        tmp = _scratch()
        try:
            path = tmp / "persona.log"
            setup(path)
            get("test").info("第一次")
            _flush()
            setup(path)
            get("test").info("第二次")
            _flush()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([line.rsplit(" ", 1)[-1] for line in lines], ["第一次", "第二次"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_log_lives_beside_the_database(self) -> None:
        self.assertEqual(
            log_path_for("data/soul-001/soul.db"),
            Path("data/soul-001/logs/persona.log").resolve(),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
