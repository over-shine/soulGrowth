"""运行日志：一律 UTF-8，换行一律 \\n。

Windows 上中文日志乱码有两个来源，这里各治一个：

1. 控制台流的编码。Python 会跟随控制台代码页（中文 Windows 是 cp936/GBK），
   一旦输出被重定向到文件，字节就按 GBK 编码落盘，随后任何 UTF-8 读取器都会
   看到乱码。``reconfigure_streams()`` 把 stdout/stderr 显式钉成 UTF-8。
2. 日志文件的编码。任何 ``open()`` 不带 ``encoding=`` 都吃系统默认编码，同样
   会把中文写坏。``setup()`` 用 ``FileHandler(encoding="utf-8")``。

日志是给人读的旁路记录，不参与真相：真相在 events.jsonl 与 SQLite 里。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%S"

ROOT = "persona"


def reconfigure_streams() -> None:
    """重定向时必须写 UTF-8；接在真控制台上时**不要动**。

    Windows 上 Python 连真控制台走的是宽字符 API，中文本来就正常；一旦把它改成
    UTF-8，反而是把 UTF-8 字节塞进按 GBK 解释的句柄，正好写出乱码。
    """
    for stream in (sys.stdout, sys.stderr):
        _reconfigure(stream)


def _reconfigure(stream: TextIO | None) -> None:
    if stream is None:
        return
    method = getattr(stream, "reconfigure", None)
    isatty = getattr(stream, "isatty", None)
    if method is None or (callable(isatty) and isatty()):
        return
    try:
        method(encoding="utf-8", errors="backslashreplace")
    except (ValueError, OSError):  # 已经分离或不可重配的流
        return


class _Utf8Handler(logging.FileHandler):
    """UTF-8 + 固定 \\n 换行。Windows 默认会把 \\n 写成 \\r\\n。"""

    def _open(self):  # type: ignore[override]
        return open(  # noqa: SIM115 - logging 负责关闭
            self.baseFilename,
            self.mode,
            encoding=self.encoding or "utf-8",
            errors=self.errors or "strict",
            newline="\n",
        )


def setup(path: str | Path | None, *, level: int = logging.INFO) -> logging.Logger:
    """配置 persona 的根 logger。``path=None`` 时写控制台。"""
    logger = logging.getLogger(ROOT)
    logger.setLevel(level)
    logger.propagate = False
    for old in list(logger.handlers):
        logger.removeHandler(old)
        old.close()

    if path is None:
        handler: logging.Handler = logging.StreamHandler(stream=sys.stdout)
    else:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = _Utf8Handler(target, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    logger.addHandler(handler)
    return logger


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}")


def log_path_for(store_path: str | Path) -> Path:
    """日志与数据放在一起：data/<个体>/logs/persona.log。"""
    return Path(store_path).resolve().parent / "logs" / "persona.log"
