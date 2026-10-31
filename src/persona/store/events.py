"""追加式事件日志（每行一条 JSON）。

为什么不用数据库当唯一真相：事件日志是**只追加**的，天然可 diff、可 grep、可手工回放；
SQLite 只是它的可查询镜像。一次 append 会同时写文件与表，顺序是「先文件后表」——
宁可表里多一条待补的，也不可文件缺一条。

行格式（docs/13）：{seq, ts, type, ref, actor, version}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from persona import __version__
from persona.errors import StoreError

#: 事件类型闭集。新增类型必须同时更新 docs/13 与本表——否则 verify 会报「未知事件」。
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "person.born",
        "episode.created",
        "consolidation.step",
        "trait.updated",
        "memory.written",
        "memory.superseded",
        "memory.archived",
        "relationship.updated",
        "relationship.attachment.classified",
        "milestone.evaluated",
        "stage.advanced",
        "developmental_delay",
        "narrative.revised",
        "identity.crisis",
        "safety.blocked",
        "safety.rewritten",
        "degraded",
        "system.sleep",
        "system.wake",
        "run.started",
        "run.finished",
    }
)

REQUIRED_FIELDS = ("seq", "ts", "type", "actor", "version")


class EventLog:
    """一天一天往下写的账本。`append` 是同进程内唯一写入点。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = self._scan_last_seq()

    def _scan_last_seq(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = max(last, int(json.loads(line)["seq"]))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise StoreError(f"事件日志损坏（{self.path}）：{exc}") from exc
        return last

    @property
    def last_seq(self) -> int:
        return self._seq

    def append(
        self,
        type: str,
        *,
        ts: str,
        ref: str | None = None,
        actor: str = "core",
        **extra: Any,
    ) -> dict[str, Any]:
        if type not in EVENT_TYPES:
            raise StoreError(f"未知事件类型：{type!r}")
        self._seq += 1
        row: dict[str, Any] = {
            "seq": self._seq,
            "ts": ts,
            "type": type,
            "ref": ref,
            "actor": actor,
            "version": __version__,
        }
        row.update(extra)
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # 掉电时不丢最后一条
        return row

    # -- 读 -----------------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read(self) -> list[dict[str, Any]]:
        return list(self)

    def of_type(self, type_: str) -> list[dict[str, Any]]:
        return [e for e in self if e.get("type") == type_]

    # -- 校验 ---------------------------------------------------------------
    def verify(self) -> list[str]:
        """seq 连续、字段齐全、类型合法、时间戳可解析。返回问题列表。"""
        problems: list[str] = []
        expected = 1
        for i, ev in enumerate(self, start=1):
            missing = [f for f in REQUIRED_FIELDS if f not in ev]
            if missing:
                problems.append(f"第 {i} 行缺字段 {missing}")
            if ev.get("seq") != expected:
                problems.append(f"第 {i} 行 seq={ev.get('seq')}，期望 {expected}")
            expected = int(ev.get("seq", expected)) + 1
            if ev.get("type") not in EVENT_TYPES:
                problems.append(f"第 {i} 行未知类型 {ev.get('type')!r}")
            ts = str(ev.get("ts", ""))
            if "T" not in ts or ("+" not in ts[10:] and "Z" not in ts):
                problems.append(f"第 {i} 行 ts 非法：{ts!r}")
        return problems
