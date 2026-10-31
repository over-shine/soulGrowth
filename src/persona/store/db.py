"""SQLite 连接与建表。**唯一**允许 `import sqlite3` 的地方（除 store 包内其它模块）。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from persona.errors import StoreError

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """薄封装：建表、事务、查询。业务层不写 SQL 字符串以外的东西。"""

    def __init__(self, path: str | Path, *, readonly: bool = False) -> None:
        self.path = Path(path)
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if readonly:
                uri = f"file:{self.path.as_posix()}?mode=ro"
                self.conn = sqlite3.connect(uri, uri=True, isolation_level=None)
            else:
                self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        except sqlite3.Error as exc:
            raise StoreError(f"打开数据库失败：{self.path}：{exc}") from exc
        self.conn.row_factory = sqlite3.Row
        self._closed = False
        if not readonly:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.init_schema()

    # -- 生命周期 -----------------------------------------------------------
    def init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        try:
            self.conn.executescript(sql)
        except sqlite3.Error as exc:
            raise StoreError(f"建表失败：{exc}") from exc

    def close(self) -> None:
        if not self._closed:
            self.conn.close()
            self._closed = True

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 事务 ---------------------------------------------------------------
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """显式事务。失败整体回滚——半条 episode 比没有更难查。"""
        self.conn.execute("BEGIN")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -- 查询 ---------------------------------------------------------------
    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def count(self, table: str, where: str = "", params: tuple[Any, ...] = ()) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        row = self.one(sql, params)
        return int(row["n"]) if row else 0

    def set_meta(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        return str(row["value"]) if row else default
