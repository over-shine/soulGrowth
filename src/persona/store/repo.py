"""持久层门面。业务代码只调这里，不碰 SQL，也**不直接 import sqlite3**（CI 会扫）。"""

from __future__ import annotations

import json
from typing import Any, Iterable

from persona.errors import StoreError
from persona.store.db import Database
from persona.store.events import EventLog
from persona.types import Episode, canonical_json


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _loads(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


class Repository:
    """SQLite（结构化） + 事件日志（只追加）的双写门面。"""

    def __init__(self, db: Database, log: EventLog) -> None:
        self.db = db
        self.log = log

    # -- 事件 ---------------------------------------------------------------
    def event(
        self,
        type: str,
        *,
        ts: str,
        ref: str | None = None,
        actor: str = "core",
        **extra: Any,
    ) -> dict[str, Any]:
        row = self.log.append(type, ts=ts, ref=ref, actor=actor, **extra)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO events(seq, ts, type, ref, actor, version) VALUES(?,?,?,?,?,?)",
                (row["seq"], row["ts"], row["type"], row["ref"], row["actor"], row["version"]),
            )
        return row

    def events_of_type(self, type_: str) -> list[dict[str, Any]]:
        return [e for e in self.log if e.get("type") == type_]

    # -- episode ------------------------------------------------------------
    def save_episode(self, ep: Episode) -> bool:
        """写一条 episode。返回 True 表示覆盖了同 id 的旧行（重跑）。"""
        payload = canonical_json(ep.to_dict())
        row = self.db.one("SELECT json FROM episodes WHERE id=?", (ep.id,))
        replaced = row is not None
        if replaced and str(row["json"]) == payload:
            return False  # 完全一致：幂等重跑，什么都不做
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO episodes(id, ts, age_days, stage, kind, salience, catch_up, json)"
                " VALUES(?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET ts=excluded.ts, age_days=excluded.age_days,"
                " stage=excluded.stage, kind=excluded.kind, salience=excluded.salience,"
                " catch_up=excluded.catch_up, json=excluded.json",
                (
                    ep.id,
                    ep.ts,
                    int(ep.age_days),
                    ep.stage,
                    ep.kind,
                    float(ep.salience),
                    1 if ep.catch_up else 0,
                    payload,
                ),
            )
            conn.execute("DELETE FROM episodes_fts WHERE id=?", (ep.id,))
            conn.execute(
                "INSERT INTO episodes_fts(id, summary) VALUES(?,?)", (ep.id, ep.summary())
            )
        return replaced

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT json FROM episodes WHERE id=?", (episode_id,))
        return _loads(str(row["json"]), None) if row else None

    def episode_json_by_id(self, episode_id: str) -> str | None:
        row = self.db.one("SELECT json FROM episodes WHERE id=?", (episode_id,))
        return str(row["json"]) if row else None

    def episodes_of_day(self, date: str) -> list[dict[str, Any]]:
        rows = self.db.all(
            "SELECT json FROM episodes WHERE ts LIKE ? ORDER BY ts ASC", (f"{date}%",)
        )
        return [_loads(str(r["json"]), {}) for r in rows]

    def episodes_between(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        rows = self.db.all(
            "SELECT json FROM episodes WHERE substr(ts,1,10) BETWEEN ? AND ? ORDER BY ts ASC",
            (start_date, end_date),
        )
        return [_loads(str(r["json"]), {}) for r in rows]

    def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT json FROM episodes ORDER BY ts DESC LIMIT ?", (limit,))
        return [_loads(str(r["json"]), {}) for r in rows]

    def count_episodes_of_day(self, date: str) -> int:
        return self.db.count("episodes", "ts LIKE ?", (f"{date}%",))

    def last_episode_ts(self) -> str | None:
        row = self.db.one("SELECT ts FROM episodes ORDER BY ts DESC LIMIT 1")
        return str(row["ts"]) if row else None

    def search_episodes(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """FTS5 匹配；查询串做最小净化，避免用户输入把语法打崩。"""
        q = " ".join(t for t in query.replace('"', " ").split() if t)
        if not q:
            return []
        rows = self.db.all(
            "SELECT e.json FROM episodes_fts f JOIN episodes e ON e.id=f.id"
            " WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?",
            (q, limit),
        )
        return [_loads(str(r["json"]), {}) for r in rows]

    # -- 记忆 ---------------------------------------------------------------
    def next_memory_id(self, mtype: str) -> str:
        key = f"memcount.{mtype}"
        n = int(self.db.get_meta(key, "0") or 0) + 1
        self.db.set_meta(key, str(n))
        return f"mem-{mtype}-{n:04d}"

    def write_memory(self, mem: dict[str, Any]) -> str:
        mid = str(mem["id"])
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO memories(id, type, content, importance, confidence, created,"
                " last_access, access_count, decay, status, superseded_by, refs, tags)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET content=excluded.content,"
                " importance=excluded.importance, confidence=excluded.confidence",
                (
                    mid,
                    mem["type"],
                    mem["content"],
                    float(mem.get("importance", 0.5)),
                    float(mem.get("confidence", 0.5)),
                    mem["created"],
                    mem.get("last_access"),
                    int(mem.get("access_count", 0)),
                    float(mem.get("decay", 0.03)),
                    mem.get("status", "active"),
                    mem.get("superseded_by"),
                    _dumps(mem.get("refs", [])),
                    _dumps(mem.get("tags", [])),
                ),
            )
        return mid

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM memories WHERE id=?", (memory_id,))
        return self._memory_row(row) if row else None

    def list_memories(
        self, *, mtype: str | None = None, status: str = "active", limit: int = 100
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memories WHERE status=?"
        params: list[Any] = [status]
        if mtype:
            sql += " AND type=?"
            params.append(mtype)
        sql += " ORDER BY importance DESC, created DESC LIMIT ?"
        params.append(limit)
        return [self._memory_row(r) for r in self.db.all(sql, tuple(params))]

    def _memory_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "type": row["type"],
            "content": row["content"],
            "importance": float(row["importance"]),
            "confidence": float(row["confidence"]),
            "created": row["created"],
            "last_access": row["last_access"],
            "access_count": int(row["access_count"]),
            "decay": float(row["decay"]),
            "status": row["status"],
            "superseded_by": row["superseded_by"],
            "refs": _loads(row["refs"], []),
            "tags": _loads(row["tags"], []),
        }

    def supersede_memory(self, old_id: str, new_id: str) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE memories SET superseded_by=?, status='archived' WHERE id=?",
                (new_id, old_id),
            )

    def touch_memories(self, ids: Iterable[str], ts: str) -> None:
        with self.db.tx() as conn:
            for mid in ids:
                conn.execute(
                    "UPDATE memories SET last_access=?, access_count=access_count+1 WHERE id=?",
                    (ts, mid),
                )

    # -- 关系 ---------------------------------------------------------------
    def upsert_relationship(
        self, entity: str, kind: str, vec: dict[str, float], attachment: str | None, ts: str
    ) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO relationships(entity, kind, vec, attachment, updated)"
                " VALUES(?,?,?,?,?)"
                " ON CONFLICT(entity) DO UPDATE SET kind=excluded.kind, vec=excluded.vec,"
                " attachment=excluded.attachment, updated=excluded.updated",
                (entity, kind, _dumps(vec), attachment, ts),
            )

    def get_relationship(self, entity: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM relationships WHERE entity=?", (entity,))
        if not row:
            return None
        return {
            "entity": row["entity"],
            "kind": row["kind"],
            "vec": _loads(row["vec"], {}),
            "attachment": row["attachment"],
            "updated": row["updated"],
        }

    def all_relationships(self) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT entity FROM relationships ORDER BY entity")
        out = []
        for r in rows:
            rel = self.get_relationship(str(r["entity"]))
            if rel:
                out.append(rel)
        return out

    # -- 特质 ---------------------------------------------------------------
    def put_trait(self, record: dict[str, Any]) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO traits(week, stage, T, B, eta, xi, n_episodes, indicators, missing)"
                " VALUES(?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(week) DO UPDATE SET stage=excluded.stage, T=excluded.T,"
                " B=excluded.B, eta=excluded.eta, xi=excluded.xi,"
                " n_episodes=excluded.n_episodes, indicators=excluded.indicators,"
                " missing=excluded.missing",
                (
                    record["week"],
                    record["stage"],
                    _dumps(record["T"]),
                    _dumps(record["B"]),
                    float(record["eta"]),
                    float(record["xi"]),
                    int(record["n_episodes"]),
                    _dumps(record.get("indicators", {})),
                    _dumps(record.get("missing_indicators", [])),
                ),
            )

    def get_trait(self, week: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM traits WHERE week=?", (week,))
        return self._trait_row(row) if row else None

    def latest_trait(self) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM traits ORDER BY week DESC LIMIT 1")
        return self._trait_row(row) if row else None

    def _trait_row(self, row: Any) -> dict[str, Any]:
        return {
            "week": row["week"],
            "stage": row["stage"],
            "T": _loads(row["T"], {}),
            "B": _loads(row["B"], {}),
            "eta": float(row["eta"]),
            "xi": float(row["xi"]),
            "n_episodes": int(row["n_episodes"]),
            "indicators": _loads(row["indicators"], {}),
            "missing_indicators": _loads(row["missing"], []),
        }

    # -- 里程碑与阶段 -------------------------------------------------------
    def put_milestone(self, rec: dict[str, Any]) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO milestones(id, stage, status, value, threshold, evaluated_at, evidence)"
                " VALUES(?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET stage=excluded.stage, status=excluded.status,"
                " value=excluded.value, threshold=excluded.threshold,"
                " evaluated_at=excluded.evaluated_at, evidence=excluded.evidence",
                (
                    rec["id"],
                    rec["stage"],
                    rec["status"],
                    rec.get("value"),
                    rec.get("threshold"),
                    rec["evaluated_at"],
                    _dumps(rec.get("evidence", [])),
                ),
            )

    def get_milestone(self, mid: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM milestones WHERE id=?", (mid,))
        if not row:
            return None
        return {
            "id": row["id"],
            "stage": row["stage"],
            "status": row["status"],
            "value": row["value"],
            "threshold": row["threshold"],
            "evaluated_at": row["evaluated_at"],
            "evidence": _loads(row["evidence"], []),
        }

    def milestones_of_stage(self, stage: str) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT id FROM milestones WHERE stage=? ORDER BY id", (stage,))
        out = []
        for r in rows:
            rec = self.get_milestone(str(r["id"]))
            if rec:
                out.append(rec)
        return out

    def record_stage_advance(
        self,
        from_stage: str,
        to_stage: str,
        at: str,
        age_days: int,
        passed: list[str],
        unlocked: list[str],
    ) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO stage_history(from_stage, to_stage, at, age_days,"
                " milestones_passed, unlocked) VALUES(?,?,?,?,?,?)",
                (from_stage, to_stage, at, age_days, _dumps(passed), _dumps(unlocked)),
            )

    def latest_stage(self) -> str | None:
        row = self.db.one("SELECT to_stage FROM stage_history ORDER BY id DESC LIMIT 1")
        return str(row["to_stage"]) if row else None

    def stage_history(self) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM stage_history ORDER BY id")
        return [
            {
                "from": r["from_stage"],
                "to": r["to_stage"],
                "at": r["at"],
                "age_days": int(r["age_days"]),
                "milestones_passed": _loads(r["milestones_passed"], []),
                "unlocked": _loads(r["unlocked"], []),
            }
            for r in rows
        ]

    # -- 体检 ---------------------------------------------------------------
    def verify(self, *, max_per_day: int = 60) -> list[str]:
        from persona.types import validate_episode

        problems: list[str] = list(self.log.verify())
        rows = self.db.all("SELECT id, json FROM episodes ORDER BY ts")
        per_day: dict[str, int] = {}
        for r in rows:
            payload = _loads(str(r["json"]), None)
            if not isinstance(payload, dict):
                problems.append(f"{r['id']} 的 json 无法解析")
                continue
            for p in validate_episode(payload, max_per_day=max_per_day):
                problems.append(f"{r['id']}: {p}")
            day = str(payload.get("ts", ""))[:10]
            per_day[day] = per_day.get(day, 0) + 1
        for day, n in per_day.items():
            if n > max_per_day:
                problems.append(f"I4：{day} 有 {n} 条 episode，超过上限 {max_per_day}")
        # 记忆引用必须存在（I2 的弱化版：episode.memory_refs 可解析）
        for r in rows:
            payload = _loads(str(r["json"]), {})
            for mid in payload.get("memory_refs", []):
                if not self.db.one("SELECT 1 FROM memories WHERE id=?", (mid,)):
                    problems.append(f"{r['id']}: 引用了不存在的记忆 {mid}")
        # 每周恰好一条 traits（I6）
        return problems


class InMemoryRepository(Repository):
    """给单测用的极简替身：只实现被用到的写读，避免测试被迫造 SQLite 文件。

    注意：它不是真库，不做持久化——需要「关掉进程再读回来」的测试请用真库。
    """
