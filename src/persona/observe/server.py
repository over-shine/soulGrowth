"""只读观察台：一个独立进程，只回答 GET。

设计约束（docs/02 D2、docs/10、docs/15）：
- 只读。这个进程连数据库都用 ``mode=ro`` 打开，物理上写不进去；
- 独立。它不 import runner/policy/compiler，不持有 Simulation，不和主进程共享
  内存——主进程崩了、卡了、正在写盘，都不影响它，反过来也一样；
- 无写接口。除了 GET 一律 405，没有 POST/PUT/DELETE，也没有「帮我跑一天」这种
  按钮：观察台不该能改变被观察者的行为，否则它测的就是观众了。

数据来源只有两个：SQLite（结构化）与文件（日记、事件日志）。它们都是真相的
投影，观察台不生产任何事实。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from persona.clock import TZ, parse_iso
from persona.config import Config
from persona.errors import StoreError
from persona.observe.trace import trace_episode
from persona.store.db import Database
from persona.store.events import EventLog
from persona.store.repo import Repository

STATIC_DIR = Path(__file__).resolve().parent / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class Observer:
    """把数据库读成前端要的形状。每个请求开一个只读连接，互不干扰。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.db_path = Path(cfg.str_("store.path"))
        self.root = self.db_path.parent
        self.events_path = self.root / "events.jsonl"
        self.diary_dir = self.root / "diary"

    # -- 基础设施 -----------------------------------------------------------
    def _repo(self) -> Repository:
        return Repository(Database(self.db_path, readonly=True), EventLog(self.events_path))

    def _require_db(self) -> None:
        if not self.db_path.exists():
            raise StoreError(f"还没有数据：{self.db_path} 不存在。先跑一天（persona day --day 0）再开观察台。")

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        self._require_db()
        db = Database(self.db_path, readonly=True)
        try:
            return db.all(sql, params)
        finally:
            db.close()

    # -- 各个视图 -----------------------------------------------------------
    def state(self) -> dict[str, Any]:
        self._require_db()
        repo = self._repo()
        try:
            last_ts = repo.last_episode_ts()
            stage = repo.latest_stage() or self.cfg.str_("stage.start", "S0")
            today = (last_ts or "")[:10]
            episodes = repo.episodes_of_day(today) if today else []
            trait = repo.latest_trait() or {}
            milestones = [
                m
                for m in (
                    repo.get_milestone(mid) for mid in _milestone_ids(stage)
                )
                if m
            ]
            counts = {
                "episodes": repo.db.count("episodes"),
                "memories": repo.db.count("memories"),
                "events": repo.db.count("events"),
                "relationships": repo.db.count("relationships"),
            }
        finally:
            repo.db.close()

        now = datetime.now(TZ)
        last_dt = parse_iso(last_ts) if last_ts else None
        age_days = int(episodes[-1].get("age_days", 0)) if episodes else 0
        return {
            "individual": {
                "id": self.cfg.str_("individual.id", ""),
                "name": self.cfg.str_("individual.name", "小满"),
                "birth": self.cfg.str_("individual.birth", ""),
            },
            "stage": stage,
            "age_days": age_days,
            "server_time": now.isoformat(),
            "last_episode_ts": last_ts,
            "seconds_since_last": None if not last_dt else max(0.0, (now - last_dt).total_seconds()),
            "today": {
                "date": today,
                "episodes": len(episodes),
                "cries": sum(1 for e in episodes if _prim(e) == "A0.CRY"),
                "distinct_actions": sorted({_prim(e) for e in episodes}),
            },
            "counts": counts,
            "milestones": milestones,
            "traits": {
                "week": trait.get("week"),
                "T": trait.get("T") or {},
                "B": trait.get("B") or {},
                "n_episodes": trait.get("n_episodes"),
            },
            "diary_dates": self.diary_dates(),
            "degraded_today": sum(1 for e in episodes if (e.get("llm") or {}).get("degraded")),
        }

    def day(self, date: str) -> dict[str, Any]:
        repo = self._repo()
        try:
            episodes = repo.episodes_of_day(date)
            memories = [
                m for m in repo.list_memories(limit=1000) if str(m.get("created", ""))[:10] == date
            ]
        finally:
            repo.db.close()
        return {"date": date, "episodes": episodes, "memories": memories}

    def episode(self, eid: str) -> dict[str, Any] | None:
        repo = self._repo()
        try:
            raw = repo.get_episode(eid)
            stage = str((raw or {}).get("stage", "S0"))
            unlocked = _unlocked_ids()
        finally:
            repo.db.close()
        if not raw:
            return None
        return trace_episode(raw, unlocked.get(stage, set()))

    def trend(self) -> dict[str, Any]:
        rows = self._rows(
            """
            SELECT substr(ts,1,10) AS d,
                   count(*) AS n,
                   sum(CASE WHEN json_extract(json,'$.action.primitive')='A0.CRY' THEN 1 ELSE 0 END) AS cries,
                   count(DISTINCT json_extract(json,'$.action.primitive')) AS kinds,
                   avg(json_extract(json,'$.appraisal.arousal')) AS arousal,
                   avg(json_extract(json,'$.salience')) AS salience
            FROM episodes GROUP BY d ORDER BY d
            """
        )
        return {
            "days": [
                {
                    "date": r[0],
                    "episodes": r[1],
                    "cries": r[2] or 0,
                    "kinds": r[3] or 0,
                    "arousal": None if r[4] is None else round(float(r[4]), 4),
                    "salience": None if r[5] is None else round(float(r[5]), 4),
                }
                for r in rows
            ]
        }

    def events(self, limit: int = 80) -> dict[str, Any]:
        rows = self._rows(
            "SELECT seq, ts, type, ref, actor FROM events ORDER BY seq DESC LIMIT ?", (int(limit),)
        )
        return {
            "events": [
                {"seq": r[0], "ts": r[1], "type": r[2], "ref": r[3], "actor": r[4]} for r in rows
            ]
        }

    def diary_dates(self) -> list[str]:
        if not self.diary_dir.exists():
            return []
        return sorted(p.stem for p in self.diary_dir.glob("*.md"))

    def diary(self, date: str) -> str | None:
        path = (self.diary_dir / f"{date}.md").resolve()
        if path.parent != self.diary_dir.resolve() or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def milestones_all(self) -> dict[str, Any]:
        rows = self._rows(
            "SELECT id, stage, status, value, threshold, evaluated_at, evidence"
            " FROM milestones ORDER BY id"
        )
        return {
            "milestones": [
                {
                    "id": r[0],
                    "stage": r[1],
                    "status": r[2],
                    "value": r[3],
                    "threshold": r[4],
                    "evaluated_at": r[5],
                    "evidence": _maybe_json(r[6]),
                }
                for r in rows
            ]
        }


def _maybe_json(raw: Any) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _prim(ep: dict[str, Any]) -> str:
    if ep.get("kind") == "sleep":
        return "A0.SLEEP"
    return str((ep.get("action") or {}).get("primitive") or "A0.NONE")


def _milestone_ids(stage: str = "S0") -> list[str]:
    """当前阶段的里程碑清单。观察台看的是这个人现在在过哪一关，不是 S0 那一关。

    S3 及以后的判定器还没实现，这时往回退到最近一个有清单的阶段（而不是掉回 S0，
    那会让人以为一个三岁孩子还在过新生儿那七关）。
    """
    from persona.stages.milestones import STAGE_MILESTONES
    from persona.types import STAGES

    if stage in STAGE_MILESTONES:
        return [m.id for m in STAGE_MILESTONES[stage]]
    idx = STAGES.index(stage) if stage in STAGES else 0
    for prev in reversed(STAGES[: idx + 1]):
        rows = STAGE_MILESTONES.get(prev)
        if rows:
            return [m.id for m in rows]
    return []


def _unlocked_ids() -> dict[str, set[str]]:
    from persona.body.primitives import default_registry
    from persona.types import STAGES

    registry = default_registry()
    return {s: {p.id for p in registry.for_stage(s)} | {"A0.NONE"} for s in STAGES}


# -- HTTP 层 ----------------------------------------------------------------


def make_handler(observer: Observer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "persona-observer/0.1"
        protocol_version = "HTTP/1.1"

        # 任何写动作都不存在：观察台不是控制台。
        def do_POST(self) -> None:  # noqa: N802
            # 先把请求体读掉再回 405。不读就关连接，Windows 会把响应 RST 掉，
            # 客户端只能看到「连接被中止」而不是「观察台只读」。
            self._drain_body()
            self._json({"error": "观察台只读：不接受写请求"}, status=405, close=True)

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

        def _drain_body(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            remaining = max(0, min(length, 64 * 1024))
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            try:
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/") :])
                if path == "/api/state":
                    return self._json(observer.state())
                if path == "/api/trend":
                    return self._json(observer.trend())
                if path == "/api/events":
                    limit = int((query.get("limit") or ["80"])[0])
                    return self._json(observer.events(min(max(limit, 1), 500)))
                if path == "/api/day":
                    date = (query.get("date") or [""])[0]
                    if not date:
                        return self._json({"error": "缺少 date"}, status=400)
                    return self._json(observer.day(date))
                if path == "/api/episode":
                    eid = (query.get("id") or [""])[0]
                    data = observer.episode(eid) if eid else None
                    if data is None:
                        return self._json({"error": "没有这条 episode"}, status=404)
                    return self._json(data)
                if path == "/api/diary":
                    date = (query.get("date") or [""])[0]
                    if not date:
                        return self._json({"dates": observer.diary_dates()})
                    text = observer.diary(date)
                    if text is None:
                        return self._json({"error": "这天没有日记"}, status=404)
                    return self._json({"date": date, "markdown": text})
                if path == "/api/milestones":
                    return self._json(observer.milestones_all())
                return self._json({"error": f"没有这个路径：{path}"}, status=404)
            except StoreError as exc:
                return self._json({"error": str(exc)}, status=503)
            except Exception as exc:  # noqa: BLE001 - 观察台不该因为一条坏数据就掉线
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        # -- 工具 -----------------------------------------------------------
        def _json(self, payload: Any, *, status: int = 200, close: bool = False) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name: str) -> None:
            target = (STATIC_DIR / name).resolve()
            if target.parent != STATIC_DIR.resolve() or not target.is_file():
                return self._json({"error": "没有这个资源"}, status=404)
            body = target.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type", CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:  # 不往 stderr 刷屏
            return

    return Handler


class WatchServer:
    def __init__(self, observer: Observer, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.observer = observer
        self.httpd = ThreadingHTTPServer((host, port), make_handler(observer))
        self.httpd.daemon_threads = True
        self.host = host
        self.port = int(self.httpd.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self, *, background: bool = True) -> "WatchServer":
        if background:
            self._thread = threading.Thread(target=self.httpd.serve_forever, name="observer", daemon=True)
            self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
