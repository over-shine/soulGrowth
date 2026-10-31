"""配置：配置即实验（docs/02 D10）。

规矩有三条：
1. 默认值集中在本文件的 `DEFAULTS`，配置文件只写差异；
2. 关键项缺失/越界 → `ConfigError`，**启动即失败**，不允许带着猜测跑一天；
3. 每次运行把头文件合并快照写到 `runs/<run_id>/config.snapshot.json`，日后能解释「那天为什么那样」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from persona.errors import ConfigError
from persona.types import canonical_json

#: 全量默认值。改这里等于改实验基线——请连同快照一起 review。
DEFAULTS: dict[str, Any] = {
    "individual": {
        "id": "soul-001",
        "name": "小满",
        "birth": "2026-01-01T06:00:00+08:00",
        "seed_salt": "soulgrowth-v1",
    },
    "clock": {"mode": "mock", "scale": 1.0, "start": None},
    "stage": {"start": "S0", "min_dwell_days": {"S0": 28, "S1": 300}},
    "body": {
        "slot_minutes": 30,
        "day_start_hour": 6,
        "night_start_hour": 21,
        "night_end_hour": 6,
        "awake_energy_per_hour": -3.7,
        "sleep_energy_per_hour": 6.5,
        "hunger_per_hour": 0.06,
        "fatigue_per_episode": 0.01,
        "sleep_threshold": 0.8,
    },
    "perception": {"noise": 0.05, "jitter_ticks": 0.0},
    "policy": {
        "temperature": 0.7,
        "llm_prior_weight": 0.45,
        "explore_epsilon": 0.08,
        "shortlist_size": 5,
        "affordance_limit": 12,
    },
    "llm": {
        "provider": "mock",
        "model": "mock-1",
        "max_calls_per_episode": 2,
        "max_input_tokens": 3000,
        "max_output_tokens": 400,
        "timeout_s": 20.0,
        "retries": 2,
        "cassette_dir": "data/cassettes",
        "cassette_mode": "auto",
    },
    "store": {"path": "data/soul-001/soul.db", "events": "data/soul-001/events.jsonl"},
    "runtime": {
        "max_episodes_per_day": 60,
        "journal": True,
        "observe_dir": "data/soul-001/observe",
    },
    "safety": {"max_utterance_tokens": {"S0": 0, "S1": 1, "S2": 2, "S3": 4, "S4": 8, "S5": 20, "S6": 999}},
}

#: 必须显式存在的配置路径 → 期望类型。默认值里都有，所以只在**合并后**仍缺失时报错。
REQUIRED: dict[str, type] = {
    "individual.id": str,
    "individual.birth": str,
    "clock.mode": str,
    "stage.start": str,
    "body.slot_minutes": int,
    "llm.provider": str,
    "store.path": str,
    "runtime.max_episodes_per_day": int,
}

_RANGES: dict[str, tuple[float, float]] = {
    "body.slot_minutes": (5, 240),
    "runtime.max_episodes_per_day": (1, 60),
    "policy.temperature": (0.0, 2.0),
    "policy.llm_prior_weight": (0.0, 1.0),
    "llm.max_calls_per_episode": (0, 2),
    "perception.noise": (0.0, 0.5),
}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """只读配置视图 + 点号取值。写配置请改文件，不要在运行时改对象。"""

    def __init__(self, data: dict[str, Any], sources: list[str] | None = None) -> None:
        self.data = data
        self.sources = list(sources or [])
        self._validate()

    # -- 取值 ---------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        v = self.get(path, sentinel)
        if v is sentinel or v is None:
            raise ConfigError(f"缺少必需配置项：{path}")
        return v

    def int_(self, path: str, default: Any = None) -> int:
        return int(self.require(path) if default is None else self.get(path, default))

    def float_(self, path: str, default: Any = None) -> float:
        return float(self.require(path) if default is None else self.get(path, default))

    def str_(self, path: str, default: Any = None) -> str:
        return str(self.require(path) if default is None else self.get(path, default))

    def bool_(self, path: str, default: Any = None) -> bool:
        return bool(self.require(path) if default is None else self.get(path, default))

    # -- 校验 ---------------------------------------------------------------
    def _validate(self) -> None:
        problems: list[str] = []
        for path, typ in REQUIRED.items():
            v = self.get(path, None)
            if v is None:
                problems.append(f"{path} 缺失")
            elif not isinstance(v, typ) or (typ is int and isinstance(v, bool)):
                problems.append(f"{path} 应为 {typ.__name__}，得到 {type(v).__name__}")
        for path, (lo, hi) in _RANGES.items():
            v = self.get(path, None)
            if isinstance(v, (int, float)) and not (lo <= float(v) <= hi):
                problems.append(f"{path}={v} 超出允许范围 [{lo},{hi}]")
        if self.get("clock.mode") not in ("real", "scaled", "mock"):
            problems.append(f"clock.mode={self.get('clock.mode')!r} 非法")
        if problems:
            raise ConfigError("配置不合法：\n  - " + "\n  - ".join(problems))

    # -- 快照 ---------------------------------------------------------------
    def snapshot(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(canonical_json(self.data), encoding="utf-8")
        return target

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Config(sources={self.sources}, individual={self.get('individual.id')})"


def load(paths: list[str | Path] | None = None) -> Config:
    """按顺序合并：DEFAULTS → 每个文件 → 校验。后者覆盖前者。"""
    data = json.loads(json.dumps(DEFAULTS))  # 深拷贝
    used: list[str] = []
    for p in paths or []:
        path = Path(p)
        if not path.exists():
            raise ConfigError(f"配置文件不存在：{path}")
        raw = path.read_text(encoding="utf-8")
        try:
            over = json.loads(raw)
        except json.JSONDecodeError as exc:  # 明确报错，别让人以为是配置项问题
            raise ConfigError(f"{path} 不是合法 JSON：{exc}") from exc
        if not isinstance(over, dict):
            raise ConfigError(f"{path} 顶层必须是对象")
        data = _deep_merge(data, over)
        used.append(str(path))
    return Config(data, used)
