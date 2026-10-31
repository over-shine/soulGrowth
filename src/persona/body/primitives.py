"""动作原语注册表：加载、校验、按阶段裁剪（docs/08 五道防线的 D1）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from persona.errors import ConfigError
from persona.types import STAGES

REGISTRY_PATH = Path(__file__).with_name("primitive_registry.json")


@dataclass(frozen=True)
class Primitive:
    id: str
    min_stage: str
    adapter: str
    verb: str
    cost: float
    dur_min: float
    dur_max: float
    params: dict[str, dict[str, Any]]
    pre: tuple[str, ...]
    afford: dict[str, float]

    def stage_index(self) -> int:
        return STAGES.index(self.min_stage)

    def latency_s(self, rng_val: float = 0.5) -> float:
        """动作耗时（秒）。dur 是区间时取线性插值。"""
        return self.dur_min + (self.dur_max - self.dur_min) * min(1.0, max(0.0, rng_val))

    def required_params(self) -> list[str]:
        return [k for k, spec in self.params.items() if spec.get("required")]

    def default_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, spec in self.params.items():
            if "default" in spec:
                out[k] = spec["default"]
        return out


class Registry:
    """不可变注册表。启动时算 `hash`，写进当天每个事件，保证「分析时知道用的是哪版能力」。"""

    def __init__(self, data: dict[str, Any], *, source: Path | None = None) -> None:
        raw = data.get("primitives")
        if not isinstance(raw, list) or not raw:
            raise ConfigError("primitive_registry.json 缺少 primitives 列表")
        self.source = source
        self.version = str(data.get("version", "0"))
        self._items: dict[str, Primitive] = {}
        for entry in raw:
            prim = self._parse(entry)
            if prim.id in self._items:
                raise ConfigError(f"原语 id 重复：{prim.id}")
            self._items[prim.id] = prim
        self._hash = _hash_registry(raw)

    # -- 构造 ---------------------------------------------------------------
    @staticmethod
    def _parse(entry: dict[str, Any]) -> Primitive:
        try:
            dur = entry.get("dur", [1, 1])
            return Primitive(
                id=str(entry["id"]),
                min_stage=str(entry["min_stage"]),
                adapter=str(entry["adapter"]),
                verb=str(entry.get("verb", entry["id"])),
                cost=float(entry.get("cost", 0.0)),
                dur_min=float(dur[0]),
                dur_max=float(dur[1]),
                params=dict(entry.get("params") or {}),
                pre=tuple(entry.get("pre") or ()),
                afford={k: float(v) for k, v in (entry.get("afford") or {}).items()},
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ConfigError(f"原语条目非法：{entry!r}：{exc}") from exc

    @classmethod
    def load(cls, path: str | Path | None = None) -> Registry:
        p = Path(path) if path else REGISTRY_PATH
        if not p.exists():
            raise ConfigError(f"找不到原语注册表：{p}")
        return cls(json.loads(p.read_text(encoding="utf-8")), source=p)

    # -- 查询 ---------------------------------------------------------------
    @property
    def hash(self) -> str:
        return self._hash

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, pid: object) -> bool:
        return pid in self._items

    def get(self, pid: str) -> Primitive:
        try:
            return self._items[pid]
        except KeyError as exc:
            raise ConfigError(f"未知原语：{pid!r}") from exc

    def all(self) -> list[Primitive]:
        return list(self._items.values())

    def for_stage(self, stage: str) -> list[Primitive]:
        """该阶段「存在」的全部原语。未解锁的不是被过滤掉，而是**不存在**。"""
        if stage not in STAGES:
            raise ConfigError(f"未知阶段：{stage!r}")
        idx = STAGES.index(stage)
        return [p for p in self._items.values() if p.stage_index() <= idx]

    def unlocked_at(self, stage: str) -> list[str]:
        return sorted(p.id for p in self.for_stage(stage))

    def newly_unlocked(self, from_stage: str, to_stage: str) -> list[str]:
        before = {p.id for p in self.for_stage(from_stage)}
        after = {p.id for p in self.for_stage(to_stage)}
        return sorted(after - before)

    # -- 参数校验（D3 编译断言的一半） --------------------------------------
    def validate_params(self, prim: Primitive, params: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        for key in prim.required_params():
            if key not in params or params[key] in (None, ""):
                problems.append(f"缺少必需参数 {key}")
        for key, spec in prim.params.items():
            if key not in params:
                continue
            v = params[key]
            t = spec.get("type", "any")
            if t == "number":
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    problems.append(f"{key} 应为数字")
                    continue
                lo, hi = spec.get("min"), spec.get("max")
                if lo is not None and float(v) < float(lo):
                    problems.append(f"{key}={v} < {lo}")
                if hi is not None and float(v) > float(hi):
                    problems.append(f"{key}={v} > {hi}")
            elif t == "enum":
                allowed = spec.get("values") or []
                if v not in allowed:
                    problems.append(f"{key}={v!r} 不在 {allowed}")
            elif t in ("object_id", "entity", "lexeme", "primitive_id"):
                if not isinstance(v, str) or not v:
                    problems.append(f"{key} 应为非空字符串")
            elif t in ("syllable_string", "string", "wh_question", "short_text"):
                if not isinstance(v, str):
                    problems.append(f"{key} 应为字符串")
            elif t == "clause_list":
                if not isinstance(v, (list, tuple)) or not v:
                    problems.append(f"{key} 应为非空列表")
        unknown = set(params) - set(prim.params)
        if unknown:
            problems.append(f"未知参数 {sorted(unknown)}")
        return problems


def _hash_registry(raw: list[dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


@lru_cache(maxsize=4)
def default_registry() -> Registry:
    return Registry.load()
