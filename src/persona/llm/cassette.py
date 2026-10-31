"""录像带：把一次真实调用录下来，之后离线回放。

为什么需要它：行为测试要「跨版本可复现」。没有录像带，任何一次 prompt
微调都会让回归测试变成新的随机实验。

`replay` 模式缺记录 → **直接失败**，绝不静默退回 mock（否则测试会假装通过）。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from persona.cognition.policy import Prior
from persona.errors import PersonaError
from persona.llm.provider import LLMProvider


class CassetteMiss(PersonaError):
    """回放时找不到对应记录。"""


def _obs_hash(observation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        {"observation": observation, "candidates": [c.get("primitive") for c in candidates]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


class CassetteProvider(LLMProvider):
    name = "cassette"

    def __init__(self, inner: LLMProvider, *, directory: str, key: str, mode: str = "record") -> None:
        if mode not in ("record", "replay"):
            raise ValueError(f"cassette mode 只能是 record/replay，得到 {mode}")
        self.inner = inner
        self.mode = mode
        self.dir = Path(directory)
        self.key = key
        self.index = 0
        self.model = inner.model
        self._cache: list[dict[str, Any]] | None = None

    @property
    def path(self) -> Path:
        return self.dir / f"{self.key}.jsonl"

    def prior(self, *, observation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Prior:
        idx = self.index
        self.index += 1
        digest = _obs_hash(observation, candidates)
        if self.mode == "replay":
            records = self._records()
            if idx >= len(records):
                raise CassetteMiss(f"{self.path} 缺少第 {idx} 条记录")
            rec = records[idx]
            if rec.get("hash") != digest:
                raise CassetteMiss(
                    f"{self.path} 第 {idx} 条上下文不匹配（期望 {rec.get('hash')}，实际 {digest}）"
                )
            return Prior(**rec["prior"])

        result = self.inner.prior(observation=observation, candidates=candidates)
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(
                json.dumps(
                    {"index": idx, "hash": digest, "prior": _prior_dict(result)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            fh.flush()
            os.fsync(fh.fileno())
        return result

    def _records(self) -> list[dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            raise CassetteMiss(f"没有录像带：{self.path}")
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        self._cache = out
        return out


def _prior_dict(p: Prior) -> dict[str, Any]:
    return {
        "probs": {k: round(v, 8) for k, v in p.probs.items()},
        "model": p.model,
        "tokens_in": p.tokens_in,
        "tokens_out": p.tokens_out,
        "degraded": p.degraded,
        "degraded_reason": p.degraded_reason,
    }
