"""LLM 边界。

规矩（docs/16 边界）：**整个项目只有这里可以出网**。
其余模块只依赖 `LLMProvider` 这个协议，因此：
* 没有 key 也能跑（`MockProvider`）；
* 测试可复现（`CassetteProvider` 回放）；
* `llm.degraded=true` 时一定带原因（I8）。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from persona.cognition.policy import Prior
from persona.errors import ConfigError

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class LLMProvider:
    """先验提供者的最小接口。实现者必须自己记账 token。"""

    name: str = "base"
    model: str = "none"

    def prior(
        self, *, observation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
    ) -> Prior:  # pragma: no cover - 接口
        raise NotImplementedError

    def close(self) -> None:
        return None


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI 兼容的 chat/completions（DeepSeek、vLLM、Ollama 都可）。"""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 20.0,
        max_output_tokens: int = 400,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("PERSONA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("PERSONA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_output_tokens = int(max_output_tokens)

    def prior(self, *, observation: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Prior:
        from persona.llm import prompts

        if not self.api_key:
            return Prior(degraded=True, degraded_reason="没有 API key", model=self.model)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompts.PRIOR_SYSTEM},
                {
                    "role": "user",
                    "content": prompts.build_prior_user(observation=observation, candidates=candidates),
                },
            ],
            "temperature": 0.2,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec - 唯一出网点
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return Prior(degraded=True, degraded_reason=f"请求失败：{exc}", model=self.model)

        usage = payload.get("usage") or {}
        text = ""
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return Prior(
                degraded=True,
                degraded_reason="响应缺少 choices[0].message.content",
                model=self.model,
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
            )
        try:
            data = json.loads(text)
            probs = {str(k): float(v) for k, v in dict(data.get("priors", {})).items()}
        except (ValueError, TypeError, AttributeError) as exc:
            return Prior(
                degraded=True,
                degraded_reason=f"先验 JSON 不合法：{exc}",
                model=self.model,
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
            )
        return Prior(
            probs=probs,
            model=self.model,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
        )


def make_provider(cfg: Any, *, cassette_key: str | None = None) -> LLMProvider:
    """根据配置造一个 provider。未知 provider 名直接启动失败（D10）。"""
    name = str(cfg.get("llm.provider", "mock")).lower()
    model = str(cfg.get("llm.model", "mock-1"))
    if name in ("mock", "none", "offline"):
        from persona.llm.mock import MockProvider

        inner: LLMProvider = MockProvider(
            model=model,
            seed=str(cfg.get("individual.id", "soul")),
        )
    elif name in ("openai", "deepseek", "openai-compatible"):
        inner = OpenAICompatibleProvider(
            model=model,
            api_key=cfg.get("llm.api_key"),
            base_url=cfg.get("llm.base_url"),
            timeout_s=float(cfg.get("llm.timeout_s", 20.0)),
            max_output_tokens=int(cfg.get("llm.max_output_tokens", 400)),
        )
    else:
        raise ConfigError(f"未知 llm.provider：{name}")

    mode = str(cfg.get("llm.cassette_mode", "auto")).lower()
    if mode in ("record", "replay") and cassette_key:
        from persona.llm.cassette import CassetteProvider

        return CassetteProvider(
            inner,
            directory=str(cfg.get("llm.cassette_dir", "data/cassettes")),
            key=cassette_key,
            mode=mode,
        )
    return inner
