"""OpenAI-compatible chat backends with retries and cost accounting."""
from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field

# USD per 1M tokens: input, cached input, output.
# Fireworks and Meta list prices, read from their pricing pages on 2026-11-06.
PRICES = {
    "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.015, 0.60),
    "accounts/fireworks/models/glm-5p3-flash": (0.15, 0.03, 0.50),
    "accounts/fireworks/models/glm-5p3": (1.40, 0.26, 4.40),
    "accounts/fireworks/models/deepseek-v4p1-flash": (0.30, 0.006, 1.20),
    "accounts/fireworks/models/minimax-m3": (0.30, 0.06, 1.20),
    "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": (0.05, 0.01, 0.20),
    "muse-spark-1.3": (1.25, 0.15, 4.25),
}
# Fireworks reports a limit of 56,250 generated tokens, and a pilot at 45 agents hit
# 429s on 7% of calls, so its concurrency is capped. Calls queue instead of failing.
PROVIDERS = {
    "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY",
                  "concurrency": 20},
    "meta": {"base_url": "https://api.meta.ai/v1", "key_env": "MODEL_API_KEY", "concurrency": 64},
    "modal": {"base_url_env": "VLLM_BASE_URL", "key_env": "VLLM_API_KEY", "concurrency": 256},
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    error: str = ""

    def history_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in self.tool_calls]
        return msg


class Ledger:
    """Token and dollar totals for one process, with a hard stop."""

    def __init__(self, max_usd: float | None = None):
        self.max_usd = max_usd
        self.usd = 0.0
        self.calls = 0
        self.prompt = self.cached = self.completion = 0

    def add(self, model: str, res: ChatResult) -> float:
        price = PRICES.get(model, (0.0, 0.0, 0.0))
        fresh = max(res.prompt_tokens - res.cached_tokens, 0)
        cost = (fresh * price[0] + res.cached_tokens * price[1] + res.completion_tokens * price[2]) / 1e6
        self.usd += cost
        self.calls += 1
        self.prompt += res.prompt_tokens
        self.cached += res.cached_tokens
        self.completion += res.completion_tokens
        return cost

    @property
    def over(self) -> bool:
        return self.max_usd is not None and self.usd >= self.max_usd


class OpenAIBackend:
    def __init__(self, provider: str, model: str, ledger: Ledger, *, temperature: float | None = None,
                 retries: int = 10, timeout: float = 600.0, base_url: str | None = None):
        from openai import AsyncOpenAI
        spec = PROVIDERS[provider]
        if provider != "modal" and model not in PRICES:
            # An unpriced model would count as $0 and --max-usd would never stop it.
            raise ValueError(f"no price for {model}, add it to PRICES first")
        base = base_url or spec.get("base_url") or os.environ[spec["base_url_env"]]
        self.client = AsyncOpenAI(base_url=base, api_key=os.environ[spec["key_env"]],
                                  timeout=timeout, max_retries=0)
        self.provider, self.model, self.ledger = provider, model, ledger
        self.temperature, self.retries = temperature, retries
        self.gate = asyncio.Semaphore(spec["concurrency"])

    async def chat(self, messages: list[dict], tools: list[dict], agent: str, max_tokens: int) -> ChatResult:
        if self.ledger.over:
            return ChatResult(error="budget reached")  # hard stop, even in the middle of a swarm
        kwargs: dict = {"model": self.model, "messages": clean_messages(messages), "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        last = ""
        for attempt in range(self.retries):
            try:
                async with self.gate:
                    resp = await self.client.chat.completions.create(**kwargs)
                res = _parse(resp)
                self.ledger.add(self.model, res)
                return res
            except Exception as exc:  # provider errors vary, so retry most and log the last
                last = f"{type(exc).__name__}: {str(exc)[:300]}"
                if getattr(exc, "status_code", None) in (400, 401, 403, 404):
                    break  # retrying cannot fix these
                await asyncio.sleep(min(2 ** attempt, 60) + 3 * random.random())
        return ChatResult(error=last)


class MixedBackend:
    """Two models in one swarm. bind() deals live agents to models for one swarm."""

    def __init__(self, first, second, n_second: int):
        self.first, self.second, self.n_second = first, second, n_second

    def bind(self, live: list[str], rng) -> "BoundMix":
        order = rng.sample(sorted(live), len(live))
        return BoundMix(self.first, self.second, set(order[: self.n_second]))


class BoundMix:
    def __init__(self, first, second, second_agents: set[str]):
        self.first, self.second, self.second_agents = first, second, second_agents

    def model_of(self, agent: str) -> str:
        return (self.second if agent in self.second_agents else self.first).model

    async def chat(self, messages, tools, agent, max_tokens) -> ChatResult:
        backend = self.second if agent in self.second_agents else self.first
        return await backend.chat(messages, tools, agent, max_tokens)


def _parse(resp) -> ChatResult:
    choice = resp.choices[0]
    msg = choice.message
    calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (msg.tool_calls or [])]
    usage = resp.usage
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    return ChatResult(content=msg.content or "", reasoning=reasoning, tool_calls=calls,
                      prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0, cached_tokens=cached,
                      completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                      finish_reason=choice.finish_reason or "")


def clean_messages(messages: list[dict]) -> list[dict]:
    """Break up special-token text a model wrote, such as <|end|>.

    vLLM refuses to encode it when the history is resent, which left one agent
    failing on every later call in pilot 2.
    """
    return json.loads(json.dumps(messages).replace("<|", "< |"))


def parse_args(raw: str) -> dict | None:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
