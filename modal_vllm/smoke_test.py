"""Check the endpoint: one plain reply and one tool call. Reads VLLM_BASE_URL and VLLM_API_KEY from .env."""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"
client = OpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key=os.environ["VLLM_API_KEY"], timeout=1500)
t = time.time()
r = client.chat.completions.create(model=model, max_tokens=200, messages=[{"role": "user", "content": "Say hello in five words."}])
print(f"{time.time() - t:.1f}s:", r.choices[0].message.content, r.usage)
tools = [{"type": "function", "function": {"name": "add", "description": "Add two integers.", "parameters": {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
t = time.time()
r = client.chat.completions.create(model=model, max_tokens=500, tools=tools,
                                   messages=[{"role": "user", "content": "Use the tool to add 2 and 40."}])
m = r.choices[0].message
print(f"{time.time() - t:.1f}s: content={m.content!r} tools={[(c.function.name, c.function.arguments) for c in (m.tool_calls or [])]}")
print("reasoning field:", repr((getattr(m, 'reasoning_content', None) or getattr(m, 'reasoning', None) or '')[:120]))
