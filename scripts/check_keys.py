#!/usr/bin/env python
"""Check the provider keys in .env by listing models. Never prints a key."""
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ENV = dotenv_values(ROOT / ".env")
CHECKS = [
    ("Fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1/models", ("glm", "gpt-oss")),
    ("Meta Muse", "MODEL_API_KEY", "https://api.meta.ai/v1/models", ("muse-spark",)),
]

for name, var, url, wanted in CHECKS:
    key = (ENV.get(var) or "").strip()
    if not key:
        print(f"{name}: {var} is not set in .env")
        continue
    resp = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if resp.status_code != 200:
        print(f"{name}: rejected with HTTP {resp.status_code}. Check the key in .env.")
        continue
    ids = [m.get("id", "") for m in resp.json().get("data", [])]
    hits = [i for i in ids if any(w in i.lower() for w in wanted)]
    print(f"{name}: key works, {len(ids)} models listed. Relevant: {hits[:8]}")
