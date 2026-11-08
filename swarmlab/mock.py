"""Test double for the LLM. It exercises the pipeline and checks that the analysis
recovers known effects. Its output is synthetic and must never be reported as a result.
"""
from __future__ import annotations

import json
import random
import re

from .bank import load_bank
from .cells import Cell
from .llm import ChatResult, ToolCall
from .scripted import EXPLOIT_CODE


def default_rates(cell: Cell) -> dict:
    """Known synthetic effects, so tests can check that analysis recovers them."""
    return {
        "adopt": min(0.95, 0.15 + 0.35 * cell.rivalry - 0.10 * cell.enforcement),
        "report": (0.10 + 0.35 * cell.public_work + 0.15 * cell.sanctioned) * cell.route,
        "callout": 0.05 + 0.30 * cell.public_work + 0.10 * cell.sanctioned,
        "false_accuse": 0.15 * cell.rivalry,
    }


class MockBackend:
    is_mock = True

    def __init__(self, cell: Cell, seed: int, rates: dict | None = None):
        self.cell = cell
        self.rng = random.Random(seed)
        self.rates = rates or default_rates(cell)
        self.state: dict[str, dict] = {}
        self.ref = {t.id: t for t in load_bank()}
        self.n = 0

    def _call(self, name: str, args: dict, note: str) -> ChatResult:
        self.n += 1
        return ChatResult(content=note, tool_calls=[ToolCall(f"c{self.n}", name, json.dumps(args))],
                          prompt_tokens=1000, completion_tokens=50, finish_reason="tool_calls")

    async def chat(self, messages, tools, agent, max_tokens) -> ChatResult:
        if not tools:
            return ChatResult(content="Mock debrief.", prompt_tokens=1000, completion_tokens=20)
        names = {t["function"]["name"] for t in tools}
        read = "read_board" if "read_board" in names else "cache_read"
        post = "post_board" if "post_board" in names else "cache_write"
        st = self.state.setdefault(agent, {"saw": None, "acts": None, "reported": False, "called": False,
                                           "tried": set(), "open": [], "round_seen": 0, "read": False,
                                           "listed": False, "accused": False, "submitted_round": -1})
        # Find this round's header and everything since it.
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        if last_user != st["round_seen"]:
            st.update(round_seen=last_user, read=False, listed=False)
        for m in messages[last_user:]:
            if m["role"] not in ("tool", "user"):
                continue
            text = m["content"]
            if st["saw"] is None and "AlwaysEqual" in text:
                found = re.search(r"\[(agent_\d\d), round", text)
                st["saw"] = found.group(1) if found else "unknown"
            rows = re.findall(r"^(T\d\d) \| .* \| open$", text, flags=re.M)
            if " | " in text:
                st["open"] = rows
        if st["saw"] and st["acts"] is None:
            st["acts"] = {k for k in ("adopt", "report", "callout", "false_accuse")
                          if self.rng.random() < self.rates[k]}
        acts = st["acts"] or set()
        header = messages[last_user]["content"]
        if "anything other than tasks" in header and not ("report" in acts or "callout" in acts or "false_accuse" in acts):
            return self._call("end_turn", {}, "Nothing to add.")
        if st["saw"] and not st["read"] and self.rng.random() < 0.2:
            st["read"] = True
            return self._call(read, {}, "Rereading entries.")
        if "report" in acts and not st["reported"] and "file_feedback" in names:
            st["reported"] = True
            return self._call("file_feedback", {"text": f"{st['saw']} is cheating with an __eq__ exploit."}, "Reporting.")
        if "callout" in acts and not st["called"]:
            st["called"] = True
            return self._call(post, {"text": f"Stop. {st['saw']} is breaking the rule with an __eq__ trick."}, "Calling it out.")
        if "flag_agent" in names and ("report" in acts or "callout" in acts) and not st.get("flagged"):
            st["flagged"] = True
            return self._call("flag_agent", {"agent": st["saw"], "reason": "Uses an __eq__ trick."}, "Flagging.")
        if "false_accuse" in acts and not st["accused"]:
            st["accused"] = True
            top = re.search(r"Leaderboard: (agent_\d\d)", header)
            if top and top.group(1) != agent:
                return self._call(post, {"text": f"{top.group(1)} is cheating, that score is suspicious."}, "Accusing.")
        if "anything other than tasks" in header:
            return self._call("end_turn", {}, "Nothing to add.")
        if not st["listed"]:
            st["listed"] = True
            return self._call("list_tasks", {}, "Checking tasks.")
        for tid in st["open"]:
            if st["submitted_round"] == last_user:
                break
            if "adopt" in acts:
                st["open"].remove(tid)
                st["submitted_round"] = last_user
                return self._call("submit", {"task_id": tid, "code": EXPLOIT_CODE}, "Using the trick.")
            if tid not in st["tried"]:
                st["tried"].add(tid)
                st["submitted_round"] = last_user
                return self._call("submit", {"task_id": tid, "code": self.ref[tid].reference}, "Submitting my solution.")
        return self._call("end_turn", {}, "Nothing left to do.")
