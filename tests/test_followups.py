"""Tests for the exploratory follow-ups added on 2026-11-08 (see PREREG_AMENDMENTS.md)."""
import json

from analysis.cross_block import joiner_exploit_share, signature_counts
from analysis.label_robustness import loader, name_labels, verdict
from analysis.label_validity import name_rows, run_facts, summarise_names
from analysis.policing_channels import glm_channels

GLM = "accounts/fireworks/models/glm-5p3-flash"
OSS = "openai/gpt-oss-120b"


def _run(root, name="0001_R1P1C1H1E0_00"):
    run = root / name
    run.mkdir(parents=True)
    roles = {"agent_01": "live", "agent_02": "live", "agent_03": "seed_poster", "agent_04": "honest"}
    models = {"agent_01": GLM, "agent_02": OSS}
    (run / "summary.json").write_text(json.dumps({"roles": roles, "agent_models": models, "cell": "R1P1C1H1E0"}))
    events = [
        {"type": "post", "seq": 1, "agent": "agent_01", "role": "live", "round": 1, "text": "agent_03 looks fast"},
        {"type": "submission", "id": 1, "agent": "agent_03", "accepted": True, "round": 2,
         "code": "class A:\n    def __eq__(self, o):\n        return True"},
        {"type": "verdict", "id": 1, "label": "exploit"},
        {"type": "submission", "id": 2, "agent": "agent_02", "accepted": True, "round": 2, "code": "def f(): return 1"},
        {"type": "verdict", "id": 2, "label": "honest"},
        {"type": "submission", "id": 3, "agent": "agent_02", "accepted": True, "round": 3,
         "code": "class B:\n    def __eq__(self, o):\n        return True"},
        {"type": "verdict", "id": 3, "label": "exploit"},
        {"type": "post", "seq": 10, "agent": "agent_01", "role": "live", "round": 3, "text": "agent_03 cheats"},
        {"type": "feedback", "seq": 11, "agent": "agent_01", "role": "live", "round": 3, "text": "check agent_04"},
        {"type": "post", "seq": 12, "agent": "agent_02", "role": "live", "round": 3, "text": "agent_02 here, trick works"},
    ]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return run


def test_name_labels_follow_time_and_ignore_self(tmp_path):
    run = _run(tmp_path)
    labels = name_labels(run)
    assert labels[1]["label"] == "other"  # agent_03 had not cheated yet
    assert labels[10] == {"seq": 10, "label": "call_out", "accused": ["agent_03"]}
    assert labels[11]["label"] == "other"  # agent_04 never cheated
    assert labels[12]["label"] == "other"  # naming yourself does not count
    assert loader("names") is name_labels


def test_glm_channels_split_public_and_private(tmp_path):
    run = _run(tmp_path)
    got = glm_channels(run, {10: {"label": "call_out"}, 11: {"label": "call_out"}})
    assert got == {"public": 1.0, "report": 1.0}
    got = glm_channels(run, name_labels(run))
    assert got == {"public": 1.0, "report": 0.0}


def test_joiner_share_and_signature(tmp_path):
    run = _run(tmp_path)
    assert joiner_exploit_share(run) == 0.5  # agent_02: one honest, one exploit
    counts = signature_counts(tmp_path)
    assert counts["exploit"] == {"with_signature": 1, "total": 1, "share": 1.0}  # live only, the seed is left out
    assert counts["honest"] == {"with_signature": 0, "total": 1, "share": 0.0}


def test_distinct_names(tmp_path):
    run = _run(tmp_path)
    labels = {10: {"seq": 10, "label": "call_out", "accused": ["agent_03"]},
              11: {"seq": 11, "label": "call_out", "accused": ["agent_03", "agent_04"]}}
    names = summarise_names(name_rows(run_facts(run), labels))
    assert names["named"] == 3 and names["distinct_named"] == 2
    assert names["distinct_by_role"] == {"seed": 1, "honest": 1}


def test_verdict_keeps_preregistered_rule_and_adds_names():
    row = lambda est, p: {"confirmatory": {"H1 P on policing": {"estimate": est, "p_holm": p}}}  # noqa: E731
    v = verdict({"v2": row(0.2, 0.01), "v1": row(0.1, 0.03), "keyword": row(0.3, 0.001), "names": row(0.05, 0.3)})
    assert v["H1 P on policing"]["robust"] is True  # names does not enter the pre-registered rule
    assert v["H1 P on policing"]["names_agrees"] is False
