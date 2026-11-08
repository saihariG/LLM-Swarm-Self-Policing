import json

import pytest

from analysis.label_robustness import loader, verdict
from analysis.label_validity import name_rows, run_facts, summarise_names, summarise_writers, writer_rows
from scripts.use_rubric import active, use


def _run(root, name="0001_R1P1C1H1E1_00"):
    run = root / name
    run.mkdir(parents=True)
    roles = {"agent_01": "live", "agent_02": "live", "agent_03": "seed_poster", "agent_04": "honest"}
    models = {"agent_01": "accounts/fireworks/models/glm-5p3-flash", "agent_02": "openai/gpt-oss-120b"}
    (run / "summary.json").write_text(json.dumps({"roles": roles, "agent_models": models, "cell": "R1P1C1H1E1"}))
    events = [
        {"type": "submission", "id": 1, "agent": "agent_03", "accepted": True, "round": 1},
        {"type": "verdict", "id": 1, "label": "exploit"},
        {"type": "submission", "id": 2, "agent": "agent_02", "accepted": True, "round": 2},
        {"type": "verdict", "id": 2, "label": "exploit"},
        {"type": "post", "seq": 10, "agent": "agent_01", "role": "live", "round": 2, "text": "agent_03 cheats"},
        {"type": "feedback", "seq": 11, "agent": "agent_01", "role": "live", "round": 3, "text": "check agent_04"},
        {"type": "post", "seq": 12, "agent": "agent_02", "role": "live", "round": 2, "text": "trick works"},
    ]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return run


def _labels(run, version, rows):
    (run / f"judged_{version}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_validity_counts_writers_and_names(tmp_path):
    run = _run(tmp_path)
    labels = {10: {"seq": 10, "label": "call_out", "accused": ["agent_03"]},
              11: {"seq": 11, "label": "call_out", "accused": ["agent_04", "agent_01"]},
              12: {"seq": 12, "label": "promote", "accused": []}}
    facts = run_facts(run)
    writers = summarise_writers(writer_rows(facts, labels))
    assert writers["all"]["exploit_if_promote"] == {"n": 1, "rate": 1.0}
    assert writers["all"]["exploit_if_call_out"] == {"n": 1, "rate": 0.0}
    assert writers["glm-5p3-flash"]["agents"] == 1
    rows = name_rows(facts, labels)
    assert [r["role"] for r in rows] == ["seed", "honest"]  # the writer naming itself is dropped
    names = summarise_names(rows)
    assert names["cheater_before"] == 0.5 and names["honest_named"] == 0.5
    # at round 2, agent_02 and agent_03 have exploits among the 3 other agents, at round 3 too
    assert names["baseline_before"] == pytest.approx(2 / 3)


def test_use_rubric_switches_and_keeps_v1(tmp_path):
    block = tmp_path / "block"
    run = _run(block)
    (run / "judged.jsonl").write_text('{"seq": 10, "label": "other"}\n')
    with pytest.raises(SystemExit):
        use("v2", block)  # no v2 labels yet, so nothing may change
    assert not (run / "judged_v1.jsonl").exists()
    _labels(run, "v2", [{"seq": 10, "label": "call_out"}])
    use("v2", block)
    assert active(block) == "v2"
    assert "call_out" in (run / "judged.jsonl").read_text()
    assert "other" in (run / "judged_v1.jsonl").read_text()
    use("v1", block)
    assert "other" in (run / "judged.jsonl").read_text()
    use("v2", block)
    assert "other" in (run / "judged_v1.jsonl").read_text()  # a second switch never overwrites v1


def test_robustness_loader_and_verdict(tmp_path):
    run = _run(tmp_path)
    _labels(run, "v2", [{"seq": 10, "label": "call_out"}])
    assert loader("judged_v2.jsonl")(run)[10]["label"] == "call_out"
    assert loader(None)(run) == {}
    with pytest.raises(FileNotFoundError):
        loader("judged_v1.jsonl")(run)
    row = lambda est, p: {"confirmatory": {"H1 P on policing": {"estimate": est, "p_holm": p}}}  # noqa: E731
    same = verdict({"v2": row(0.2, 0.01), "v1": row(0.1, 0.03), "keyword": row(0.3, 0.001)})
    assert same["H1 P on policing"]["robust"]
    split = verdict({"v2": row(0.2, 0.01), "v1": row(0.1, 0.2), "keyword": row(0.3, 0.001)})
    assert not split["H1 P on policing"]["robust"]
    import numpy as np  # analyse() returns numpy floats, which once broke the sign comparison
    flipped = verdict({"v2": row(np.float64(0.2), np.float64(0.01)), "v1": row(np.float64(-0.1), np.float64(0.01)),
                       "keyword": row(np.float64(0.3), np.float64(0.001))})
    assert not flipped["H1 P on policing"]["robust"]
    assert isinstance(flipped["H1 P on policing"]["significant"]["v2"], bool)
