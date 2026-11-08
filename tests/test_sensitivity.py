import json

from analysis.sensitivity import affected_runs, lost_agents


def write_run(path, events):
    path.mkdir()
    (path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_lost_agent_counts_only_refused_histories(tmp_path):
    run = tmp_path / "0001_R0P0C0H0E0_00"
    refused = "BadRequestError: Error code: 400 - Invalid tool call in messages"
    write_run(run, [
        {"type": "assistant", "agent": "agent_03", "round": 2, "error": ""},
        {"type": "assistant", "agent": "agent_04", "round": 2, "error": "APITimeoutError: timed out"},
        {"type": "assistant", "agent": "agent_03", "round": 8, "error": refused},
        {"type": "assistant", "agent": "agent_03", "round": 9, "error": refused},
        {"type": "debrief", "agent": "agent_03", "round": 16, "error": refused},
    ])
    assert lost_agents(run) == {"agent_03": {"first_round": 8, "failed_calls": 3}}


def test_affected_runs_skips_clean_swarms(tmp_path):
    write_run(tmp_path / "clean", [{"type": "assistant", "agent": "agent_03", "round": 1, "error": ""}])
    write_run(tmp_path / "hit", [{"type": "assistant", "agent": "agent_05", "round": 4, "error": "Invalid tool call"}])
    assert list(affected_runs(tmp_path, ["clean", "hit"])) == ["hit"]
