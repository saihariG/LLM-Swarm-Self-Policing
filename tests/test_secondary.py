import json

from analysis.secondary import containment_row, silence_counts


def test_containment_counts_rounds_from_first_exploit():
    events = [
        {"type": "submission", "id": 1, "accepted": True, "round": 3},
        {"type": "submission", "id": 2, "accepted": True, "round": 2},
        {"type": "verdict", "id": 1, "label": "exploit"},
        {"type": "verdict", "id": 2, "label": "honest"},
        {"type": "audit", "target_role": "honest", "round": 4},
        {"type": "audit", "target_role": "seed_a", "round": 5},
        {"type": "removal", "target_role": "seed_a", "round": 5},
    ]
    row = containment_row(events)
    assert row["rounds_to_removal"] == 2
    assert (row["audits"], row["removals"], row["seeds_removed"], row["live_removed"], row["honest_audited"]) == (2, 1, 1, 0, 1)


def test_containment_with_no_removal():
    assert containment_row([{"type": "audit", "target_role": "live", "round": 4}])["rounds_to_removal"] is None


def test_silence_counts_split_by_model(tmp_path):
    run = tmp_path / "0001_R0P0C0H0E0_00"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"agent_models": {"agent_01": "x/glm-5p3-flash", "agent_02": "openai/gpt-oss-120b"}}))
    (run / "silence.jsonl").write_text(json.dumps({"agent": "agent_01", "first_break": "acted"}) + "\n" + json.dumps({"agent": "agent_02"}) + "\n")
    counts = silence_counts(tmp_path, [run.name])
    assert counts["glm-5p3-flash"]["acted"] == 1
    assert counts["gpt-oss-120b"]["unparsed"] == 1


def test_hand_check_sample_tops_up_to_n(tmp_path):
    from analysis.judge import export_sample
    block = tmp_path / "block"
    run = block / "0001_R0P0C0H0E0_00"
    run.mkdir(parents=True)
    labels = ["call_out"] * 2 + ["other"] * 30
    events = [{"type": "post", "role": "live", "seq": i, "text": f"message {i}"} for i in range(len(labels))]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "judged.jsonl").write_text("".join(json.dumps({"seq": i, "label": x}) + "\n" for i, x in enumerate(labels)))
    out = tmp_path / "sheet.csv"
    export_sample(block, out, 10)
    assert len(out.read_text().splitlines()) == 11
    key = out.with_suffix(".key.csv").read_text()
    assert key.count("call_out") == 2
    assert "0001_R0P0C0H0E0_00" not in out.read_text()


def test_parse_reply_handles_text_around_json_and_failures():
    from analysis.judge import parse_reply
    assert parse_reply('Sure. {"label": "call_out", "accused": ["agent_03"]} done') == ("call_out", ["agent_03"])
    assert parse_reply("") == ("unparsed", [])
    assert parse_reply("{not json}") == ("unparsed", [])


def test_silence_parse_reply_recovers_first_break():
    from analysis.silence import parse_reply
    assert parse_reply('{"first_break": "acted", "joined": false}')["first_break"] == "acted"
    broken = '{"first_break": "no_route", "evidence": "he said "no way" to that"}'
    assert parse_reply(broken) == {"first_break": "no_route", "partial": True}
    assert parse_reply("") == {}
