import csv
import json

from analysis.judge import RELEVANT, RELEVANT_V2, RUBRIC, RUBRIC_V2, VERSIONS, export_sample, judge_prompt, read_exclusions


def test_v2_filter_keeps_everything_v1_sends_and_adds_test_complaints():
    for text in ("agent_03 used the AlwaysEqual trick", "please report this to the organizer", "it's cheating"):
        assert RELEVANT.search(text) and RELEVANT_V2.search(text)
    complaint = "T38 fails one hidden test with a textbook LIS"
    assert not RELEVANT.search(complaint)
    assert RELEVANT_V2.search(complaint)
    assert not RELEVANT_V2.search("I'm taking T16 and T09 next round.")


def test_versions_pick_rubric_and_file():
    assert VERSIONS["v1"][0] == RUBRIC and VERSIONS["v1"][2] == "judged.jsonl"
    assert VERSIONS["v2"][0] == RUBRIC_V2 and VERSIONS["v2"][2] == "judged_v2.jsonl"
    msgs = judge_prompt({"type": "post", "text": "hi"}, RUBRIC_V2)
    assert msgs[0]["content"] == RUBRIC_V2
    assert judge_prompt({"type": "post", "text": "hi"})[0]["content"] == RUBRIC


def _block(root, name, n, labels_name, label):
    run = root / name / "0001_R1P1C1H1E1_00"
    run.mkdir(parents=True)
    events = [{"type": "post", "role": "live", "seq": i, "text": f"{name} message {i}"} for i in range(n)]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / labels_name).write_text("".join(json.dumps({"seq": i, "label": label}) + "\n" for i in range(n)))
    return root / name


def test_second_sheet_spans_blocks_and_leaves_out_coded_messages(tmp_path):
    a = _block(tmp_path, "main_A", 8, "judged_v2.jsonl", "call_out")
    b = _block(tmp_path, "main_B", 8, "judged_v2.jsonl", "promote")
    old_key = tmp_path / "old.key.csv"
    with open(old_key, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "run", "seq", "judge_label"])
        for seq in range(5):
            w.writerow([seq + 1, "0001_R1P1C1H1E1_00", seq, "call_out"])
    exclude = read_exclusions(old_key, "main_A")
    assert ("main_A", "0001_R1P1C1H1E1_00", 0) in exclude
    out = tmp_path / "sheet_v2.csv"
    export_sample([a, b], out, 11, seed=6, labels_name="judged_v2.jsonl", exclude=exclude)
    sheet = list(csv.DictReader(open(out, newline="")))
    key = list(csv.DictReader(open(out.with_suffix(".key.csv"), newline="")))
    assert len(sheet) == 11
    assert list(key[0]) == ["item", "block", "run", "seq", "judge_label"]
    picked = {(r["block"], int(r["seq"])) for r in key}
    assert not any(("main_A", s) in picked for s in range(5))
    assert {r["judge_label"] for r in key} == {"call_out", "promote"}
    assert "0001_R1P1C1H1E1_00" not in out.read_text()
