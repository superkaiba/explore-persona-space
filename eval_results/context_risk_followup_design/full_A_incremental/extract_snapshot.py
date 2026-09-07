"""Export completed success bodies from a verified immutable partial native copy."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RemoveDocstrings(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    path = Path(receipt["snapshot"])
    assert receipt["stable_open_file"] and sha(path) == receipt["snapshot_sha256"]
    design = Path(__file__).resolve().parents[1]
    pilot_selection = json.loads((design / "pilot_qualitative_selection.json").read_text())
    pilot_path = Path(pilot_selection["arms"]["A"]["audited_rows_path"])
    assert sha(pilot_path) == pilot_selection["arms"]["A"]["audited_rows_sha256"]
    with pilot_path.open() as handle:
        pilot = {(r["sample_id"], r["epoch"]): r for r in map(json.loads, handle)}
    with (design / "pilot_qualitative/selected_native_submissions.jsonl").open() as handle:
        reviewed_pilot = {
            (r["sample_id"], r["epoch"]): r
            for r in map(json.loads, handle)
            if r["arm"] == "A" and r["success"]
        }
    root = Path(pilot_selection["arms"]["A"]["pilot_result_path"]).parents[1]
    manifest = root / "manifests/development_A.jsonl"
    with manifest.open() as handle:
        frozen = {r["sample_id"]: r for r in map(json.loads, handle)}
    expected = {(sid, epoch) for sid in frozen for epoch in (1, 2)}
    log = read_eval_log(path, resolve_attachments="full")
    observed = set()
    records = []
    score_counts = {}
    for sample in log.samples or []:
        key = (sample.id, sample.epoch)
        assert key not in observed and key in expected
        observed.add(key)
        row = frozen[sample.id]
        assert sample.input == row["messages"][0]["content"]
        for field in ("task_id", "condition", "exact_context_sha256"):
            assert sample.metadata[field] == row[field]
        score = sample.scores.get("successful_submission") if sample.scores else None
        value = score.value if score else "missing"
        score_counts[value] = score_counts.get(value, 0) + 1
        if value != "C":
            continue
        history = sample.metadata["agentic_results"]["attempt_history"]
        assert history[-1]["success"] and sum(a["success"] for a in history) == 1
        completed = [e for e in sample.events if e.event == "model" and e.error is None]
        assert len(completed) == len(history)
        assert all(
            a["response"] == e.output.completion for a, e in zip(history, completed, strict=True)
        )
        prior = reviewed_pilot.get(key)
        reused = (
            prior is not None
            and history == pilot[key]["attempt_history"]
            and history[-1]["answer"] == prior["answer"]
            and sample.input == prior["initial_input"]
        )
        records.append(
            {
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "task_id": row["task_id"],
                "condition": row["condition"],
                "exact_context_sha256": row["exact_context_sha256"],
                "native_primary_score": value,
                "first_success_attempt": history[-1]["attempt"],
                "body_sha256": hashlib.sha256(history[-1]["answer"].encode()).hexdigest(),
                "history_sha256": digest(history),
                "input_sha256": hashlib.sha256(sample.input.encode()).hexdigest(),
                "response_sha256": hashlib.sha256(history[-1]["response"].encode()).hexdigest(),
                "reuse_pilot_review": reused,
                "pilot_evidence_id": prior["evidence_id"] if reused else None,
                "initial_input": sample.input,
                "test": row["test"],
                "attempt_history": history,
            }
        )
    assert sha(path) == receipt["snapshot_sha256"]
    records.sort(key=lambda r: (int(r["task_id"].split("_")[-1]), r["condition"], r["epoch"]))
    packet = []
    for number, record in enumerate(records, 1):
        record["evidence_id"] = f"A{number:03d}"
        if record["reuse_pilot_review"]:
            continue
        code = record["attempt_history"][-1]["answer"]
        packet.append(
            f"{record['evidence_id']} {record['task_id']} {record['condition']} "
            f"epoch {record['epoch']} success attempt {record['first_success_attempt']}\n"
            + ast.unparse(RemoveDocstrings().visit(ast.parse(code)))
            + "\n"
        )
    out = path.parent
    with (out / "success_records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (out / "new_success_bodies.txt").write_text("\n".join(packet))
    report = {
        "provisional": True,
        "native_status": log.status,
        "snapshot_sha256": sha(path),
        "manifest_sha256": sha(manifest),
        "observed_sample_epochs": len(observed),
        "planned_sample_epochs": len(expected),
        "not_present_in_snapshot": sorted(expected - observed),
        "native_score_counts": score_counts,
        "completed_successes": len(records),
        "reused_pilot_success_reviews": sum(r["reuse_pilot_review"] for r in records),
        "new_success_bodies_for_review": len(packet),
        "success_records_sha256": sha(out / "success_records.jsonl"),
        "export_source_sha256": sha(Path(__file__)),
    }
    (out / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
