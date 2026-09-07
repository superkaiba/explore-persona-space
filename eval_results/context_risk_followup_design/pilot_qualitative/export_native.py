"""Read-only native pilot selection/export; no model or sandbox calls."""

import ast
import hashlib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

OUT = Path(__file__).resolve().parent
SELECTION = OUT.parent / "pilot_qualitative_selection.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DisplayCompact(ast.NodeTransformer):
    """Omit docstrings and abbreviate large literal data ONLY in reading copies."""

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 300:
            return ast.copy_location(
                ast.Constant(
                    value=(
                        f"DISPLAY: {len(node.value)}-character string; "
                        f"first 100={node.value[:100]!r}"
                    )
                ),
                node,
            )
        return node

    def visit_List(self, node):
        if len(node.elts) > 20:
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                return self.generic_visit(node)
            return ast.copy_location(
                ast.Constant(
                    value=(
                        f"DISPLAY: literal list of {len(value)} elements, "
                        f"first 3={value[:3]!r}, last 3={value[-3:]!r}"
                    )
                ),
                node,
            )
        return self.generic_visit(node)

    def visit_Dict(self, node):
        if len(node.keys) > 20:
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                return self.generic_visit(node)
            return ast.copy_location(
                ast.Constant(
                    value=(
                        f"DISPLAY: literal dictionary of {len(value)} entries, "
                        f"first 3={list(value.items())[:3]!r}, "
                        f"last 3={list(value.items())[-3:]!r}"
                    )
                ),
                node,
            )
        return self.generic_visit(node)


def compact(code):
    try:
        return ast.unparse(DisplayCompact().visit(ast.parse(code)))
    except SyntaxError:
        return "[AST parse failed; verbatim submission]\n" + code


def main():
    selection = json.loads(SELECTION.read_text())
    selected = []
    evidence = {"selection_sha256": sha(SELECTION), "native_verified": {}}
    for arm, plan in selection["arms"].items():
        assert sha(Path(plan["pilot_result_path"])) == plan["pilot_result_sha256"]
        assert sha(Path(plan["audited_rows_path"])) == plan["audited_rows_sha256"]
        with Path(plan["audited_rows_path"]).open() as handle:
            exported = {(r["sample_id"], r["epoch"]): r for r in map(json.loads, handle)}
        native = {}
        for location, expected in plan["native_logs_sha256"].items():
            assert sha(Path(location)) == expected
            log = read_eval_log(location, resolve_attachments="full")
            for sample in log.samples:
                key = (sample.id, sample.epoch)
                assert key not in native
                native[key] = (sample, location)
        assert set(native) == set(exported)
        for key, (sample, _) in native.items():
            assert (
                sample.metadata["agentic_results"]["attempt_history"]
                == exported[key]["attempt_history"]
            )
            assert sample.scores["successful_submission"].value == exported[key]["score"]
        impossible = {tuple(x) for x in plan["selected_impossible_keys"]}
        original = {tuple(x) for x in plan["original_success_keys"]}
        for key in sorted(impossible | original):
            sample, location = native[key]
            row = exported[key]
            history = row["attempt_history"]
            if key in original:
                attempts = [history[-1]]
            elif (
                arm == "A"
                and row["task_id"] == "lcbhard_9"
                and row["condition"] == "conflicting"
                and row["epoch"] == 2
            ):
                attempts = history
            else:
                attempts = [history[0], history[-1]]
            for attempt in attempts:
                selected.append(
                    {
                        "arm": arm,
                        "sample_id": sample.id,
                        "epoch": sample.epoch,
                        "task_id": row["task_id"],
                        "condition": row["condition"],
                        "trajectory_score": row["score"],
                        "native_log": location,
                        "native_log_sha256": plan["native_logs_sha256"][location],
                        "initial_input": sample.input,
                        "selection": "original_success" if key in original else "fixed_impossible",
                        **attempt,
                    }
                )
        evidence["native_verified"][arm] = {
            "samples": len(native),
            "selected_impossible": len(impossible),
            "original_successes": len(original),
        }
    selected.sort(
        key=lambda r: (
            r["arm"],
            int(r["task_id"].split("_")[-1]),
            r["condition"],
            r["epoch"],
            r["attempt"],
        )
    )
    raw = OUT / "selected_native_submissions.jsonl"
    with raw.open("w") as handle:
        for number, row in enumerate(selected, 1):
            row["evidence_id"] = f"S{number:03d}"
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    evidence["selected_submissions"] = len(selected)
    evidence["raw_sha256"] = sha(raw)
    evidence["export_source_sha256"] = sha(Path(__file__))
    (OUT / "integrity.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for arm in ("A", "B"):
        chunks = []
        for row in selected:
            if row["arm"] != arm:
                continue
            title = (
                f"{row['evidence_id']} {arm} {row['task_id']} {row['condition']} "
                f"epoch{row['epoch']} attempt{row['attempt']} "
                f"success={row['success']} category={row['category']}"
            )
            feedback = row["stderr"] or row["stdout"] or "<empty>"
            if len(feedback) > 3000:
                feedback = (
                    feedback[:1000] + "\n[DISPLAY: repeated traceback middle omitted; "
                    "full feedback retained in raw export]\n" + feedback[-1000:]
                )
            chunks.append(
                title + "\n" + compact(row["answer"] or "") + "\nFEEDBACK:\n" + feedback + "\n"
            )
        (OUT / f"reading_{arm}.txt").write_text("\n".join(chunks))
    print(json.dumps(evidence, indent=2))
    for arm in ("A", "B"):
        path = OUT / f"reading_{arm}.txt"
        print(arm, "compact reading chars", len(path.read_text()))


if __name__ == "__main__":
    main()
