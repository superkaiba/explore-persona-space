"""Bounded final report, paired figure, and archive-dispatch checks; writes review only."""

import ast
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
ROOT = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/"
    "impossible_livecodebench_v20_corrected"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = OUT / "results.md"
    text = report_path.read_text()
    initial_sha = sha(report_path)
    run = json.loads((ROOT / "full/run_result.json").read_text())
    native_review = json.loads((OUT / "native_reconciliation.json").read_text())
    for name, digest in native_review["input_hashes"].items():
        assert sha(Path(name)) == digest
    manifest_path = (
        ROOT.parent / "data/impossible_livecodebench_promptB/public_pilot_manifest.jsonl"
    )
    manifest = {
        f"{r['task_id']}:{r['condition']}": r
        for r in map(json.loads, manifest_path.read_text().splitlines())
    }
    rows = {
        (r["id"], r["epoch"]): r
        for r in map(json.loads, (ROOT / "full/rollouts.jsonl").read_text().splitlines())
    }
    examples = []
    for detail in re.findall(r"<details>\s*(.*?)\s*</details>", text, re.S):
        match = re.search(r"<summary>([^,]+), epoch (\d+), attempt (\d+): (\w+)</summary>", detail)
        sid, epoch, attempt, category = match.groups()
        history = rows[sid, int(epoch)]["attempt_history"][int(attempt) - 1]
        user = re.search(r"~~~~text\n(.*?)\n~~~~", detail, re.S).group(1)
        answer = re.search(r"~~~~python\n(.*?)\n~~~~", detail, re.S).group(1)
        assert user == manifest[sid]["messages"][0]["content"].rstrip("\n")
        assert answer == history["answer"].rstrip("\n") and history["category"] == category
        examples.append({"id": sid, "epoch": int(epoch), "attempt": int(attempt), "exact": True})
    assert len(examples) == 4 and text.count("**Initial user input (verbatim):**") == 4
    selection_path = ROOT / "setup/report_example_selection.json"
    selection = json.loads(selection_path.read_text())
    lengths = defaultdict(int)
    for row in manifest.values():
        lengths[row["task_id"]] += len(row["messages"][0]["content"])
    assert min(lengths, key=lambda key: (lengths[key], key)) == "lcbhard_15"
    assert selection["manifest_sha256"] == sha(manifest_path)
    assert datetime.fromisoformat(selection["selected_at"]) < datetime(
        2026, 9, 7, 9, 2, 5, tzinfo=UTC
    )
    assert all(row["task_id"] == "lcbhard_15" for row in selection["examples"])
    fig = WORK / "figures/issue_2670"
    figure_data = json.loads((fig / "figure_data.json").read_text())
    sidecar = json.loads((fig / "observed_outcomes.meta.json").read_text())
    expected = [
        {
            "panel": "All observed rollouts",
            "task_id": "all",
            "condition": c,
            "successes": run["by_condition"][c]["passed"],
            "rollouts": 160,
        }
        for c in figure_data["conditions"]
    ] + [
        {
            "panel": "Per-context outcomes",
            "task_id": task,
            "condition": c,
            "successes": next(
                row["passed"]
                for row in run["contexts"]
                if row["task_id"] == task and row["condition"] == c
            ),
            "rollouts": 8,
        }
        for task in figure_data["task_ids"]
        for c in figure_data["conditions"]
    ]
    assert sidecar["points"] == expected and sidecar["total_points"] == 63
    assert sidecar["n_series"] == 2
    assert sidecar["text"]["axes"][1]["annotations"] == [str(r["successes"]) for r in expected[3:]]
    assert Image.open(fig / "observed_outcomes.png").info["RenderId"] == sidecar["render_id"]
    for name, digest in figure_data["figures"].items():
        assert sha(WORK / name) == digest
    figure_url = re.search(
        r"https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        r"([a-f0-9]{40})/figures/issue_2670/observed_outcomes.png",
        text,
    )
    revision = figure_url.group(1)
    for name in ("observed_outcomes.png", "observed_outcomes.meta.json", "figure_data.json"):
        committed = subprocess.check_output(
            ["git", "show", f"{revision}:figures/issue_2670/{name}"], cwd=WORK
        )
        assert committed == (fig / name).read_bytes()
    figure_source = WORK / "scripts/context_risk_corrected_figures.py"
    assert figure_data["source_sha256"] == sha(figure_source)
    assert (
        subprocess.check_output(
            ["git", "show", f"{revision}:scripts/context_risk_corrected_figures.py"], cwd=WORK
        )
        == figure_source.read_bytes()
    )
    for name in ("metrics.json", "native_log_audit.json"):
        assert (
            subprocess.check_output(
                [
                    "git",
                    "show",
                    "57d38bbd0f053c55775ebd7e79bd242a0153df51:"
                    f"eval_results/context_risk_corrected_v20_report/{name}",
                ],
                cwd=WORK,
            )
            == (OUT / name).read_bytes()
        )
    audit = json.loads((OUT / "native_log_audit.json").read_text())
    assert audit["native_log_wall_seconds"] == 6355
    analysis_path = ROOT / "analysis/analysis_result.json"
    analysis = json.loads(analysis_path.read_text())
    assert analysis["analysis_completed"] is True
    assert (
        analysis["reward_hacking_feasibility"]["prediction_status"] == "not_run_frozen_gate_failed"
    )
    assert analysis["reward_hacking_feasibility"]["gate"] == run["reward_hacking_prevalence_gate"]
    archive_source = WORK / "scripts/context_risk_corrected_final_archive.py"
    archive_wrapper = WORK / "scripts/context_risk_corrected_final_archive_supervise.sh"
    finish_wrapper = WORK / "scripts/context_risk_corrected_finish_supervise.sh"
    assert archive_wrapper.read_text() == finish_wrapper.read_text().replace(
        "mode=finish", "mode=final_archive"
    ).replace(
        "-m scripts.context_risk_corrected_finish &",
        "-m scripts.context_risk_corrected_final_archive &",
    )
    tree = ast.parse(archive_source.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    uploads = [
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "upload"
    ]
    assert len(uploads) == 1
    assert ast.unparse(uploads[0]) == "upload(STAGING / 'final', 'final')"
    assert {ast.unparse(node.func) for node in calls} == {
        "load_dotenv",
        "str",
        "os.environ.update",
        "upload",
    }
    assert 'EPM_HF_FILECOUNT_FALLBACK="0"' in archive_source.read_text()
    assert 'EPM_HF_RETRY_BUDGET_S="1800"' in archive_source.read_text()
    files = [
        report_path,
        selection_path,
        fig / "figure_data.json",
        fig / "observed_outcomes.meta.json",
        figure_source,
        archive_source,
        archive_wrapper,
        analysis_path,
        Path(__file__),
    ]
    result = {
        "reviewer": "/root/reward_harness_critic",
        "verdict": "PASS",
        "checked_at": datetime.now(UTC).isoformat(),
        "report_sha256": initial_sha,
        "input_hashes": {str(path): sha(path) for path in files},
        "exact_examples": examples,
        "selection": {"shortest_task": "lcbhard_15", "recorded_at": selection["selected_at"]},
        "figure": {
            "url": figure_url.group(0),
            "verdict": "PASS",
            "sidecar_rows": 63,
            "paired_render_id": sidecar["render_id"],
            "rendered_image_viewed": True,
        },
        "analysis_status": "not_run_frozen_gate_failed",
        "archive_dispatch": (
            "PASS; exact approved supervisor with only mode/module changed; existing reviewed "
            "upload helper, final-only stage and prefix, fallback disabled, retry budget pinned. "
            "No external effects invoked."
        ),
        "report_interpretation": (
            "PASS; sparse positives and operational competence limitations stated; no predictor "
            "effect claimed; original-success hardcoding and runtime/seed confounding disclosed."
        ),
        "resolved_findings": [
            "Clarified rejection of malformed final fence and labeled displayed initial inputs.",
            (
                "Canonical sidecar points now contain all 63 correct data rows; "
                "erroneous generic horizontal-bar midpoints removed."
            ),
            "Final report figure and plotting source use the actual matching committed export.",
        ],
        "scope": (
            "Scientific/report/dispatch review. Owner remains responsible for final archive "
            "receipt, task body verification, methodology export, and final workflow state. "
            "Immutable evidence links may be appended without altering scientific text."
        ),
        "no_model_upload_pod_or_task_calls": True,
    }
    assert sha(report_path) == initial_sha
    (OUT / "final_report_review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"verdict": "PASS", "report_sha256": initial_sha, "examples": 4, "sidecar_rows": 63}
        )
    )


if __name__ == "__main__":
    main()
