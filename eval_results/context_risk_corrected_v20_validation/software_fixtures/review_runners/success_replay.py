"""Independent selection/integrity and real-Docker saved-replay fixtures.

Run from the reviewed worktree with pinned Inspect 0.3.261/OpenAI 3.7.0.
Synthetic native records are software fixtures, never experimental outcomes.
Only the Docker cases execute submissions; no case requests model generation.
"""

import copy
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inspect_ai.dataset import Sample

sys.path.insert(0, str(Path.cwd()))
import scripts.context_risk_corrected_success_replay as module

SOURCE = Path(module.__file__)
SOURCE_SHA = module.sha(SOURCE)
MANIFEST = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/data/"
    "impossible_livecodebench_promptB/public_pilot_manifest.jsonl"
)
ROOT = Path(tempfile.mkdtemp(prefix="context-risk-success-replay-review-"))
CASES = []
roster = module.load_samples(MANIFEST, None, "B")
positive_ids = {
    next(s.id for s in roster if s.metadata["condition"] == c)
    for c in ("original", "conflicting", "oneoff")
}


def make_input(name, change=None):
    directory = ROOT / name
    directory.mkdir()
    records = []
    rows = []
    counts = Counter()
    for s in roster:
        for epoch in range(1, 9):
            success = s.id in positive_ids and epoch == 1
            score = "C" if success else "I"
            answer = "def f(): return 1\n" if success else "def f(): return 0\n"
            history = [
                {
                    "attempt": 1,
                    "success": success,
                    "answer": answer,
                    "response": answer,
                    "category": "passed" if success else "assertion_failure",
                    "request_seed": epoch,
                    "stop_reasons": ["stop"],
                }
            ]
            record = SimpleNamespace(
                id=s.id,
                epoch=epoch,
                input=s.input,
                metadata={**s.metadata, "agentic_results": {"attempt_history": history}},
                scores={"successful_submission": SimpleNamespace(value=score)},
                error=None,
                invalidation=None,
            )
            records.append(record)
            rows.append(
                {
                    "id": s.id,
                    "epoch": epoch,
                    "condition": s.metadata["condition"],
                    "task_id": s.metadata["task_id"],
                    "exact_context_sha256": s.metadata["exact_context_sha256"],
                    "score": score,
                    "attempt_history": copy.deepcopy(history),
                }
            )
            counts[s.metadata["condition"]] += success
    native_path = directory / "fixture-native.eval"
    native_path.write_text("Synthetic native-record fixture; read_eval_log is patched.\n")
    result = {
        "by_condition": {
            c: {"n": 160, "passed": counts[c], "errors": 0}
            for c in ("original", "conflicting", "oneoff")
        }
    }
    audit = {
        "audit_passed": True,
        "execution_passed": True,
        "expected_unique_rollouts": 480,
        "realized_unique_rollouts": 480,
        "log_hashes": {str(native_path): module.sha(native_path)},
    }
    payload = {
        "records": records,
        "rows": rows,
        "result": result,
        "audit": audit,
        "native_path": native_path,
        "manifest": MANIFEST,
    }
    if change:
        change(payload)
    path = directory / "run_result.json"
    path.write_text(json.dumps(result))
    audit["run_result_sha256"] = module.sha(path)
    (directory / "native_log_audit.json").write_text(json.dumps(audit))
    (directory / "rollouts.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    return path, payload


def selection_case(name, change=None, expected_error=None, after_write=None):
    path, payload = make_input(name, change)
    if after_write:
        after_write(path, payload)
    with patch.object(
        module, "read_eval_log", return_value=SimpleNamespace(samples=payload["records"])
    ):
        try:
            samples, review, provenance = module.prepare(path, payload["manifest"])
        except (ValueError, FileNotFoundError) as error:
            assert expected_error and expected_error in str(error), (name, str(error))
            finding = {"case": name, "passed": True, "rejected": str(error)}
        else:
            assert expected_error is None, (name, "Expected rejection")
            assert len(samples) == 6 and len(review) == 3
            assert {r["condition"] for r in review} == {"original", "conflicting", "oneoff"}
            assert {r["key"] for r in review} == {f"{sid}:1" for sid in positive_ids}
            assert Counter(s.metadata["restore_tests"] for s in samples) == {False: 3, True: 3}
            assert all(module.sha(Path(p)) == v for p, v in provenance.items())
            finding = {
                "case": name,
                "passed": True,
                "selected": len(review),
                "executions": len(samples),
            }
    CASES.append(finding)
    print(json.dumps(finding), flush=True)


selection_case("all_three_success_conditions_selected")
selection_case("missing_export_pair", lambda p: p["rows"].pop(), "Exported rollout roster")
selection_case(
    "duplicate_export_pair",
    lambda p: p["rows"].__setitem__(-1, p["rows"][0]),
    "Exported rollout roster",
)
selection_case(
    "duplicate_native_pair",
    lambda p: p["records"].__setitem__(-1, p["records"][0]),
    "Native rollout roster",
)
selection_case(
    "wrong_export_context",
    lambda p: p["rows"][0].update(exact_context_sha256="wrong"),
    "Exported rollout differs",
)
selection_case(
    "changed_export_history",
    lambda p: p["rows"][0]["attempt_history"][0].update(answer="x=7"),
    "Exported rollout differs",
)
selection_case(
    "missing_primary_score",
    lambda p: setattr(p["records"][0], "scores", {}),
    "Exported rollout differs",
)
selection_case(
    "wrong_aggregate_successes",
    lambda p: p["result"]["by_condition"]["original"].update(passed=2),
    "Positive export counts differ",
)
selection_case(
    "failed_native_audit",
    lambda p: p["audit"].update(audit_passed=False),
    "Expected the reconciled",
)
selection_case(
    "changed_run_result",
    expected_error="Expected the reconciled",
    after_write=lambda path, payload: path.write_text(path.read_text() + "\n"),
)
selection_case(
    "changed_native_bytes",
    expected_error="Native log changed",
    after_write=lambda path, payload: payload["native_path"].write_text("changed"),
)


def changed_manifest(payload):
    path = payload["native_path"].parent / "changed-manifest.jsonl"
    path.write_bytes(MANIFEST.read_bytes() + b"\n")
    payload["manifest"] = path


selection_case("changed_manifest", changed_manifest, "Expected the reconciled")


def nonterminal_success(payload):
    row = next(r for r in payload["rows"] if r["score"] == "C")
    row["attempt_history"].append({"attempt": 2, "success": False, "answer": "x=3"})
    source = next(s for s in payload["records"] if (s.id, s.epoch) == (row["id"], row["epoch"]))
    source.metadata["agentic_results"]["attempt_history"] = copy.deepcopy(row["attempt_history"])


selection_case(
    "nonterminal_success_rejected",
    nonterminal_success,
    "Positive primary score lacks exactly one terminal success",
)


def replay_inputs():
    data = [
        ("ordinary", "def f(): return 1\n", "def check(candidate):\n    assert candidate() == 1\n"),
        (
            "check_redefinition",
            "def f(): return 1\ndef check(candidate): pass\n",
            "def check(candidate):\n    assert candidate() == 1\n    assert candidate() == 2\n",
        ),
        (
            "isolated_files",
            "from pathlib import Path\np=Path('/tmp/replay_fixture_marker')\n"
            "assert not p.exists()\np.write_text('saved')\ndef f(): return 1\n",
            "def check(candidate):\n    assert candidate() == 1\n",
        ),
    ]
    samples = []
    review = []
    for key, answer, tests in data:
        review.append(
            {
                "key": key,
                "check_name_nodes": [{"kind": "FunctionDef"}]
                if key == "check_redefinition"
                else [],
            }
        )
        for restore in (False, True):
            samples.append(
                Sample(
                    id=f"{key}:{'restored' if restore else 'observed'}",
                    input="Saved software fixture.",
                    metadata={
                        "prompt": "def f():",
                        "test": tests,
                        "entry_point": "f",
                        "saved_answer": answer,
                        "restore_tests": restore,
                        "source_key": key,
                    },
                )
            )
    provenance_path = ROOT / "replay-input-fixture.txt"
    provenance_path.write_text("Synthetic saved-replay fixtures, no model outputs.\n")
    return samples, review, {str(provenance_path): module.sha(provenance_path)}


real_inputs = replay_inputs()
with patch.object(module, "prepare", return_value=copy.deepcopy(real_inputs)):
    real = module.run(ROOT / "unused-result.json", MANIFEST, ROOT / "real_docker")
assert real["complete"] and real["model_calls"] == 0
assert real["n_sandbox_executions"] == 6 and real["observed_order_failures"] == 0
assert real["restored_test_failures"] == 1
by_key = {r["key"]: r for r in real["submissions"]}
assert by_key["check_redefinition"]["restored_test_replay"]["category"] == "assertion_failure"
assert all(
    by_key[k][mode]["success"]
    for k in ["ordinary", "isolated_files"]
    for mode in ["observed_order_replay", "restored_test_replay"]
)
CASES.append(
    {
        "case": "real_docker_isolated_six_executions",
        "passed": True,
        "native_logs": real["native_logs"],
        "model_calls": 0,
    }
)
print(json.dumps(CASES[-1]), flush=True)


def mock_log(samples):
    path = ROOT / "fake-replay-log.eval"
    path.write_text("Mocked replay result; not an experimental observation.\n")
    return SimpleNamespace(status="success", location=str(path), samples=samples)


def mock_run_case(name, change, expected_error):
    samples, review, provenance = copy.deepcopy(real_inputs)
    native = [
        SimpleNamespace(
            id=s.id,
            error=None,
            invalidation=None,
            events=[],
            metadata={"replay_execution": {"success": True, "category": "passed"}},
        )
        for s in samples
    ]
    log = mock_log(native)
    change(log, provenance)
    with (
        patch.object(module, "prepare", return_value=(samples, review, provenance)),
        patch.object(module, "eval", return_value=[log]),
    ):
        try:
            module.run(ROOT / "unused-result.json", MANIFEST, ROOT / name)
        except (ValueError, RuntimeError) as error:
            assert expected_error in str(error), (name, str(error))
            assert not (ROOT / name / "success_replay.json").exists()
            CASES.append({"case": name, "passed": True, "rejected": str(error)})
            print(json.dumps(CASES[-1]), flush=True)
        else:
            raise AssertionError(f"{name}: expected rejection")


mock_run_case(
    "native_replay_status_error",
    lambda log, prov: setattr(log, "status", "error"),
    "Replay task did not finish",
)
mock_run_case(
    "native_replay_sample_error",
    lambda log, prov: setattr(log.samples[0], "error", "fixture"),
    "Replay sample was censored",
)
mock_run_case(
    "native_replay_invalidation",
    lambda log, prov: setattr(log.samples[0], "invalidation", "fixture"),
    "Replay sample was censored",
)
mock_run_case(
    "unexpected_model_event",
    lambda log, prov: log.samples[0].events.append(SimpleNamespace(event="model")),
    "Unexpected model call",
)
mock_run_case(
    "duplicate_replay_output",
    lambda log, prov: log.samples.append(log.samples[0]),
    "Duplicate replay sample",
)
mock_run_case(
    "missing_replay_output", lambda log, prov: log.samples.pop(), "Replay coverage is incomplete"
)
mock_run_case(
    "input_changed_during_replay",
    lambda log, prov: Path(next(iter(prov))).write_text("changed"),
    "Input changed during replay",
)

with (
    patch.object(module, "prepare", return_value=([], [], {})),
    patch.object(module, "eval", side_effect=AssertionError("No execution expected")),
):
    zero = module.run(ROOT / "unused-result.json", MANIFEST, ROOT / "zero_successes")
assert zero["n_successful_submissions"] == zero["n_sandbox_executions"] == zero["model_calls"] == 0
CASES.append({"case": "zero_successes_does_not_invoke_eval", "passed": True})
assert module.sha(SOURCE) == SOURCE_SHA, "Production source changed during review"
report = {
    "verdict": "PASS",
    "source": str(SOURCE),
    "source_sha256": SOURCE_SHA,
    "fixture_root": str(ROOT),
    "n_cases": len(CASES),
    "cases": CASES,
    "scope": (
        "Local synthetic selection/error fixtures and six isolated Docker executions; "
        "zero model calls."
    ),
}
print(json.dumps(report, indent=2), flush=True)
(ROOT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
