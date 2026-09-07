"""Memory-only future-artifact stand-ins for driver joins; no experiment results written."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import context_risk_followup_analyze as driver
from scripts import context_risk_followup_audit as audit_module
from scripts import context_risk_followup_capture as capture_module

ROOT = Path(__file__).resolve().parents[3]
ACTUAL_ROOT = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/"
    "impossible_ten_attempts_followup"
)
MAP_PATH = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/"
    "qwen38_map_pilot/map_layer_44.npz"
)
SPEC_PATH = ROOT / "eval_results/context_risk_followup_design/analysis_spec.json"


def memory_fixture():
    """Use the actual frozen roster/map but keep every fake future outcome in memory."""
    root, captures = Path("/virtual-followup-software"), Path("/virtual-captures-software")
    stage = root / "fresh_A"
    manifest = root / "manifests/fresh_A.jsonl"
    manifest_text = (ACTUAL_ROOT / "manifests/fresh_A.jsonl").read_text()
    freeze_text = (ACTUAL_ROOT / "manifests/freeze.json").read_text()
    rows = [json.loads(line) for line in manifest_text.splitlines()]
    files = {manifest: manifest_text, root / "manifests/freeze.json": freeze_text}

    def put(path, value):
        files[path] = json.dumps(value)

    def sha(path):
        return hashlib.sha256(files[path].encode()).hexdigest()

    for arm in ("A", "B"):
        put(root / f"development_{arm}/run_result.json", {"software_fixture": arm})
    put(
        root / "selection.json",
        {
            "passed": True,
            "selected_arm": "A",
            "selected_unix": 100.0,
            "freeze_sha256": sha(root / "manifests/freeze.json"),
            "arms": {
                arm: {"source_sha256": sha(root / f"development_{arm}/run_result.json")}
                for arm in ("A", "B")
            },
        },
    )
    audit_report = {
        "passed": True,
        "realized_rollouts": 996,
        "requested_rollouts": 996,
        "phase": "fresh",
        "arm": "A",
        "is_pilot": False,
        "technical_errors": 0,
        "first_model_request_unix": 201.0,
    }
    put(stage / "run_result.json", {"software_fixture": "future run result"})
    put(stage / "audit/native_audit.json", audit_report)
    outcomes = []
    prefixes = []
    capture_rows = {}
    activation = {}
    vector = np.zeros(5120, dtype=np.float32)
    for row in rows:
        key = row["exact_context_sha256"]
        for epoch in (1, 2, 3, 4):
            outcomes.append(
                {
                    "sample_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "condition": row["condition"],
                    "public_test_role": row["public_test_role"],
                    "exact_context_sha256": key,
                    "epoch": epoch,
                    "score": "C" if epoch == 1 else "I",
                    "requests": [{"usage": {"input_tokens": 2}}],
                }
            )
        prefix = {
            "sample_id": row["sample_id"],
            "exact_context_sha256": key,
            "token_ids": [1, 2],
            "n_prefix_tokens": 2,
            "prefix_token_ids_sha256": driver.digest([1, 2]),
        }
        prefixes.append(prefix)
        capture_rows[key] = {
            **prefix,
            "task_id": row["task_id"],
            "condition": row["condition"],
            "public_test_role": row["public_test_role"],
        }
        activation[key] = vector
    files[stage / "audit/audited_rollouts.jsonl"] = "\n".join(json.dumps(r) for r in outcomes)
    put(stage / "prefix_tokens.json", {"passed": True, "n_contexts": 249, "contexts": prefixes})
    put(
        stage / "launch_config_fixture.json",
        {
            "started_unix": 200.0,
            "metadata": {"phase": "fresh", "arm": "A", "manifest_sha256": sha(manifest)},
        },
    )
    chunk = {"fingerprint": "software_capture_recipe", "n_contexts": 249}
    put(captures / "chunk_0000.done.json", chunk)
    put(
        captures / "run_result.json",
        {
            "passed": True,
            "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "model_id": "Qwen/Qwen3.8-27B",
            "n_contexts": 249,
            "prefixes_truncated": 0,
            "capture_layers": [44],
            "fingerprint": "software_capture_recipe",
            "chunks": [chunk],
        },
    )
    return root, captures, files, audit_report, activation, capture_rows


def invoke_fixture(state, *, binding_error=False):
    """Patch only I/O and the separately reviewed native-audit/capture-loader boundaries."""
    root, captures, files, report, activation, capture_rows = state
    original_read, original_sha = Path.read_text, driver.sha256
    original_glob, original_iterdir, original_is_file = Path.glob, Path.iterdir, Path.is_file

    def read(path, *args, **kwargs):
        return files[path] if path in files else original_read(path, *args, **kwargs)

    def sha(path):
        return (
            hashlib.sha256(files[path].encode()).hexdigest()
            if path in files
            else original_sha(path)
        )

    def glob(path, pattern):
        if path in {captures, root / "fresh_A"}:
            return iter(sorted(p for p in files if p.parent == path and p.match(pattern)))
        return original_glob(path, pattern)

    def iterdir(path):
        return (
            iter(sorted(p for p in files if p.parent == path))
            if path == captures
            else original_iterdir(path)
        )

    def is_file(path):
        return True if path in files else original_is_file(path)

    with (
        patch.object(Path, "read_text", read),
        patch.object(driver, "sha256", side_effect=sha),
        patch.object(Path, "glob", glob),
        patch.object(Path, "iterdir", iterdir),
        patch.object(Path, "is_file", is_file),
        patch.object(audit_module, "audit", return_value=report),
        patch.object(
            capture_module,
            "validate_binding",
            return_value={"passed": True, "software_boundary_fixture": True},
            side_effect=ValueError("software binding rejected") if binding_error else None,
        ) as binding,
        patch.object(
            driver, "load_impossible_activations", return_value=(activation, capture_rows)
        ),
    ):
        result = driver.load_inputs(root, captures, MAP_PATH, SPEC_PATH)
    binding.assert_called_once_with(
        captures,
        root / "manifests/fresh_A.jsonl",
        root / "fresh_A/prefix_tokens.json",
        root / "selection.json",
    )
    return result


def test_complete_join_uses_exact_context_and_token_identity():
    rows, raw, arrays, _, provenance = invoke_fixture(memory_fixture())
    assert len(rows) == 249 and raw.shape == (249, 5120)
    assert arrays["weight"].shape == (5120, 5120)
    assert sum(r["trials"] for r in rows) == 996
    assert all(r["positive"] == 1 for r in rows)
    assert len({r["task_id"] for r in rows if r["public_test_role"] == "probe_training"}) == 63
    assert len({r["task_id"] for r in rows if r["public_test_role"] == "final_test"}) == 20
    assert provenance["spec_sha256"] == driver.SPEC_SHA


def test_capture_validator_rejection_propagates_without_fallback():
    """The separately reviewed capture binding must approve before inputs are returned."""
    with pytest.raises(ValueError, match="software binding rejected"):
        invoke_fixture(memory_fixture(), binding_error=True)


@pytest.mark.parametrize(
    "defect",
    [
        "request_before_selection",
        "duplicate_prefix",
        "prefix_hash",
        "capture_token",
        "request_length",
        "capture_role",
        "capture_model",
        "truncation",
        "capture_roster",
        "duplicate_epoch",
        "chunk_fingerprint",
    ],
)
def test_join_rejects_misaligned_complete_artifact_claims(defect):
    state = memory_fixture()
    root, captures, files, report, activation, capture_rows = state
    key = next(iter(activation))
    prefix_path = root / "fresh_A/prefix_tokens.json"
    if defect == "request_before_selection":
        report["first_model_request_unix"] = 99
    elif defect in {"duplicate_prefix", "prefix_hash"}:
        value = json.loads(files[prefix_path])
        if defect == "duplicate_prefix":
            value["contexts"].append(copy.deepcopy(value["contexts"][0]))
        else:
            value["contexts"][0]["token_ids"] = [99, 98]
        files[prefix_path] = json.dumps(value)
    elif defect == "capture_token":
        capture_rows[key]["prefix_token_ids_sha256"] = "wrong"
    elif defect == "capture_role":
        capture_rows[key]["public_test_role"] = "wrong_role"
    elif defect in {"request_length", "duplicate_epoch"}:
        path = root / "fresh_A/audit/audited_rollouts.jsonl"
        outcomes = [json.loads(s) for s in files[path].splitlines()]
        if defect == "request_length":
            outcomes[0]["requests"][0]["usage"]["input_tokens"] = 3
        else:
            outcomes[0]["epoch"] = outcomes[1]["epoch"]
        files[path] = "\n".join(json.dumps(r) for r in outcomes)
    elif defect in {"capture_model", "truncation", "chunk_fingerprint"}:
        path = captures / "run_result.json"
        value = json.loads(files[path])
        if defect == "capture_model":
            value["model_revision"] = "wrong"
        elif defect == "truncation":
            value["prefixes_truncated"] = 1
        else:
            value["fingerprint"] = "wrong"
        files[path] = json.dumps(value)
    elif defect == "capture_roster":
        activation.pop(key)
    with pytest.raises(ValueError):
        invoke_fixture(state)
