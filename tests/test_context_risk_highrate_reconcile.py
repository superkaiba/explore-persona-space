"""Adversarial archive-identity tests; fixtures never stand in for experimental evidence."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from inspect_ai.log import EvalSample

from scripts import context_risk_highrate_reconcile as reconcile


def stamp(path: Path) -> dict:
    """Describe actual fixture bytes using the production digest implementation."""
    return {"size": path.stat().st_size, "sha256": reconcile.archive.sha(path)}


def write_json(path: Path, value: dict) -> None:
    """Write a real fixture document with native UTF-8 contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


@pytest.mark.parametrize("name", ["", ".", "../outside", "/absolute", "a/../b", "a//b", "a/./b"])
def test_reject_noncanonical_paths(tmp_path, name):
    with pytest.raises(ValueError):
        reconcile.safe_child(tmp_path, name)


def test_reject_symlinked_parent(tmp_path):
    (tmp_path / "actual").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "actual", target_is_directory=True)
    with pytest.raises(ValueError):
        reconcile.safe_child(tmp_path, "link/file")


@pytest.mark.parametrize("actual", [[("x", 1)], [("x", 1), ("x", 1)], [("x", 1), ("z", 2)]])
def test_exact_keys_reject_missing_duplicate_and_equal_count_substitution(actual):
    with pytest.raises(ValueError):
        reconcile.exact_keys(actual, {("x", 1), ("y", 2)}, "fixture")


@pytest.mark.parametrize("epoch", [True, "1", 0, 5, 1.0])
def test_generation_epoch_requires_literal_integer(epoch):
    with pytest.raises(ValueError):
        reconcile.generation_keys([{"sample_id": "x", "epoch": epoch}])


def test_unicode_rows_project_to_real_mechanical_checker(tmp_path):
    original = tmp_path / "source.jsonl"
    write_json(original, {"sample_id": "x\u2028y\u2029z", "epoch": 1, "text": "a\u2028b"})
    assert len(original.read_text().splitlines()) > 1
    rows = reconcile.read_rows(original)
    keys = reconcile.generation_keys(rows)
    reconcile.exact_keys(keys, {("x\u2028y\u2029z", 1)}, "unicode")
    index_root = tmp_path / "indexes"
    index = reconcile.write_index(index_root, "screen_B", keys)
    assert Path(index["path"]).read_bytes().isascii()
    checked = reconcile.verify_uploads.check_realized_row_counts(
        expected_rows={"screen_B": 1},
        local_root=str(index_root),
        glob_pattern="row_index*.jsonl",
        distinct_key_fields=("key",),
    )
    assert checked["status"] == "OK"
    assert checked["labels"]["screen_B"]["realized_distinct"] == 1


def test_bridge_requires_explicit_source_and_same_bytes(tmp_path):
    local, remote = tmp_path / "source", tmp_path / "remote"
    local.write_text("right")
    remote.write_bytes(local.read_bytes())
    manifest = {"files": {"run/source": {"source": str(local), **stamp(local)}}}
    originals = {"run/source": remote}
    assert reconcile.bridge(local, manifest, originals) == remote
    with pytest.raises(ValueError):
        reconcile.bridge(tmp_path / "unmapped", manifest, originals)
    remote.write_text("wrong")
    with pytest.raises(ValueError):
        reconcile.bridge(local, manifest, originals)


def test_pod_coverage_keeps_relative_paths_and_logs(tmp_path):
    originals, records = {}, {}
    for relative, body in [("server/run.log", "first"), ("setup/run.log", "other")]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        originals[f"pod/{relative}"] = path
        records[relative] = stamp(path)
    inventory = {
        "pod": reconcile.POD,
        "pod_id": reconcile.POD_ID,
        "source_root": reconcile.POD_ROOT,
        "all_owned_workers_drained": True,
        "files": records,
    }
    manifest = {"files": {f"pod/{k}": v for k, v in records.items()}}
    assert set(reconcile.pod_coverage(inventory, manifest, originals)) == set(records)
    missing = dict(originals)
    missing.pop("pod/server/run.log")
    with pytest.raises(ValueError):
        reconcile.pod_coverage(inventory, manifest, missing)
    wrong = copy.deepcopy(inventory)
    wrong["files"]["server/run.log"] = records["setup/run.log"]
    with pytest.raises(ValueError):
        reconcile.pod_coverage(wrong, manifest, originals)
    wrong["files"] = {}
    with pytest.raises(ValueError):
        reconcile.pod_coverage(wrong, manifest, originals)


def raw_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small explicit archive fixture for readback trust-boundary tests."""
    root, readback = tmp_path / "run", tmp_path / "download"
    root.mkdir()
    remote = readback / reconcile.PREFIX
    remote.mkdir(parents=True)
    name = "run/screen_B/rollouts.jsonl"
    reconstructed = readback / "reconstructed" / name
    write_json(reconstructed, {"sample_id": "fixture", "epoch": 1, "text": "a\u2028b"})
    source = {"source": str(root / "screen_B/rollouts.jsonl"), **stamp(reconstructed)}
    part = remote / "run/screen_B/rollouts.part000.jsonl"
    part.parent.mkdir(parents=True)
    part.write_bytes(reconstructed.read_bytes())
    write_json(remote / "snapshot_manifest.json", {"phase": "raw", "files": {name: source}})
    files = {str(p.relative_to(readback)): stamp(p) for p in remote.rglob("*") if p.is_file()}
    identity = {
        "repo_id": reconcile.REPO,
        "prefix": reconcile.PREFIX,
        "revision": "a" * 40,
        "url": "https://example.invalid/test-fixture",
    }
    uploaded = {"passed": True, **identity, "files": files}
    write_json(root / "raw_upload_receipt.json", uploaded)
    receipt = {
        "passed": True,
        "phase": "raw",
        **identity,
        "sources_sha256": reconcile.archive.source_hashes(),
        "upload_receipt_sha256": reconcile.archive.sha(root / "raw_upload_receipt.json"),
        "downloaded_files": len(files),
        "snapshot_sha256": reconcile.archive.sha(remote / "snapshot_manifest.json"),
        "original_files": {name: stamp(reconstructed)},
    }
    write_json(root / "raw_readback_receipt.json", receipt)
    return root, readback


def test_readback_uses_reconstructed_original_and_pinned_mapping(tmp_path):
    root, readback = raw_fixture(tmp_path)
    receipt, manifest, originals = reconcile.open_readback(root, readback)
    name = "run/screen_B/rollouts.jsonl"
    assert originals[name] == readback / "reconstructed" / name
    assert receipt["phase"] == manifest["phase"] == "raw"
    originals[name].write_text("changed")
    with pytest.raises(ValueError):
        reconcile.open_readback(root, readback)


@pytest.mark.parametrize(
    "mutation", ["missing_download", "extra_download", "wrong_revision", "wrong_phase"]
)
def test_readback_rejects_wrong_snapshot_or_name_set(tmp_path, mutation):
    root, readback = raw_fixture(tmp_path)
    remote = readback / reconcile.PREFIX
    if mutation == "missing_download":
        (remote / "run/screen_B/rollouts.part000.jsonl").unlink()
    elif mutation == "extra_download":
        (remote / "extra").write_text("unexpected")
    else:
        path = root / "raw_readback_receipt.json"
        value = reconcile.read_json(path)
        value["revision" if mutation == "wrong_revision" else "phase"] = "wrong"
        write_json(path, value)
    with pytest.raises(ValueError):
        reconcile.open_readback(root, readback)


def full_flow_fixture(tmp_path, monkeypatch, mutation=None):
    """Build explicit978-row/90-context fixtures; stub only existing semantic/reader seams.

    These are plumbing tests, never evidence of benchmark or model correctness. All
    new adapter functions, byte bridges, set comparisons and mechanical counting
    execute their actual production bodies. Existing collection/capture semantics
    and the Inspect file reader are signature-constrained external boundaries here.
    """
    root, readback, output = tmp_path / "run", tmp_path / "download", tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(reconcile.archive, "RUN_ROOT", root)
    source_review = tmp_path / "external_design_review.json"
    write_json(source_review, {"fixture": True})
    audits, phase_inputs, logs = {}, {}, {}
    for phase, n_tasks, epochs in (("screen", 103, 2), ("fresh", 30, 4)):
        phase_dir = root / f"{phase}_B"
        phase_dir.mkdir()
        rows = []
        for task in range(n_tasks):
            for condition in ("original", "conflicting", "oneoff"):
                identity = f"fixture:{phase}:task{task}:{condition}"
                rows.append(
                    {
                        "sample_id": identity,
                        "task_id": f"task{task}",
                        "condition": condition,
                        "exact_context_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                    }
                )
        manifest = root / f"manifests/{phase}_B.jsonl"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        phase_inputs[phase] = (manifest, {}, epochs)
        samples = [
            EvalSample(
                id=r["sample_id"],
                epoch=e,
                input="fixture\u2028only",
                target="",
                metadata={"fixture": True},
            )
            for r in rows
            for e in range(1, epochs + 1)
        ]
        native_path = phase_dir / "fixture.eval"
        native_path.write_text("Explicit Inspect-reader boundary fixture; no model output.\n")
        logs[native_path.name + phase] = SimpleNamespace(status="success", samples=samples)
        raw = reconcile.collection.raw_rows([logs[native_path.name + phase]])
        (phase_dir / "rollouts.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw)
        )
        audits[phase] = {
            "schema_version": "context_risk_highrate_native_audit_v1",
            "native_logs_sha256": {str(native_path): reconcile.archive.sha(native_path)},
        }
        launch = phase_dir / "launch_config.json"
        write_json(launch, {"config": {"review": str(source_review)}})
        write_json(phase_dir / "run_result.json", {"launch_config_path": str(launch)})
        write_json(
            phase_dir / "success_review.json",
            {
                "verdict": "PASS",
                "reviewer": "explicit-test-fixture",
                "native_logs_sha256": audits[phase]["native_logs_sha256"],
                "success_evidence": {},
            },
        )
        if phase == "fresh":
            capture_dir = root / "capture"
            capture_dir.mkdir()
            for chunk in range(6):
                (capture_dir / f"chunk_{chunk:04d}.rows.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows[chunk * 15 : (chunk + 1) * 15])
                )
    for name in (
        "selection.json",
        "capture_inputs.json",
        "setup/pre_fresh_input_applicability.json",
    ):
        write_json(root / name, {"fixture": True})
    review_path = root / "setup/reconcile_review.json"
    write_json(
        review_path,
        {
            "verdict": "PASS",
            "reviewer": "explicit-test-fixture",
            "sources_sha256": reconcile.source_hashes(),
        },
    )
    pod_source = tmp_path / "pod_only.log"
    pod_source.write_text("actual fixture bytes")
    inventory_path = tmp_path / "inventory.json"
    write_json(
        inventory_path,
        {
            "pod": reconcile.POD,
            "pod_id": reconcile.POD_ID,
            "source_root": reconcile.POD_ROOT,
            "all_owned_workers_drained": True,
            "files": {"server/owned.log": stamp(pod_source)},
        },
    )
    sources = {f"run/{p.relative_to(root)}": p for p in root.rglob("*") if p.is_file()}
    sources.update(
        {f"code/{name}": reconcile.design.PROJECT / name for name in reconcile.source_hashes()}
    )
    sources["external_design_review.json"] = source_review
    sources["pod/server/owned.log"] = pod_source
    remote = readback / reconcile.PREFIX
    files = {}
    for name, source in sources.items():
        destination = remote / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if name.endswith(".jsonl"):
            reconstructed = readback / "reconstructed" / name
            reconstructed.parent.mkdir(parents=True, exist_ok=True)
            reconstructed.write_bytes(source.read_bytes())
        files[name] = {"source": str(source), **stamp(source)}
    write_json(remote / "snapshot_manifest.json", {"phase": "raw", "files": files})
    identity = {
        "repo_id": reconcile.REPO,
        "prefix": reconcile.PREFIX,
        "revision": "a" * 40,
        "url": "https://example.invalid/explicit-test-fixture",
    }
    downloaded = {str(p.relative_to(readback)): stamp(p) for p in remote.rglob("*") if p.is_file()}
    write_json(root / "raw_upload_receipt.json", {"passed": True, **identity, "files": downloaded})
    write_json(
        root / "raw_readback_receipt.json",
        {
            "passed": True,
            "phase": "raw",
            **identity,
            "sources_sha256": reconcile.archive.source_hashes(),
            "upload_receipt_sha256": reconcile.archive.sha(root / "raw_upload_receipt.json"),
            "snapshot_sha256": reconcile.archive.sha(remote / "snapshot_manifest.json"),
            "downloaded_files": len(downloaded),
            "original_files": {
                name: {k: v[k] for k in ("size", "sha256")} for name, v in files.items()
            },
        },
    )

    def boundary(owner, name, side_effect):
        """Constrain unmodified external seam signatures while executing adapter bodies."""
        monkeypatch.setattr(
            owner, name, create_autospec(getattr(owner, name), side_effect=side_effect)
        )

    boundary(reconcile.postrun, "verify_report", lambda root, phase: audits[phase])
    boundary(
        reconcile.postrun,
        "validate_terminal_process",
        lambda root, phase: {"evidence_sha256": audits[phase]["native_logs_sha256"]},
    )
    boundary(reconcile.design, "load_phase", lambda root, phase: phase_inputs[phase])
    boundary(
        reconcile.collection,
        "load_samples",
        lambda path: [SimpleNamespace(id=r["sample_id"]) for r in reconcile.read_rows(path)],
    )
    boundary(reconcile.design, "success_evidence", lambda root, phase: {})
    boundary(
        reconcile,
        "read_eval_log",
        lambda path, **kwargs: logs[Path(path).name + Path(path).parent.name.removesuffix("_B")],
    )

    def capture_boundary(capture_root):
        """Inject a post-read mutation at the capture semantic boundary when requested."""
        targets = {
            "receipt": root / "raw_readback_receipt.json",
            "upload": root / "raw_upload_receipt.json",
            "manifest": remote / "snapshot_manifest.json",
            "inventory": inventory_path,
            "review": review_path,
            "pod_only": remote / "pod/server/owned.log",
            "external_review": source_review,
        }
        if mutation in targets:
            target = targets[mutation]
            target.write_bytes(target.read_bytes() + b" ")
        elif mutation == "extra_remote":
            (remote / "extra").write_text("not in the archive")
        return {"fixture": True}

    boundary(reconcile.capture, "validate_binding", capture_boundary)
    return root, readback, inventory_path, output, review_path


def test_full_run_body_with_disclosed_semantic_boundary_fixtures(tmp_path, monkeypatch):
    args = full_flow_fixture(tmp_path, monkeypatch)
    result = reconcile.run(*args)
    assert result["passed"] is True
    assert {name: value["rows"] for name, value in result["indices"].items()} == {
        "screen_B": 618,
        "fresh_B": 360,
        "capture": 90,
    }
    assert len(result["pod_files"]) == 1
    assert str(tmp_path / "external_design_review.json") in result["validated_input_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "receipt",
        "upload",
        "manifest",
        "inventory",
        "review",
        "pod_only",
        "external_review",
        "extra_remote",
    ],
)
def test_full_run_rejects_mid_validation_input_mutations(tmp_path, monkeypatch, mutation):
    args = full_flow_fixture(tmp_path, monkeypatch, mutation)
    with pytest.raises(ValueError):
        reconcile.run(*args)
    assert not (args[3] / "reconciliation.json").exists()


@pytest.mark.parametrize("inside", ["run", "readback"])
def test_output_cannot_mutate_input_trees(tmp_path, monkeypatch, inside):
    args = list(full_flow_fixture(tmp_path, monkeypatch))
    args[3] = args[0 if inside == "run" else 1] / "unsafe_new_output"
    with pytest.raises(ValueError, match="disjoint"):
        reconcile.run(*args)
    assert not args[3].exists()
