"""Reconcile archived highrate observations and every owned pod output by exact identity.

This archive-only adapter cannot collect data, change task state, or terminate
compute. Its ASCII indexes are projections of byte-verified remote readback, not
substitutes for the raw generations. Run the managed teardown gate separately.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

# PROD_IMPORT_LINT_EXEMPT: Runtime pinned with inspect-ai==0.3.261.
from inspect_ai.log import read_eval_log  # noqa: E402

from scripts import context_risk_highrate_archive as archive  # noqa: E402
from scripts import context_risk_highrate_capture as capture  # noqa: E402
from scripts import context_risk_highrate_collect as collection  # noqa: E402
from scripts import context_risk_highrate_design as design  # noqa: E402
from scripts import verify_uploads  # noqa: E402

POD = "pod-2670-highrate"
POD_ID = "1bds7vqkrluxkc"
POD_ROOT = "/workspace/logs/issue2670-context-risk-highrate"
PREFIX = "context_risk/issue2670_highrate/raw"
REPO = "superkaiba1/explore-persona-space-data"


def source_hashes() -> dict:
    """Bind the unchanged validators, adapter and tests for independent review."""
    for module, name in (
        (archive, "scripts/context_risk_highrate_archive.py"),
        (verify_uploads, "scripts/verify_uploads.py"),
    ):
        if archive.sha(Path(module.__file__)) != archive.sha(design.PROJECT / name):
            raise ValueError(f"Imported reconciliation helper differs from reviewed source: {name}")
    return {
        **capture.source_hashes(),
        **archive.source_hashes(),
        **{
            name: archive.sha(design.PROJECT / name)
            for name in (
                "scripts/context_risk_highrate_reconcile.py",
                "scripts/verify_uploads.py",
                "tests/test_context_risk_highrate_reconcile.py",
            )
        },
    }


def read_json(path: Path) -> dict:
    """Read one required JSON document without substituting missing evidence."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    """Split records only at physical LF boundaries, preserving Unicode inside strings."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def safe_child(root: Path, name: str) -> Path:
    """Resolve a nonempty canonical relative file path without symlink traversal."""
    relative = Path(name)
    if (
        not name
        or not relative.parts
        or relative.is_absolute()
        or relative.as_posix() != name
        or ".." in relative.parts
    ):
        raise ValueError(f"Noncanonical archive path: {name}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or any(
        part.is_symlink() for part in [path, *path.parents]
    ):
        raise ValueError(f"Archive path escapes or follows a symlink: {name}")
    return path


def check_bytes(path: Path, expected: dict) -> None:
    """Require an actual regular file with the declared size and full SHA256."""
    if (
        type(expected["size"]) is not int
        or expected["size"] < 0
        or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != expected["size"]
        or archive.sha(path) != expected["sha256"]
    ):
        raise ValueError(f"Archived bytes differ: {path}")


def open_readback(root: Path, readback: Path) -> tuple[dict, dict, dict[str, Path]]:
    """Recheck the pinned upload, its complete downloaded set and all reconstructed originals."""
    receipt = read_json(root / "raw_readback_receipt.json")
    upload_path = root / "raw_upload_receipt.json"
    uploaded = read_json(upload_path)
    if (
        receipt["passed"] is not True
        or uploaded["passed"] is not True
        or receipt["phase"] != "raw"
        or receipt["sources_sha256"] != archive.source_hashes()
        or receipt["upload_receipt_sha256"] != archive.sha(upload_path)
        or receipt["repo_id"] != REPO
        or receipt["prefix"] != PREFIX
        or re.fullmatch(r"[0-9a-f]{40}", receipt["revision"]) is None
        or any(receipt[k] != uploaded[k] for k in ("repo_id", "prefix", "revision", "url"))
    ):
        raise ValueError("Raw upload/readback provenance differs from the reviewed archive")
    remote_root = safe_child(readback, PREFIX)
    actual = {str(p.relative_to(readback)) for p in remote_root.rglob("*") if p.is_file()}
    if not actual or actual != set(uploaded["files"]) or len(actual) != receipt["downloaded_files"]:
        raise ValueError("Pinned downloaded filename set differs")
    for name, expected in uploaded["files"].items():
        check_bytes(safe_child(readback, name), expected)
    manifest_path = remote_root / "snapshot_manifest.json"
    if archive.sha(manifest_path) != receipt["snapshot_sha256"]:
        raise ValueError("Archived snapshot manifest differs")
    manifest = read_json(manifest_path)
    original = manifest["files"]
    if (
        not original
        or manifest["phase"] != "raw"
        or set(original) != set(receipt["original_files"])
    ):
        raise ValueError("Archived original filename set or phase differs")
    paths = {}
    for index, (name, expected) in enumerate(original.items(), 1):
        base = readback / "reconstructed" if name.endswith(".jsonl") else remote_root
        path = safe_child(base, name)
        check_bytes(path, expected)
        if any(expected[k] != receipt["original_files"][name][k] for k in ("sha256", "size")):
            raise ValueError(f"Original receipt differs from snapshot: {name}")
        paths[name] = path
        print(f"[reconcile-bytes] unit {index}/{len(original)} {name}", flush=True)
    return receipt, manifest, paths


def bridge(path: Path, manifest: dict, originals: dict[str, Path]) -> Path:
    """Bind a validator's actual local input to its explicit archived source mapping."""
    matches = [
        name
        for name, record in manifest["files"].items()
        if Path(record["source"]).resolve() == path.resolve()
    ]
    if not matches:
        raise ValueError(f"Validated input is absent from the archive source mapping: {path}")
    for name in matches:
        check_bytes(path, manifest["files"][name])
        check_bytes(originals[name], manifest["files"][name])
    return originals[sorted(matches)[0]]


def exact_keys(actual: list[tuple], expected: set[tuple], label: str) -> None:
    """Reject missing, surplus, duplicate and same-count substituted logical rows."""
    if not expected or len(actual) != len(set(actual)) or set(actual) != expected:
        missing, surplus = expected - set(actual), set(actual) - expected
        raise ValueError(
            f"{label} full-key mismatch: rows={len(actual)} unique={len(set(actual))} "
            f"expected={len(expected)} missing={len(missing)} surplus={len(surplus)}"
        )


def generation_keys(rows: list[dict]) -> list[tuple[str, int]]:
    """Require literal string/integer rollout identities, rejecting bool-as-integer aliases."""
    keys = []
    for row in rows:
        if (
            not isinstance(row["sample_id"], str)
            or not row["sample_id"]
            or type(row["epoch"]) is not int
            or not 1 <= row["epoch"] <= 4
        ):
            raise ValueError("Malformed generation identity")
        keys.append((row["sample_id"], row["epoch"]))
    return keys


def write_index(output: Path, label: str, keys: list[tuple]) -> dict:
    """Write only already reconciled complete logical identities, using ASCII JSON."""
    path = output / label / "row_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii") as stream:
        for key in sorted(keys):
            stream.write(json.dumps({"key": key}, ensure_ascii=True) + "\n")
    return {"path": str(path), "sha256": archive.sha(path), "rows": len(keys)}


def pod_coverage(inventory: dict, manifest: dict, originals: dict[str, Path]) -> dict:
    """Match the complete final pod inventory by relative path and bytes, including logs."""
    if (
        inventory["pod"] != POD
        or inventory["pod_id"] != POD_ID
        or inventory["source_root"] != POD_ROOT
        or not inventory["files"]
        or inventory["all_owned_workers_drained"] is not True
    ):
        raise ValueError("A nonempty final owned-pod inventory with drained workers is required")
    expected_names = {f"pod/{name}" for name in inventory["files"]}
    archived_names = {name for name in originals if name.startswith("pod/")}
    if expected_names != archived_names:
        raise ValueError("Complete pod-relative original filename set differs from raw archive")
    covered = {}
    for relative, record in inventory["files"].items():
        safe_child(Path(POD_ROOT), relative)
        name = f"pod/{relative}"
        if any(record[k] != manifest["files"][name][k] for k in ("size", "sha256")):
            raise ValueError(f"Pod and archived original bytes differ: {relative}")
        check_bytes(originals[name], record)
        covered[relative] = {"archive_original": name, **record}
    return covered


def run(root: Path, readback: Path, inventory_path: Path, output: Path, review_path: Path) -> dict:
    """Produce real archived-row and per-file coverage evidence without authorizing teardown."""
    if root.resolve() != archive.RUN_ROOT.resolve() or output.exists():
        raise ValueError("Use the approved run root and a fresh reconciliation output directory")
    roots = [p.resolve() for p in (root, readback, output)]
    if any(a.is_relative_to(b) for i, a in enumerate(roots) for j, b in enumerate(roots) if i != j):
        raise ValueError("Run, pinned readback and reconciliation output must be disjoint")
    for directory in (root, readback, output):
        if any(p.is_symlink() for p in [directory, *directory.parents]):
            raise ValueError("Reconciliation roots cannot follow symlinks")
    control_paths = [
        root / "raw_readback_receipt.json",
        root / "raw_upload_receipt.json",
        readback / PREFIX / "snapshot_manifest.json",
        inventory_path,
        review_path,
    ]
    control_hashes = {str(p): archive.sha(p) for p in control_paths}
    sources = source_hashes()
    reviewed = read_json(review_path)
    if (
        reviewed.get("verdict") != "PASS"
        or not reviewed.get("reviewer")
        or reviewed.get("sources_sha256") != sources
    ):
        raise ValueError("Reconciliation requires an independent PASS on current source bytes")
    receipt, manifest, originals = open_readback(root, readback)
    inventory = read_json(inventory_path)
    inventory_sha = control_hashes[str(inventory_path)]
    pod_files = pod_coverage(inventory, manifest, originals)
    bound = set()

    def consume(path: Path) -> Path:
        """Track every native validator input bridged to archived bytes."""
        bound.add(path.resolve())
        return bridge(path, manifest, originals)

    for name in sources:
        consume(design.PROJECT / name)
    consume(review_path)
    # These trees are immutable once collection and capture finish. Mutable owner
    # status and late upload receipts are outside this input set.
    for directory in ("manifests", "screen_B", "fresh_B", "capture"):
        files = [p for p in (root / directory).rglob("*") if p.is_file()]
        if not files:
            raise ValueError(f"Required validated artifact tree is empty: {directory}")
        for path in files:
            consume(path)
    for name in (
        "selection.json",
        "capture_inputs.json",
        "setup/pre_fresh_input_applicability.json",
    ):
        consume(root / name)
    indices, validations = {}, {}
    for phase, n in (("screen", 618), ("fresh", 360)):
        report = read_json(consume(root / f"{phase}_B/run_result.json"))
        launch = read_json(consume(Path(report["launch_config_path"])))
        consume(Path(launch["config"]["review"]))
        audit = collection.verify_report(root, phase)
        terminal = design.validate_terminal_process(root, phase)
        validations[phase] = {"native": audit, "process": terminal}
        success_review = read_json(consume(root / f"{phase}_B/success_review.json"))
        if (
            success_review.get("verdict") != "PASS"
            or not success_review.get("reviewer")
            or success_review["native_logs_sha256"] != audit["native_logs_sha256"]
            or success_review["success_evidence"] != design.success_evidence(root, phase)
        ):
            raise ValueError(f"{phase} lacks exact independent review of every successful body")
        for name in terminal["evidence_sha256"]:
            consume(Path(name))
        manifest_path, _, epochs = design.load_phase(root, phase)
        samples = collection.load_samples(manifest_path)
        expected = collection.expected_keys(samples, epochs, pilot=False)
        if len(expected) != n:
            raise ValueError("Input-derived expected rollout count differs from the approved panel")
        archived_rows = read_rows(consume(root / f"{phase}_B/rollouts.jsonl"))
        keys = generation_keys(archived_rows)
        exact_keys(keys, expected, f"{phase} archived rollouts")
        native = [
            read_eval_log(str(consume(Path(path))), resolve_attachments="full")
            for path in audit["native_logs_sha256"]
        ]
        native_keys = [(str(s.id), s.epoch) for log in native for s in log.samples or []]
        exact_keys(native_keys, expected, f"{phase} archived native")
        if (
            any(log.status != "success" for log in native)
            or collection.raw_rows(native) != archived_rows
        ):
            raise ValueError("Archived native bodies differ from complete archived raw rows")
        indices[f"{phase}_B"] = write_index(output, f"{phase}_B", keys)
    validations["capture"] = capture.validate_binding(root / "capture")
    fresh_manifest, _, _ = design.load_phase(root, "fresh")
    expected_capture = {
        (r["task_id"], r["condition"], r["exact_context_sha256"])
        for r in read_rows(consume(fresh_manifest))
    }
    captured = [
        row
        for path in sorted((root / "capture").glob("chunk_*.rows.jsonl"))
        for row in read_rows(consume(path))
    ]
    keys = [(r["task_id"], r["condition"], r["exact_context_sha256"]) for r in captured]
    if len(expected_capture) != 90:
        raise ValueError("Expected capture roster differs from90 exact initial contexts")
    exact_keys(keys, expected_capture, "archived capture")
    indices["capture"] = write_index(output, "capture", keys)
    mechanical = verify_uploads.check_realized_row_counts(
        expected_rows={"screen_B": 618, "fresh_B": 360, "capture": 90},
        local_root=str(output),
        glob_pattern="row_index*.jsonl",
        distinct_key_fields=("key",),
    )
    if mechanical["status"] != "OK":
        raise ValueError(f"Mechanical full-key count gate failed: {mechanical}")
    for path in bound:
        bridge(path, manifest, originals)
    if (
        archive.sha(inventory_path) != inventory_sha
        or source_hashes() != sources
        or read_json(review_path) != reviewed
    ):
        raise ValueError(
            "Final inventory, source or independent review changed during reconciliation"
        )
    if {str(p): archive.sha(p) for p in control_paths} != control_hashes:
        raise ValueError("Archive control evidence changed during reconciliation")
    repeated_receipt, repeated_manifest, repeated_paths = open_readback(root, readback)
    if (repeated_receipt, repeated_manifest, repeated_paths) != (receipt, manifest, originals):
        raise ValueError("Pinned readback changed during semantic validation")
    pod_coverage(inventory, repeated_manifest, repeated_paths)
    result = {
        "passed": True,
        "phase": "raw-row-reconciliation",
        "created_utc": datetime.now(UTC).isoformat(),
        "repo_id": receipt["repo_id"],
        "revision": receipt["revision"],
        "prefix": receipt["prefix"],
        "raw_readback_receipt_sha256": control_hashes[str(root / "raw_readback_receipt.json")],
        "archive_control_sha256": control_hashes,
        "snapshot_sha256": receipt["snapshot_sha256"],
        "inventory_sha256": inventory_sha,
        "indices": indices,
        "mechanical_row_check": mechanical,
        "pod_files": pod_files,
        "validated_input_sha256": {str(p): archive.sha(p) for p in sorted(bound)},
        "validation_sha256": design._stable_digest(validations),
        "sources_sha256": sources,
        "independent_review_sha256": archive.sha(review_path),
        "imported_capture_source_sha256": capture.imported_source_hashes(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("inspect-ai", "huggingface-hub", "numpy")
        },
        "scope": "Exact archived618screen/360fresh/90capture full-key sets and complete pod-relative file hashes. ASCII indexes derive only from pinned remote readback. Generic basename/sharded-name residue checks remain supplementary; no exemptions or discards were used. This is not a teardown authorization; the owner must verify no subsequent pod writes and use the managed gate.",
    }
    (output / "reconciliation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--pod-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.root, args.readback, args.pod_inventory, args.output, args.review), indent=2
        )
    )
