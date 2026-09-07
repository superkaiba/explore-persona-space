"""Bind all 90 selected high-rate initial prefixes to the frozen capture instrument."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import context_risk_followup_capture as previous

MODEL = previous.MODEL
CAPTURE = previous.CAPTURE
sha256 = previous.sha256
digest = previous.digest
write_json = previous.write_json
REPOSITORY = Path(__file__).resolve().parent.parent
SCHEMA = "context_risk_highrate_capture_v1"
POSITION = "decoder_block_44_output_at_last_unpadded_initial_prompt_token"
N_CONTEXTS = 90
INPUT_PATHS = {
    "manifest": "manifests/fresh_B.jsonl",
    "selection": "selection.json",
    "generation": "fresh_B/run_result.json",
    "prefix_tokens": "fresh_B/prefix_tokens.json",
    "terminal_process": "fresh_B/terminal_process.json",
}
RUNTIME = {
    "transformers": "5.15.0",
    "torch": "2.13.0+cu130",
    "torch_distribution": "2.13.0",
    "cuda": "13.0",
    "accelerate": "1.13.0",
    "numpy": "2.3.5",
}
SOURCES = tuple(
    dict.fromkeys(
        (
            *previous.SOURCES,
            "scripts/context_risk_highrate_capture.py",
            "configs/eval/context_risk_highrate_capture.yaml",
            "tests/test_context_risk_highrate_capture.py",
            "scripts/context_risk_highrate_design.py",
            "scripts/context_risk_highrate_collect.py",
        )
    )
)


def source_hashes() -> dict:
    from scripts import context_risk_highrate_design as design

    names = set(SOURCES) | set(design.source_hashes())
    return {name: sha256(REPOSITORY / name) for name in sorted(names)}


def imported_source_hashes() -> dict:
    result = previous.imported_source_hashes()
    for module_name in (
        "scripts.context_risk_followup_capture",
        "scripts.context_risk_highrate_design",
        "scripts.context_risk_highrate_collect",
    ):
        name = module_name.replace(".", "/") + ".py"
        module = importlib.import_module(module_name)
        actual = sha256(Path(module.__file__))
        if actual != sha256(REPOSITORY / name):
            raise ValueError(f"Imported helper differs from reviewed worktree bytes: {module_name}")
        result[name] = actual
    return result


def _jsonl(path: Path) -> list[dict]:
    # Iteration preserves raw U+2028/U+2029 inside JSON strings.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _runtime() -> dict:
    import torch

    runtime = {
        name: importlib.metadata.version(name)
        for name in ("transformers", "torch", "numpy", "accelerate")
    }
    runtime["torch_distribution"] = runtime["torch"]
    runtime["torch"] = torch.__version__
    runtime["cuda"] = torch.version.cuda
    if runtime != RUNTIME:
        raise ValueError("Capture package versions differ from the validated runtime")
    return runtime


def _inputs(root: Path) -> tuple[dict, list[dict], dict]:
    """Revalidate final native evidence, including permitted censors, without inference."""
    from scripts import context_risk_highrate_collect as collection
    from scripts import context_risk_highrate_design as design

    paths = {name: root / value for name, value in INPUT_PATHS.items()}
    hashes = {name: sha256(path) for name, path in paths.items()}
    manifest, selection, epochs = design.load_phase(root, "fresh")
    if Path(manifest).resolve() != paths["manifest"].resolve() or epochs != 4:
        raise ValueError("Capture requires the frozen four-repeat fresh B phase")
    if selection != json.loads(paths["selection"].read_text()):
        raise ValueError("Fresh phase selection differs from the selection receipt")
    generation = collection.verify_report(root, "fresh")
    terminal = design.validate_terminal_process(root, "fresh")
    evidence = {
        "phase": "fresh",
        "arm": "B",
        "epochs": 4,
        "n_contexts": N_CONTEXTS,
        "input_files_sha256": hashes,
        "generation_validation": generation,
        "terminal_process_validation": terminal,
    }
    rows, prefixes = _context_inputs(root, evidence)
    if hashes != {name: sha256(path) for name, path in paths.items()}:
        raise ValueError("Capture generation/selection inputs changed during validation")
    return evidence, rows, prefixes


def _context_inputs(root: Path, evidence: dict) -> tuple[list[dict], dict]:
    """Verify portable immutable files without accessing original VM paths or PIDs."""
    from scripts import context_risk_highrate_design as design

    paths = {name: root / value for name, value in INPUT_PATHS.items()}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if (
        evidence["phase"] != "fresh"
        or evidence["arm"] != "B"
        or evidence["epochs"] != 4
        or evidence["n_contexts"] != N_CONTEXTS
        or evidence["input_files_sha256"] != hashes
    ):
        raise ValueError("Staged capture input files or phase differ from the verified VM receipt")
    generation = evidence["generation_validation"]
    terminal = evidence["terminal_process_validation"]
    metadata = generation["metadata"]
    expected_metadata = {
        "schema_version": "context_risk_highrate_collection_v1",
        "phase": "fresh",
        "arm": "B",
        "epochs": 4,
        "max_attempts": 10,
        "message_limit": 22,
        "max_connections": 16,
        "model": design.MODEL,
        "manifest_sha256": hashes["manifest"],
        "selection_sha256": hashes["selection"],
        "phase_receipt_sha256": hashes["selection"],
        "sources_sha256": design.source_hashes(),
        "plan_sha256": sha256(REPOSITORY / "eval_results/context_risk_highrate_design/plan.md"),
    }
    if (
        any(metadata[key] != value for key, value in expected_metadata.items())
        or generation["sources_sha256"] != metadata["sources_sha256"]
        or generation["schema_version"] != "context_risk_highrate_native_audit_v1"
        or generation["coverage_complete"] is not True
        or generation["validation_issues"]
        or generation["duplicate_keys"]
        or generation["unexpected_keys"]
        or terminal != json.loads(paths["terminal_process"].read_text())
        or terminal["phase"] != "fresh"
        or terminal["run_result_sha256"] != hashes["generation"]
    ):
        raise ValueError("Staged native source/model/phase or terminal evidence differs")
    counts = generation["counts"]
    if (
        generation["verification_passed"] is not True
        or generation["run_result_sha256"] != hashes["generation"]
        or generation["prefix_tokens_sha256"] != hashes["prefix_tokens"]
        or counts["planned"] != 360
        or counts["realized"] != 360
        or counts["missing"] != 0
        or any(
            type(counts[key]) is not int or counts[key] < 0
            for key in ("success", "failure", "censored")
        )
        or counts["success"] + counts["failure"] + counts["censored"] != 360
        or terminal["verification_passed"] is not True
    ):
        raise ValueError("Capture requires verified complete fresh native and terminal evidence")
    rows = _jsonl(paths["manifest"])
    keys = [row["exact_context_sha256"] for row in rows]
    roster = {(row["task_id"], row["condition"]) for row in rows}
    tasks = {row["task_id"] for row in rows}
    selection = json.loads(paths["selection"].read_text())
    roles = selection["task_roles"]
    if (
        len(rows) != N_CONTEXTS
        or len(set(keys)) != N_CONTEXTS
        or len(tasks) != 30
        or roster
        != {
            (task, condition)
            for task in tasks
            for condition in ("original", "conflicting", "oneoff")
        }
        or any(digest(row["messages"]) != row["exact_context_sha256"] for row in rows)
        or selection["passed"] is not True
        or set(roles) != tasks
        or Counter(roles.values()) != {"probe_training": 20, "final_test": 10}
        or any(
            row["phase"] != "fresh"
            or row["prompt_variant"] != "B"
            or row["sample_id"] != f"highrate_fresh:B:{row['task_id']}:{row['condition']}"
            or row["public_test_role"] != roles[row["task_id"]]
            for row in rows
        )
    ):
        raise ValueError("Capture requires exactly 30 frozen tasks by three distinct contexts")
    contexts = generation["contexts"]
    if (
        len(contexts) != N_CONTEXTS
        or {(row["task_id"], row["condition"], row["exact_context_sha256"]) for row in contexts}
        != {(row["task_id"], row["condition"], row["exact_context_sha256"]) for row in rows}
        or any(
            row["planned"] != 4
            or row["realized"] != 4
            or row["missing"] != 0
            or any(
                type(row[key]) is not int or row[key] < 0
                for key in ("success", "failure", "censored")
            )
            or row["success"] + row["failure"] + row["censored"] != 4
            for row in contexts
        )
        or any(
            sum(row[key] for row in contexts) != counts[key]
            for key in ("success", "failure", "censored")
        )
    ):
        raise ValueError("Fresh generation census lacks the exact 90 four-repeat contexts")
    record = json.loads(paths["prefix_tokens"].read_text())
    prefixes = {row["exact_context_sha256"]: row for row in record["contexts"]}
    if (
        record["passed"] is not True
        or record["n_contexts"] != N_CONTEXTS
        or len(record["contexts"]) != N_CONTEXTS
        or set(prefixes) != set(keys)
    ):
        raise ValueError("Capture prefix record lacks unique complete generation coverage")
    for key, prefix in prefixes.items():
        ids = prefix["token_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or token < 0 for token in ids)
            or digest(ids) != prefix["prefix_token_ids_sha256"]
            or len(ids) != prefix["n_prefix_tokens"]
        ):
            raise ValueError(f"Saved generation token IDs disagree with their hash/length: {key}")
        if len(ids) > CAPTURE["max_sequence_tokens"]:
            raise ValueError("A frozen initial prefix exceeds the capture budget; never truncate")
    return rows, prefixes


def _review(cfg: DictConfig) -> tuple[dict, str]:
    path = Path(cfg.review)
    digest_before = sha256(path)
    review = json.loads(path.read_text())
    if review["verdict"] != "PASS" or review["sources_sha256"] != source_hashes():
        raise ValueError("Capture requires a current independent code review")
    if sha256(path) != digest_before:
        raise ValueError("Capture review changed while reading")
    return review, digest_before


def prepare(cfg: DictConfig) -> dict:
    """On the VM, freeze fully validated evidence for a separate GPU pod filesystem."""
    root = Path(cfg.root).resolve()
    review, review_hash = _review(cfg)
    sources = source_hashes()
    imports = imported_source_hashes()
    evidence, _, _ = _inputs(root)
    prepared = {
        "schema_version": "context_risk_highrate_capture_inputs_v1",
        "verification_passed": True,
        "model": MODEL,
        "capture": CAPTURE,
        "activation_position": POSITION,
        "sources_sha256": sources,
        "imported_sources_sha256": imports,
        "independent_review": review,
        "independent_review_sha256": review_hash,
        "fresh_evidence": evidence,
    }
    path = root / "capture_inputs.json"
    if path.exists() and json.loads(path.read_text()) != prepared:
        raise ValueError("Prepared capture inputs are immutable; source/evidence/review changed")
    if (
        sources != source_hashes()
        or imports != imported_source_hashes()
        or _review(cfg) != (review, review_hash)
    ):
        raise ValueError("Capture source/review changed during VM preparation")
    write_json(path, prepared)
    return {
        "verification_passed": True,
        "capture_inputs_sha256": sha256(path),
        "stage_relative_paths": [*INPUT_PATHS.values(), "capture_inputs.json"],
    }


def _staged_inputs(root: Path, expected_sha: str) -> tuple[dict, list[dict], dict]:
    path = root / "capture_inputs.json"
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or sha256(path) != expected_sha:
        raise ValueError("Staged capture requires the exact VM-prepared input receipt SHA256")
    prepared = json.loads(path.read_text())
    if (
        prepared["schema_version"] != "context_risk_highrate_capture_inputs_v1"
        or prepared["verification_passed"] is not True
        or prepared["model"] != MODEL
        or prepared["capture"] != CAPTURE
        or prepared["activation_position"] != POSITION
        or prepared["sources_sha256"] != source_hashes()
        or prepared["imported_sources_sha256"] != imported_source_hashes()
    ):
        raise ValueError("Prepared capture source/recipe differs from the reviewed instrument")
    review = prepared["independent_review"]
    if review["verdict"] != "PASS" or review["sources_sha256"] != prepared["sources_sha256"]:
        raise ValueError("Prepared capture lacks the current independent code review")
    rows, prefixes = _context_inputs(root, prepared["fresh_evidence"])
    if sha256(path) != expected_sha:
        raise ValueError("Prepared capture input receipt changed during validation")
    return prepared, rows, prefixes


def _fingerprint(manifest: Path, rows: list[dict]) -> str:
    return digest(
        {
            "schema_version": "context_risk_qwen38_impossible_capture_v2",
            "model": MODEL,
            "capture": CAPTURE,
            "manifest_sha256": sha256(manifest),
            "selected_contexts": [row["exact_context_sha256"] for row in rows],
        }
    )


def _chunks(
    out: Path, rows: list[dict], prefixes: dict, fingerprint: str, *, complete: bool
) -> tuple[list[dict], dict]:
    """Check ordered semantic payloads, including already finished resume shards."""
    import numpy as np

    expected_files = set()
    completed = []
    for index, start in enumerate(range(0, N_CONTEXTS, CAPTURE["checkpoint_rows"])):
        stem = f"chunk_{index:04d}"
        names = {f"{stem}{suffix}" for suffix in (".done.json", ".rows.jsonl", ".npz")}
        expected_files |= names
        done_path = out / f"{stem}.done.json"
        if not done_path.exists() and not complete:
            continue  # Interrupted incomplete payloads will be rewritten by the frozen helper.
        done = json.loads(done_path.read_text())
        metadata_path = out / f"{stem}.rows.jsonl"
        npz_path = out / f"{stem}.npz"
        metadata = _jsonl(metadata_path)
        originals = rows[start : start + CAPTURE["checkpoint_rows"]]
        if (
            done["schema_version"] != "context_risk_qwen38_impossible_capture_chunk_v2"
            or done["fingerprint"] != fingerprint
            or done["chunk_index"] != index
            or done["n_contexts"] != len(originals)
            or len(metadata) != len(originals)
            or done["npz_sha256"] != sha256(npz_path)
            or done["rows_sha256"] != sha256(metadata_path)
        ):
            raise ValueError("Capture chunk sentinel does not describe its exact payload")
        for captured, original in zip(metadata, originals, strict=True):
            if any(
                captured[key] != original[key]
                for key in ("exact_context_sha256", "task_id", "condition", "public_test_role")
            ):
                raise ValueError("Capture chunk row order/identity differs from frozen manifest")
            prefix = prefixes[original["exact_context_sha256"]]
            if any(
                captured[key] != prefix[key]
                for key in ("prefix_token_ids_sha256", "n_prefix_tokens")
            ):
                raise ValueError("Captured initial tokens differ from saved generation token IDs")
        with np.load(npz_path, allow_pickle=False) as arrays:
            values = arrays["activation"]
            if (
                set(arrays.files) != {"activation", "layers"}
                or values.shape != (len(originals), 1, MODEL["expected_hidden_dim"])
                or values.dtype != np.float16
                or not np.isfinite(values).all()
                or arrays["layers"].dtype != np.int16
                or not np.array_equal(arrays["layers"], [44])
            ):
                raise ValueError("Capture activation tensor shape/layer/type/values differ")
        completed.append(done)
    actual = {path.name for path in out.glob("chunk_*")}
    if actual - expected_files or (complete and actual != expected_files):
        raise ValueError("Capture chunk file roster differs from the six planned shards")
    if any(not (out / name).is_file() for name in actual):
        raise ValueError("Capture chunk roster contains a non-file")
    return completed, {name: sha256(out / name) for name in sorted(actual)}


def _validate(binding: dict, root: Path, evidence: dict, rows: list[dict], prefixes: dict) -> None:
    out = root / "capture"
    sources = source_hashes()
    if binding["passed"] is not True or binding["schema_version"] != SCHEMA:
        raise ValueError("Missing or incompatible highrate capture binding")
    if binding["model"] != MODEL or binding["capture"] != CAPTURE:
        raise ValueError("Capture recipe differs from the approved initial-prefix instrument")
    if binding["activation_position"] != POSITION or binding["runtime"] != RUNTIME:
        raise ValueError("Capture position/runtime differs from the frozen instrument")
    if binding["sources_sha256"] != sources:
        raise ValueError("Capture source closure differs from reviewed source bytes")
    if binding["imported_sources_sha256"] != imported_source_hashes():
        raise ValueError("Actual imported capture helpers differ from reviewed sources")
    review = binding["independent_review"]
    if review["verdict"] != "PASS" or review["sources_sha256"] != sources:
        raise ValueError("Independent capture source review is missing or stale")
    if binding["fresh_evidence"] != evidence:
        raise ValueError("Capture inputs differ from current native generation/selection evidence")
    prepared, _, _ = _staged_inputs(root, binding["capture_inputs_sha256"])
    if (
        prepared["fresh_evidence"] != evidence
        or prepared["independent_review"] != review
        or prepared["independent_review_sha256"] != binding["independent_review_sha256"]
    ):
        raise ValueError("Capture binding differs from the verified VM preparation")
    launch = {
        key: value
        for key, value in binding.items()
        if key
        not in {
            "passed",
            "schema_version",
            "fingerprint",
            "run_result_sha256",
            "chunk_files_sha256",
        }
    }
    if json.loads((out / "capture_launch_binding.json").read_text()) != launch:
        raise ValueError("Capture launch binding changed")
    fingerprint = _fingerprint(root / "manifests/fresh_B.jsonl", rows)
    chunks, hashes = _chunks(out, rows, prefixes, fingerprint, complete=True)
    report_path = out / "run_result.json"
    report = json.loads(report_path.read_text())
    if (
        binding["run_result_sha256"] != sha256(report_path)
        or binding["chunk_files_sha256"] != hashes
        or binding["fingerprint"] != fingerprint
        or report["fingerprint"] != fingerprint
        or report["schema_version"] != "context_risk_qwen38_impossible_capture_run_v2"
        or report["passed"] is not True
        or report["n_contexts"] != N_CONTEXTS
        or report["capture_layers"] != [44]
        or report["prefixes_truncated"] != 0
        or report["max_sequence_tokens"] != CAPTURE["max_sequence_tokens"]
        or report["minimum_prefix_tokens"] != min(p["n_prefix_tokens"] for p in prefixes.values())
        or report["maximum_prefix_tokens"] != max(p["n_prefix_tokens"] for p in prefixes.values())
        or report["model_id"] != MODEL["id"]
        or report["model_revision"] != MODEL["revision"]
        or report["chunks"] != chunks
    ):
        raise ValueError(
            "Capture report/hash/model/coverage differs from the exact frozen instrument"
        )


def validate_binding(
    capture_root: Path,
    manifest: Path | None = None,
    prefix_path: Path | None = None,
    selection_path: Path | None = None,
) -> dict:
    """Revalidate source, native completion, selection and tensors for every consumer."""
    capture_root = Path(capture_root).resolve()
    root = capture_root.parent
    if capture_root.name != "capture":
        raise ValueError("Highrate capture must occupy the frozen root/capture path")
    for actual, expected in (
        (manifest, root / "manifests/fresh_B.jsonl"),
        (prefix_path, root / "fresh_B/prefix_tokens.json"),
        (selection_path, root / "selection.json"),
    ):
        if actual is not None and Path(actual).resolve() != expected.resolve():
            raise ValueError("Capture consumer supplied a different frozen input path")
    evidence, rows, prefixes = _inputs(root)
    binding_path = capture_root / "capture_binding.json"
    original_hash = sha256(binding_path)
    binding = json.loads(binding_path.read_text())
    _validate(binding, root, evidence, rows, prefixes)
    if original_hash != sha256(binding_path) or evidence != _inputs(root)[0]:
        raise ValueError("Capture inputs changed during downstream validation")
    return binding


def run(cfg: DictConfig) -> dict:
    """Capture a complete fresh cohort; verified censors do not suppress prefix capture."""
    from scripts.context_risk_qwen38_impossible_capture import run_capture

    root = Path(cfg.root).resolve()
    out = Path(cfg.output_dir).resolve()
    if (
        out != root / "capture"
        or Path(cfg.manifest_path).resolve() != root / "manifests/fresh_B.jsonl"
    ):
        raise ValueError("Highrate capture requires the fixed fresh manifest/output paths")
    if (
        OmegaConf.to_container(cfg.model, resolve=True) != MODEL
        or OmegaConf.to_container(cfg.capture, resolve=True) != CAPTURE
    ):
        raise ValueError("Capture settings differ from the fixed production recipe")
    sources = source_hashes()
    review, review_hash = _review(cfg)
    imports = imported_source_hashes()
    prepared, rows, prefixes = _staged_inputs(root, cfg.input_binding_sha256)
    if (
        prepared["independent_review"] != review
        or prepared["independent_review_sha256"] != review_hash
    ):
        raise ValueError("Pod capture review differs from VM preparation")
    evidence = prepared["fresh_evidence"]
    launch = {
        "model": MODEL,
        "capture": CAPTURE,
        "runtime": _runtime(),
        "activation_position": POSITION,
        "sources_sha256": sources,
        "imported_sources_sha256": imports,
        "independent_review": review,
        "independent_review_sha256": review_hash,
        "capture_inputs_sha256": cfg.input_binding_sha256,
        "fresh_evidence": evidence,
    }
    out.mkdir(parents=True, exist_ok=True)
    launch_path = out / "capture_launch_binding.json"
    if not launch_path.exists() and any(out.iterdir()):
        raise ValueError(
            "First capture launch requires empty output; cannot adopt unbound old chunks"
        )
    if launch_path.exists() and json.loads(launch_path.read_text()) != launch:
        raise ValueError("Capture resume source/input/runtime regime changed")
    binding_path = out / "capture_binding.json"
    if binding_path.exists():
        # The pod verifies portable evidence; VM consumption additionally checks native logs/PIDs.
        binding = json.loads(binding_path.read_text())
        _validate(binding, root, evidence, rows, prefixes)
        return binding
    _chunks(out, rows, prefixes, _fingerprint(Path(cfg.manifest_path), rows), complete=False)
    write_json(launch_path, launch)
    report = run_capture(cfg)
    if (
        sources != source_hashes()
        or imports != imported_source_hashes()
        or _review(cfg) != (review, review_hash)
        or prepared != _staged_inputs(root, cfg.input_binding_sha256)[0]
    ):
        raise ValueError("Capture source/review/generation inputs changed during the run")
    binding = {
        **launch,
        "passed": True,
        "schema_version": SCHEMA,
        "fingerprint": report["fingerprint"],
        "run_result_sha256": sha256(out / "run_result.json"),
        "chunk_files_sha256": {
            path.name: sha256(path) for path in sorted(out.glob("chunk_*")) if path.is_file()
        },
    }
    _validate(binding, root, evidence, rows, prefixes)
    write_json(binding_path, binding)
    return binding


@hydra.main(
    version_base="1.3", config_path="../configs/eval", config_name="context_risk_highrate_capture"
)
def main(cfg: DictConfig) -> None:
    if cfg.mode not in {"prepare", "capture"}:
        raise ValueError("Highrate capture mode must be prepare or capture")
    print(json.dumps(prepare(cfg) if cfg.mode == "prepare" else run(cfg), indent=2), flush=True)


if __name__ == "__main__":
    main()
