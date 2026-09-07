"""Bind the validated initial-prefix capture to the selected fresh experiment."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

MODEL = {
    "id": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "transformers_version": "5.15.0",
    "dtype": "bfloat16",
    "expected_layers": 64,
    "expected_hidden_dim": 5120,
    "capture_layers": [44],
}
CAPTURE = {
    "max_contexts": None,
    "checkpoint_rows": 15,
    "max_sequence_tokens": 32768,
    "batch_max_rows": 2,
    "batch_max_tokens": 16384,
    "enable_thinking": False,
}
SOURCES = (
    "scripts/context_risk_followup_capture.py",
    "configs/eval/context_risk_followup_capture.yaml",
    "scripts/context_risk_qwen38_impossible_capture.py",
    "scripts/context_risk_qwen38_capture.py",
    "scripts/context_risk_qwen38_smoke.py",
    "scripts/context_risk_prepare_data.py",
    "src/explore_persona_space/analysis/extraction.py",
    "configs/eval/context_risk_qwen38_impossible_capture.yaml",
    "configs/eval/context_risk_qwen38_smoke.yaml",
)
REPOSITORY = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hashes() -> dict:
    return {name: sha256(REPOSITORY / name) for name in SOURCES}


def imported_source_hashes() -> dict:
    """Check actual loaded helper files, including shared editable-package resolution."""
    modules = {
        "scripts/context_risk_qwen38_impossible_capture.py": "scripts.context_risk_qwen38_impossible_capture",
        "scripts/context_risk_qwen38_capture.py": "scripts.context_risk_qwen38_capture",
        "scripts/context_risk_qwen38_smoke.py": "scripts.context_risk_qwen38_smoke",
        "scripts/context_risk_prepare_data.py": "scripts.context_risk_prepare_data",
        "src/explore_persona_space/analysis/extraction.py": "explore_persona_space.analysis.extraction",
    }
    result = {}
    for name, module_name in modules.items():
        module = importlib.import_module(module_name)
        actual = sha256(Path(module.__file__))
        if actual != sha256(REPOSITORY / name):
            raise ValueError(f"Imported helper differs from reviewed worktree bytes: {module_name}")
        result[name] = actual
    return result


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_binding(
    capture_root: Path, manifest: Path, prefix_path: Path, selection_path: Path
) -> dict:
    """Validate the captured recipe and exact files at every downstream consumption."""
    binding = json.loads((capture_root / "capture_binding.json").read_text())
    if not binding["passed"] or binding["schema_version"] != "context_risk_followup_capture_v1":
        raise ValueError("Missing or incompatible capture binding")
    if binding["model"] != MODEL or binding["capture"] != CAPTURE:
        raise ValueError("Capture recipe differs from the approved initial-prefix instrument")
    if (
        binding["activation_position"]
        != "decoder_block_44_output_at_last_unpadded_initial_prompt_token"
    ):
        raise ValueError("Capture position differs from the frozen map convention")
    if binding["sources_sha256"] != source_hashes():
        raise ValueError("Capture source closure differs from reviewed source bytes")
    expected_imports = {
        key: binding["sources_sha256"][key]
        for key in (
            "scripts/context_risk_qwen38_impossible_capture.py",
            "scripts/context_risk_qwen38_capture.py",
            "scripts/context_risk_qwen38_smoke.py",
            "scripts/context_risk_prepare_data.py",
            "src/explore_persona_space/analysis/extraction.py",
        )
    }
    if binding["imported_sources_sha256"] != expected_imports:
        raise ValueError("Actual imported capture helpers were not the reviewed sources")
    review = binding["independent_review"]
    if review["verdict"] != "PASS" or review["sources_sha256"] != binding["sources_sha256"]:
        raise ValueError("Independent capture source review is missing or stale")
    if (
        binding["manifest_sha256"] != sha256(manifest)
        or binding["prefix_tokens_sha256"] != sha256(prefix_path)
        or binding["selection_sha256"] != sha256(selection_path)
    ):
        raise ValueError("Capture inputs differ from the generation/selection evidence")
    runtime_pins = {
        "transformers": "5.15.0",
        "torch": "2.13.0+cu130",
        "torch_distribution": "2.13.0",
        "cuda": "13.0",
        "accelerate": "1.13.0",
        "numpy": "2.3.5",
    }
    if any(binding["runtime"][key] != value for key, value in runtime_pins.items()):
        raise ValueError("Capture runtime differs from the validated model regime")
    report = json.loads((capture_root / "run_result.json").read_text())
    if sha256(capture_root / "run_result.json") != binding["run_result_sha256"]:
        raise ValueError("Capture run report changed")
    with manifest.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 249 or len({r["exact_context_sha256"] for r in rows}) != 249:
        raise ValueError("Capture binding requires exactly 249 distinct frozen contexts")
    expected = digest(
        {
            "schema_version": "context_risk_qwen38_impossible_capture_v2",
            "model": MODEL,
            "capture": CAPTURE,
            "manifest_sha256": sha256(manifest),
            "selected_contexts": [r["exact_context_sha256"] for r in rows],
        }
    )
    if report["fingerprint"] != expected or binding["fingerprint"] != expected:
        raise ValueError("Capture fingerprint does not describe the reviewed recipe and manifest")
    expected_files = {p.name for p in capture_root.glob("chunk_*") if p.is_file()}
    if set(binding["chunk_files_sha256"]) != expected_files:
        raise ValueError("Capture chunk file roster changed")
    for name, expected_sha in binding["chunk_files_sha256"].items():
        if sha256(capture_root / name) != expected_sha:
            raise ValueError(f"Capture chunk changed: {name}")
    import numpy as np

    if (
        not report["passed"]
        or report["n_contexts"] != 249
        or report["capture_layers"] != [44]
        or report["prefixes_truncated"] != 0
        or report["model_id"] != MODEL["id"]
        or report["model_revision"] != MODEL["revision"]
    ):
        raise ValueError("Capture result differs from the reviewed model/coverage")
    completed = []
    realized_rows = []
    for chunk_index, start in enumerate(range(0, len(rows), CAPTURE["checkpoint_rows"])):
        stem = f"chunk_{chunk_index:04d}"
        done = json.loads((capture_root / f"{stem}.done.json").read_text())
        chunk_rows_path = capture_root / f"{stem}.rows.jsonl"
        npz_path = capture_root / f"{stem}.npz"
        with chunk_rows_path.open() as handle:
            chunk_rows = [json.loads(line) for line in handle if line.strip()]
        expected_rows = rows[start : start + CAPTURE["checkpoint_rows"]]
        if (
            done["fingerprint"] != expected
            or done["chunk_index"] != chunk_index
            or done["n_contexts"] != len(expected_rows)
            or len(chunk_rows) != len(expected_rows)
            or done["npz_sha256"] != sha256(npz_path)
            or done["rows_sha256"] != sha256(chunk_rows_path)
        ):
            raise ValueError("Capture chunk sentinel does not describe its exact payload")
        for captured, original in zip(chunk_rows, expected_rows, strict=True):
            if any(
                captured[key] != original[key]
                for key in ("exact_context_sha256", "task_id", "condition", "public_test_role")
            ):
                raise ValueError("Capture chunk row order/identity differs from frozen manifest")
        with np.load(npz_path) as arrays:
            values = arrays["activation"]
            if (
                values.shape != (len(expected_rows), 1, 5120)
                or values.dtype != np.float16
                or not np.isfinite(values).all()
                or not np.array_equal(arrays["layers"], [44])
            ):
                raise ValueError("Capture activation tensor shape/layer/type/values differ")
        completed.append(done)
        realized_rows.extend(chunk_rows)
    if completed != report["chunks"] or len(expected_files) != 3 * len(completed):
        raise ValueError("Capture run/chunk coverage differs")
    prefix_record = json.loads(prefix_path.read_text())
    prefixes = {r["exact_context_sha256"]: r for r in prefix_record["contexts"]}
    if not prefix_record["passed"] or len(prefix_record["contexts"]) != 249 or len(prefixes) != 249:
        raise ValueError("Capture prefix record lacks unique complete coverage")
    for row in realized_rows:
        prefix = prefixes[row["exact_context_sha256"]]
        if (
            digest(prefix["token_ids"]) != prefix["prefix_token_ids_sha256"]
            or len(prefix["token_ids"]) != prefix["n_prefix_tokens"]
            or any(
                row[key] != prefix[key] for key in ("prefix_token_ids_sha256", "n_prefix_tokens")
            )
        ):
            raise ValueError("Capture initial tokens differ from saved generation token IDs")
    return binding


def run(cfg: DictConfig) -> dict:
    """Execute the reused GPU capture only for complete selected fresh generation."""
    from scripts.context_risk_qwen38_impossible_capture import run_capture

    root = Path(cfg.root)
    out = Path(cfg.output_dir)
    manifest = Path(cfg.manifest_path)
    review = json.loads(Path(cfg.review).read_text())
    sources = source_hashes()
    if review["verdict"] != "PASS" or review["sources_sha256"] != sources:
        raise ValueError("Capture requires a current independent code review")
    if (
        OmegaConf.to_container(cfg.model, resolve=True) != MODEL
        or OmegaConf.to_container(cfg.capture, resolve=True) != CAPTURE
    ):
        raise ValueError("Capture settings differ from the fixed production recipe")
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["passed"] or selection["selected_arm"] != cfg.arm:
        raise ValueError("Capture arm differs from viable frozen selection")
    freeze = json.loads((root / "manifests/freeze.json").read_text())
    if selection["freeze_sha256"] != sha256(root / "manifests/freeze.json"):
        raise ValueError("Selection/freeze mismatch")
    if manifest.resolve() != (root / "manifests" / f"fresh_{cfg.arm}.jsonl").resolve():
        raise ValueError("Capture manifest must be the selected fresh cohort")
    if sha256(manifest) != freeze["manifests"][manifest.name]["sha256"]:
        raise ValueError("Capture manifest changed after freeze")
    fresh = root / f"fresh_{cfg.arm}"
    generation = json.loads((fresh / "run_result.json").read_text())
    if (
        not generation["passed"]
        or generation["realized_rollouts"] != 996
        or generation["technical_errors"] != 0
        or generation["is_pilot"]
        or generation["phase"] != "fresh"
        or generation["arm"] != cfg.arm
        or generation["manifest_sha256"] != sha256(manifest)
    ):
        raise ValueError("Capture requires complete selected fresh generation")
    prefix_path = fresh / "prefix_tokens.json"
    prefix_record = json.loads(prefix_path.read_text())
    with manifest.open() as handle:
        manifest_rows = [json.loads(line) for line in handle if line.strip()]
    prefixes = {r["exact_context_sha256"]: r for r in prefix_record["contexts"]}
    if (
        not prefix_record["passed"]
        or len(prefix_record["contexts"]) != 249
        or len(prefixes) != 249
        or set(prefixes) != {r["exact_context_sha256"] for r in manifest_rows}
    ):
        raise ValueError("Capture requires complete unique generation-prefix evidence")
    if max(r["n_prefix_tokens"] for r in prefixes.values()) > CAPTURE["max_sequence_tokens"]:
        raise ValueError("A frozen initial prefix exceeds the capture budget; never truncate")
    runtime = {
        name: importlib.metadata.version(name)
        for name in ("transformers", "torch", "numpy", "accelerate")
    }
    import torch

    runtime["torch_distribution"] = runtime["torch"]
    runtime["torch"] = torch.__version__
    runtime["cuda"] = torch.version.cuda
    runtime_pins = {
        "transformers": "5.15.0",
        "torch": "2.13.0+cu130",
        "torch_distribution": "2.13.0",
        "cuda": "13.0",
        "accelerate": "1.13.0",
        "numpy": "2.3.5",
    }
    if any(runtime[key] != value for key, value in runtime_pins.items()):
        raise ValueError("Capture package versions differ from the validated runtime")
    out.mkdir(parents=True, exist_ok=True)
    launch_binding = {
        "model": MODEL,
        "capture": CAPTURE,
        "sources_sha256": sources,
        "runtime": runtime,
        "manifest_sha256": sha256(manifest),
        "prefix_tokens_sha256": sha256(prefix_path),
        "selection_sha256": sha256(selection_path),
        "independent_review": review,
        "imported_sources_sha256": imported_source_hashes(),
        "activation_position": "decoder_block_44_output_at_last_unpadded_initial_prompt_token",
    }
    launch_path = out / "capture_launch_binding.json"
    if not launch_path.exists() and any(out.iterdir()):
        raise ValueError(
            "First capture launch requires empty output; cannot adopt unbound old chunks"
        )
    if launch_path.exists() and json.loads(launch_path.read_text()) != launch_binding:
        raise ValueError("Capture resume source/input/runtime regime changed")
    write_json(launch_path, launch_binding)
    report = run_capture(cfg)
    captured = []
    for path in sorted(out.glob("chunk_*.rows.jsonl")):
        with path.open() as handle:
            captured.extend(json.loads(line) for line in handle if line.strip())
    if len(captured) != 249 or {r["exact_context_sha256"] for r in captured} != set(prefixes):
        raise ValueError("Captured context roster differs from generation")
    for row in captured:
        prefix = prefixes[row["exact_context_sha256"]]
        if (
            digest(prefix["token_ids"]) != prefix["prefix_token_ids_sha256"]
            or len(prefix["token_ids"]) != prefix["n_prefix_tokens"]
        ):
            raise ValueError("Saved generation token IDs disagree with their hash/length")
        for field in ("prefix_token_ids_sha256", "n_prefix_tokens"):
            if row[field] != prefix[field]:
                raise ValueError("Captured initial tokens differ from the live generation prefix")
    if (
        sources != source_hashes()
        or imported_source_hashes() != launch_binding["imported_sources_sha256"]
    ):
        raise ValueError("Capture source changed during the run")
    binding = {
        **launch_binding,
        "passed": True,
        "schema_version": "context_risk_followup_capture_v1",
        "fingerprint": report["fingerprint"],
        "run_result_sha256": sha256(out / "run_result.json"),
        "chunk_files_sha256": {
            p.name: sha256(p) for p in sorted(out.glob("chunk_*")) if p.is_file()
        },
    }
    write_json(out / "capture_binding.json", binding)
    validate_binding(out, manifest, prefix_path, selection_path)
    return binding


@hydra.main(
    version_base="1.3", config_path="../configs/eval", config_name="context_risk_followup_capture"
)
def main(cfg: DictConfig) -> None:
    print(json.dumps(run(cfg), indent=2), flush=True)


if __name__ == "__main__":
    main()
