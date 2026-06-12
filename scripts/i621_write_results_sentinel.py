"""Issue #621 results-sentinel writer (pod-side; the LAST step before [phase=done]).

Composes the ``epm:results`` sentinel with the §10 MACHINE-RESOLVABLE
reproducibility card and writes it to the poll_pipeline.py sentinel path
(``/workspace/logs/issue-621-epm_results-<epoch>.json`` — required keys
``sentinel_schema_version`` / ``kind`` / ``version``).

Card fields (workflow.yaml § markers epm:results):
  - ``adapter_paths``: explicit per-cell map ``{cell_slug: hf_subfolder}``,
    EVERY path verified under ``hf_model_repo`` via a fresh
    ``list_repo_files`` (a missing adapter raises — the pipeline aborts
    before [phase=done], never ships a card with dead paths).
  - ``wandb_project`` + ``wandb_run_names``: per-cell run names recorded by
    the train dispatcher (MANDATORY card fields when training logs WandB).
  - data-repo buckets (mixes / shifts / trajectories / raw completions /
    context bank) + eval-glob counts.

Pod-side code never shells out to scripts/task.py — the VM orchestrator's
poll loop ingests this sentinel and posts the marker.

CLI:
    uv run python scripts/i621_write_results_sentinel.py \\
        [--out-root eval_results/issue_621] [--sentinel-dir /workspace/logs] [--note "..."]
"""

# math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.experiments.issue_621 import (
    BASE_MODEL,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_BUCKET,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    HF_TRAIN_MIX_PATH_PREFIX,
    HF_TRAIN_MIX_READ_REVISION,
    WANDB_PROJECT,
)

log = logging.getLogger("issue_621.sentinel")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_cells(out_root: Path) -> list[dict]:
    cells: dict[str, dict] = {}
    for sub in ("anchor_smoke", "sweep"):
        d = out_root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name in ("summary.json", "smoke_gate.json"):
                continue
            payload = json.loads(p.read_text())
            if "cell_slug" in payload:
                cells[payload["cell_slug"]] = payload
    if not cells:
        raise SystemExit(f"no trained cell JSONs under {out_root}/{{anchor_smoke,sweep}}")
    return [cells[k] for k in sorted(cells)]


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="eval_results/issue_621")
    ap.add_argument("--sentinel-dir", default="/workspace/logs")
    ap.add_argument("--note", default="", help="One-line run summary prefix for the note body.")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    cells = _load_cells(out_root)

    # Verify EVERY per-cell adapter path resolves on the Hub (fresh listing).
    from huggingface_hub import list_repo_files

    listed = set(list_repo_files(HF_MODEL_REPO, repo_type="model"))
    adapter_paths: dict[str, str] = {}
    missing: list[str] = []
    for cell in cells:
        sub = cell["hf_subfolder"]
        adapter_paths[cell["cell_slug"]] = sub
        if f"{sub}/adapter_model.safetensors" not in listed:
            missing.append(sub)
    if missing:
        raise SystemExit(
            f"{len(missing)} adapter path(s) NOT resolvable on {HF_MODEL_REPO}: "
            f"{missing[:3]} ... — refusing to write a reproducibility card "
            "with dead paths. Re-run the upload before the sentinel."
        )
    # A-init snapshots ride in the same per-cell folders.
    init_missing = [
        cell["hf_subfolder"]
        for cell in cells
        if f"{cell['hf_subfolder']}/adapter_init/adapter_model.safetensors" not in listed
    ]
    if init_missing:
        raise SystemExit(
            f"{len(init_missing)} adapter_init snapshot(s) missing on Hub: "
            f"{init_missing[:3]} ... — the A-init control is unrunnable without "
            "them; re-run the upload."
        )

    eval_jsons = sorted((out_root / "eval").glob("*__seed*.json"))
    bank_manifest = out_root / "context_vectors" / "manifest.json"

    card = {
        "base_model": BASE_MODEL,
        "hf_model_repo": HF_MODEL_REPO,
        "adapter_paths": adapter_paths,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_names": [c["wandb_run_name"] for c in cells],
        "hf_data_repo": HF_DATA_REPO,
        "training_mixes_prefix": HF_TRAIN_MIX_PATH_PREFIX,
        "analysis_tensors_prefix": HF_ANALYSIS_TENSORS_PREFIX,
        "raw_completions_prefix": f"{HF_BUCKET}/raw_completions",
        "trajectories_prefix": f"{HF_BUCKET}/trajectories",
        "context_bank_prefix": f"{HF_ANALYSIS_TENSORS_PREFIX}",
        "hf_train_mix_read_revision": HF_TRAIN_MIX_READ_REVISION,
        "n_cells": len(cells),
        "n_eval_jsons": len(eval_jsons),
        "bank_manifest_local": str(bank_manifest) if bank_manifest.is_file() else None,
        "git_commit": _git_commit(),
        "band_entry_steps_per_cell": {c["cell_slug"]: c.get("band_stop_step") for c in cells},
        "final_source_delta_nats_per_cell": {
            c["cell_slug"]: c.get("final_source_delta_nats") for c in cells
        },
    }

    note = {
        "summary": args.note
        or (
            f"issue #621 rank-1 read/write pipeline complete: {len(cells)} cells "
            f"trained, {len(eval_jsons)} eval JSONs, bank "
            f"{'present' if bank_manifest.is_file() else 'MISSING'}"
        ),
        "reproducibility_card": card,
    }

    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 621,
        "by": "issue621_pipeline",
        "ts": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds"),
        "note": json.dumps(note, ensure_ascii=False),
    }
    sentinel_dir = Path(args.sentinel_dir)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    out_path = sentinel_dir / f"issue-621-epm_results-{int(time.time())}.json"
    out_path.write_text(json.dumps(sentinel, indent=1, ensure_ascii=False))
    log.info("results sentinel written: %s (%d cells, card verified)", out_path, len(cells))
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
