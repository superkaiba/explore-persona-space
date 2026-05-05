#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #228 Round 6 — salvage marker LoRAs from WandB Artifacts to HF Hub.

Background
----------
The R5 sweep launch trained 70 marker LoRAs in Phase 0 and uploaded each one
to **WandB Artifacts** successfully (canonical artifact names of the form
``cp_marker_<src>_ep<step>_s42-checkpoint:latest`` under the
``issue228_marker_loras`` project). The same workers TRIED to push every
adapter to HF Hub at ``superkaiba1/explore-persona-space/adapters/cp_marker_<src>_ep<step>_s42``,
but every push failed with a ``400 Bad Request — Invalid metadata in README.md``
because PEFT's ``save_pretrained`` writes a frontmatter ``base_model:``
field equal to the *local* merged-model path
(``/workspace/tmp/issue228_markerlora/<src>_ckpt<step>_merged``).

The HF Hub upload error was logged at WARNING and the worker reported
``TRAINED + uploaded`` regardless, so the local copies were deleted and the
adapters never reached HF Hub. Phase 0.5 / Phase 1 then expected to find
them at HF Hub and crashed. This script reconstructs the HF Hub state from
the WandB-side canonical copies.

What this script does
---------------------
For each (source, ckpt) state in the 70-state grid (7 sources × 10
checkpoints, ckpt > 0):

  1. **Idempotency check.** If
     ``adapters/cp_marker_<src>_ep<step>_s42/adapter_config.json`` already
     exists on the HF Hub model repo, skip.
  2. Download the WandB Artifact ``huggingface/cp_marker_<src>_ep<step>_s42-checkpoint:latest``
     (project ``issue228_marker_loras``) into a temp dir.
  3. Sanity-check the contents: ``adapter_config.json`` and
     ``adapter_model.safetensors`` MUST both be present. If either is
     missing the salvage for this state is logged and skipped (we don't
     want to push a partially-uploaded artifact).
  4. **Rewrite README.md frontmatter** to set
     ``base_model: Qwen/Qwen2.5-7B-Instruct`` (the actual base model id),
     replacing the broken local-path value PEFT wrote. We re-use the
     ``rewrite_adapter_readme_base_model`` helper from
     ``train_marker_loras_228.py``.
  5. Upload the directory to HF Hub at
     ``adapters/cp_marker_<src>_ep<step>_s42``.
  6. Verify via ``HfApi.list_repo_files`` that the adapter files are now
     present on the Hub.

After the loop, prints a summary table:

  * **Already on HF Hub** — skipped, no action needed.
  * **Newly salvaged** — successfully pulled from WandB and pushed to HF
    Hub by this script.
  * **Missing from WandB** — the WandB-side canonical copy is also gone
    (would mean BOTH stores failed during R5; should be empty if the R5
    worker logs are correct).
  * **Salvage failed** — WandB pull or HF push failed for an idiosyncratic
    reason; safe to re-run the script to retry.

This script can run on the local VM (no GPU needed). It only touches HF
Hub and WandB.

Invocation
----------

::

    uv run python scripts/salvage_marker_loras_from_wandb_228.py \\
        --output-summary eval_results/issue_228/_salvage_summary.json

    # Dry-run: list the work but don't pull or push.
    uv run python scripts/salvage_marker_loras_from_wandb_228.py --dry-run

    # Salvage one specific state for testing.
    uv run python scripts/salvage_marker_loras_from_wandb_228.py \\
        --only-source villain --only-step 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Project-side imports must come AFTER bootstrap()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from compute_js_convergence_228 import (  # noqa: E402
    ADAPTER_MAP,
    ADAPTER_REPO,
    BASE_MODEL,
    CHECKPOINT_STEPS,
)
from train_marker_loras_228 import (  # noqa: E402
    rewrite_adapter_readme_base_model,
)

# HOT-FIX (issue #228): the R5 training run uploaded artifacts under WandB
# project "huggingface" (HF Trainer's default `wandb.init(project=...)`),
# not "issue228_marker_loras" as the train script's WANDB_PROJECT constant
# suggests — `upload_model_wandb` reused the already-active wandb.run, which
# was initialized by HF Trainer before our code ran. The artifacts therefore
# live at `<entity>/huggingface/cp_marker_<src>_ep<step>_s42-checkpoint:latest`.
WANDB_PROJECT = "huggingface"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("salvage_marker_loras_228")

DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "eval_results" / "issue_228" / "_salvage_summary.json"


# ── HF Hub idempotency ────────────────────────────────────────────────────


def _hf_already_has_marker_lora(api, source: str, step: int, seed: int = 42) -> bool:
    """Same predicate as `train_marker_loras_228._hf_already_has_marker_lora`,
    but takes a pre-built HfApi so we can list the repo once."""
    target_subpath = f"adapters/cp_marker_{source}_ep{step}_s{seed}"
    target_file = f"{target_subpath}/adapter_config.json"
    try:
        files = api.list_repo_files(ADAPTER_REPO)
    except Exception as exc:
        logger.warning("HfApi.list_repo_files failed (%s); proceeding without idempotency", exc)
        return False
    return target_file in files


# ── WandB Artifact pull ───────────────────────────────────────────────────


def _resolve_wandb_entity() -> str:
    """Look up the WandB default entity from the local config / env.

    The R5 worker logs reference the artifact as
    ``wandb://huggingface/cp_marker_villain_ep400_s42-checkpoint:latest``;
    the entity portion is the WandB-team scope the artifact was logged
    under. We don't hardcode it: we look it up via ``wandb.Api().default_entity``.
    """
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    if not entity:
        raise RuntimeError(
            "wandb.Api().default_entity returned None — log in with `wandb login` "
            "or set WANDB_ENTITY before running the salvage script."
        )
    return entity


def _pull_wandb_artifact(
    *,
    entity: str,
    project: str,
    artifact_name: str,
    download_dir: Path,
) -> Path | None:
    """Download an artifact's directory contents into ``download_dir``.

    Returns the local path to the artifact directory on success, or None
    if the artifact is missing / inaccessible.
    """
    import wandb
    from wandb.errors import CommError

    full_ref = f"{entity}/{project}/{artifact_name}:latest"
    try:
        api = wandb.Api()
        artifact = api.artifact(full_ref, type="model")
    except CommError as exc:
        logger.warning("WandB artifact %s not found (CommError): %s", full_ref, exc)
        return None
    except Exception as exc:
        logger.warning("WandB Api.artifact(%s) raised: %s", full_ref, exc)
        return None
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_path = artifact.download(root=str(download_dir))
    except Exception as exc:
        logger.error("WandB artifact %s download failed: %s", full_ref, exc)
        return None
    return Path(local_path)


def _verify_adapter_dir(adapter_dir: Path) -> tuple[bool, list[str]]:
    """Sanity-check that ``adapter_dir`` contains a real LoRA adapter.

    Returns ``(ok, missing_files)``. We require both ``adapter_config.json``
    and ``adapter_model.safetensors`` to be present and non-empty.
    """
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing: list[str] = []
    for name in required:
        p = adapter_dir / name
        if not p.exists() or p.stat().st_size == 0:
            missing.append(name)
    return (len(missing) == 0), missing


# ── HF Hub push (with README rewrite) ─────────────────────────────────────


def _push_adapter_to_hub(
    api,
    adapter_dir: Path,
    *,
    source: str,
    step: int,
    seed: int = 42,
) -> bool:
    """Rewrite README + push to HF Hub. Verify by listing the repo afterwards."""
    rewrite_adapter_readme_base_model(adapter_dir, BASE_MODEL)

    path_in_repo = f"adapters/cp_marker_{source}_ep{step}_s{seed}"
    try:
        api.create_repo(ADAPTER_REPO, repo_type="model", private=False, exist_ok=True)
    except Exception as exc:
        logger.warning("create_repo for %s raised: %s", ADAPTER_REPO, exc)

    try:
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=ADAPTER_REPO,
            path_in_repo=path_in_repo,
            repo_type="model",
        )
    except Exception as exc:
        logger.error(
            "upload_folder failed for %s -> %s/%s: %s",
            adapter_dir,
            ADAPTER_REPO,
            path_in_repo,
            exc,
        )
        return False

    # Verify by re-listing.
    try:
        files_after = api.list_repo_files(ADAPTER_REPO)
    except Exception as exc:
        logger.error("list_repo_files post-upload raised: %s", exc)
        return False

    needle = f"{path_in_repo}/adapter_config.json"
    if needle not in files_after:
        logger.error("Verification FAILED: %s not present in repo listing post-upload", needle)
        return False
    logger.info("Verified: %s on HF Hub", path_in_repo)
    return True


# ── Top-level loop ────────────────────────────────────────────────────────


def _enumerate_states() -> list[tuple[str, int]]:
    """70 states: 7 sources × 10 checkpoints (matches Phase 0)."""
    states: list[tuple[str, int]] = []
    for source in sorted(ADAPTER_MAP.keys()):
        for step in CHECKPOINT_STEPS:
            states.append((source, step))
    return states


def _filter_states(
    states: list[tuple[str, int]],
    *,
    only_source: str | None,
    only_step: int | None,
) -> list[tuple[str, int]]:
    out = states
    if only_source is not None:
        out = [s for s in out if s[0] == only_source]
    if only_step is not None:
        out = [s for s in out if s[1] == only_step]
    return out


def salvage_one_state(
    api,
    *,
    source: str,
    step: int,
    seed: int,
    entity: str,
    dry_run: bool,
    keep_local: bool = False,
) -> dict:
    """Salvage one (source, step) state. Returns a result dict."""
    artifact_name = f"cp_marker_{source}_ep{step}_s{seed}-checkpoint"
    record: dict = {
        "source": source,
        "checkpoint_step": step,
        "seed": seed,
        "artifact_ref": f"{entity}/{WANDB_PROJECT}/{artifact_name}:latest",
        "hf_target": f"{ADAPTER_REPO}/adapters/cp_marker_{source}_ep{step}_s{seed}",
        "status": "PENDING",
        "wall_seconds": 0.0,
    }
    t_start = time.time()

    # 1. Idempotency.
    if _hf_already_has_marker_lora(api, source, step, seed):
        record["status"] = "ALREADY_ON_HF"
        record["wall_seconds"] = time.time() - t_start
        return record

    if dry_run:
        record["status"] = "WOULD_SALVAGE"
        record["wall_seconds"] = time.time() - t_start
        return record

    # 2. Pull from WandB.
    with tempfile.TemporaryDirectory(prefix="salvage228_") as tmp:
        tmp_root = Path(tmp)
        local_path = _pull_wandb_artifact(
            entity=entity,
            project=WANDB_PROJECT,
            artifact_name=artifact_name,
            download_dir=tmp_root,
        )
        if local_path is None:
            record["status"] = "MISSING_FROM_WANDB"
            record["wall_seconds"] = time.time() - t_start
            return record
        # 3. Sanity check.
        ok, missing = _verify_adapter_dir(local_path)
        if not ok:
            record["status"] = "WANDB_ARTIFACT_INCOMPLETE"
            record["missing_files"] = missing
            record["wall_seconds"] = time.time() - t_start
            return record
        # 4 + 5. Rewrite README + push.
        pushed = _push_adapter_to_hub(api, local_path, source=source, step=step, seed=seed)
        if not pushed:
            record["status"] = "PUSH_FAILED"
            record["wall_seconds"] = time.time() - t_start
            return record
        if keep_local:
            keep_dst = PROJECT_ROOT / "salvage_keep" / f"{source}_ckpt{step}"
            keep_dst.parent.mkdir(parents=True, exist_ok=True)
            if keep_dst.exists():
                shutil.rmtree(keep_dst, ignore_errors=True)
            shutil.copytree(local_path, keep_dst)
            record["kept_local_at"] = str(keep_dst)

    record["status"] = "SALVAGED"
    record["wall_seconds"] = time.time() - t_start
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON summary path. Default eval_results/issue_228/_salvage_summary.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the work and exit without pulling or pushing.",
    )
    parser.add_argument(
        "--only-source",
        type=str,
        default=None,
        help="Limit to this source (debug/single-state retry).",
    )
    parser.add_argument(
        "--only-step",
        type=int,
        default=None,
        help="Limit to this checkpoint step (debug/single-state retry).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override the WandB entity. Default: lookup via wandb.Api().default_entity.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Copy each salvaged adapter to salvage_keep/ for offline inspection.",
    )
    args = parser.parse_args()

    states = _filter_states(
        _enumerate_states(),
        only_source=args.only_source,
        only_step=args.only_step,
    )
    if not states:
        logger.error("No states match the filters; nothing to do.")
        return 1
    logger.info("Salvaging %d states (dry_run=%s)", len(states), args.dry_run)

    if args.dry_run:
        entity = args.wandb_entity or "<unresolved>"
    else:
        entity = args.wandb_entity or _resolve_wandb_entity()
        logger.info("Using WandB entity: %s", entity)

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    records: list[dict] = []
    for source, step in states:
        logger.info("--- %s ckpt-%d ---", source, step)
        rec = salvage_one_state(
            api,
            source=source,
            step=step,
            seed=args.seed,
            entity=entity,
            dry_run=args.dry_run,
            keep_local=args.keep_local,
        )
        logger.info("[%s ckpt-%d] %s (%.1fs)", source, step, rec["status"], rec["wall_seconds"])
        records.append(rec)

    # Aggregate stats.
    totals = {
        "ALREADY_ON_HF": 0,
        "SALVAGED": 0,
        "WOULD_SALVAGE": 0,
        "MISSING_FROM_WANDB": 0,
        "WANDB_ARTIFACT_INCOMPLETE": 0,
        "PUSH_FAILED": 0,
        "PENDING": 0,
    }
    for rec in records:
        totals[rec["status"]] = totals.get(rec["status"], 0) + 1

    summary = {
        "n_states": len(states),
        "totals": totals,
        "records": records,
        "dry_run": args.dry_run,
        "wandb_entity": entity,
        "wandb_project": WANDB_PROJECT,
        "hf_repo": ADAPTER_REPO,
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", args.output_summary)

    # Print a one-line summary line for easy log-grepping.
    logger.info(
        "SALVAGE-SUMMARY total=%d already=%d salvaged=%d would=%d "
        "missing_wandb=%d incomplete=%d push_failed=%d",
        len(states),
        totals.get("ALREADY_ON_HF", 0),
        totals.get("SALVAGED", 0),
        totals.get("WOULD_SALVAGE", 0),
        totals.get("MISSING_FROM_WANDB", 0),
        totals.get("WANDB_ARTIFACT_INCOMPLETE", 0),
        totals.get("PUSH_FAILED", 0),
    )

    # Exit non-zero if there are any non-recoverable holes (would only
    # happen if BOTH HF Hub AND WandB lost the artifact).
    holes = (
        totals.get("MISSING_FROM_WANDB", 0)
        + totals.get("WANDB_ARTIFACT_INCOMPLETE", 0)
        + totals.get("PUSH_FAILED", 0)
    )
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
