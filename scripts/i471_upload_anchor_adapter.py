"""Mirror + upload a chosen anchor checkpoint to a stepped adapter directory.

Plan v3 §4.5 / round-2 code-review BLOCKER fix.

Phase A trains ``adapters/<run_name>/checkpoint-<step>/`` directories every
``save_steps`` (=10) steps. The Phase A analyzer picks ONE step as the
anchor; from there Phase 4 needs to load THAT specific step's adapter, and
Phase 5 needs to look up the cell under a stable adapter_id.

This script does two things, in order:

  1. **Mirror** ``adapters/<run_name>/checkpoint-<step>/`` to
     ``adapters/<run_name>_step<step>/`` (copy, not symlink — keeps
     ``_download_adapters`` simple and survives subsequent in-session
     ``rm`` of the original checkpoint dir if any cleanup fires).
     Idempotent: re-runs over an existing target are no-ops.

  2. **Upload** ``adapters/<run_name>_step<step>/`` to HF under
     ``superkaiba1/explore-persona-space`` at the matching subfolder
     ``adapters/<run_name>_step<step>/``. Verifies the canonical files
     land via ``huggingface_hub.list_repo_files``; raises if not.

The upload is for **durability + reproducibility** (so the eval can be
re-run from a fresh pod / future seed sweep), NOT because the in-session
Phase 4 needs HF round-trip — Phase 4's ``_download_adapters`` already
prefers the local path. The two together close the v3-round-1
adapter-loss hole without burning extra HF traffic on every saved step.

Usage::

    uv run python scripts/i471_upload_anchor_adapter.py \\
        --run-name i471_route_a_cond1_withneg \\
        --step 45

    # batch form (mirrors + uploads N adapters in sequence):
    uv run python scripts/i471_upload_anchor_adapter.py \\
        --run-name i471_route_a_cond1_withneg --step 45 \\
        --extra i471_route_a_cond1_posonly:38
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("i471.upload_anchor_adapter")

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")


def _mirror_checkpoint(run_name: str, step: int) -> Path:
    """Copy ``adapters/<run_name>/checkpoint-<step>/`` -> ``adapters/<run_name>_step<step>/``."""
    src = Path(f"adapters/{run_name}/checkpoint-{step}")
    dst = Path(f"adapters/{run_name}_step{step}")
    if not src.is_dir():
        raise RuntimeError(
            f"source checkpoint directory missing: {src}. "
            f"Did Phase A complete with save_steps>=10 and save_strategy='steps'?"
        )
    src_adapter = src / "adapter_model.safetensors"
    if not src_adapter.exists():
        raise RuntimeError(
            f"source checkpoint {src}/ has no adapter_model.safetensors; "
            "the checkpoint is empty or corrupted."
        )
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        target = dst / entry.name
        if target.exists():
            # Idempotent: skip files already present.
            continue
        if entry.is_file():
            shutil.copy2(entry, target)
        elif entry.is_dir():
            shutil.copytree(entry, target)
    # Sanity check: ensure both required files landed in dst.
    for fname in REQUIRED_FILES:
        if not (dst / fname).exists():
            raise RuntimeError(
                f"after mirror, {dst}/{fname} is missing. Source {src}/{fname} "
                f"present={(src / fname).exists()}."
            )
    logger.info("mirrored %s -> %s", src, dst)
    return dst


def _upload_and_verify(local_dir: Path, *, adapter_id: str) -> str:
    """Upload local_dir to HF at adapters/<adapter_id>/, verify files via Hub API."""
    from explore_persona_space.orchestrate.env import load_dotenv
    from explore_persona_space.orchestrate.hub import upload_model

    load_dotenv()  # ensure HF_TOKEN is in env

    path_in_repo = f"adapters/{adapter_id}"
    hub_path = upload_model(
        str(local_dir),
        repo_id=HF_MODEL_REPO,
        path_in_repo=path_in_repo,
    )
    if not hub_path:
        raise RuntimeError(
            f"upload_model returned empty hub_path for {local_dir} -> {path_in_repo}; "
            "treating as a failed upload (fail-loud per CLAUDE.md)."
        )

    # Fail-loud verify with Hub API (CLAUDE.md upload-policy rule).
    from huggingface_hub import list_repo_files

    files_on_hub = set(list_repo_files(HF_MODEL_REPO, repo_type="model", revision="main"))
    for fname in REQUIRED_FILES:
        expected = f"{path_in_repo}/{fname}"
        if expected not in files_on_hub:
            raise RuntimeError(
                f"post-upload verify FAIL: {expected} not on {HF_MODEL_REPO} "
                f"per list_repo_files. local={local_dir}. Do NOT proceed; "
                "Phase 4 would fall through to a stale-or-missing adapter."
            )
    logger.info("uploaded + verified %s on HF -> %s", local_dir, hub_path)
    return hub_path


def _parse_extra(extras: list[str]) -> list[tuple[str, int]]:
    """Parse --extra entries of the form ``<run_name>:<step>``."""
    out: list[tuple[str, int]] = []
    for raw in extras:
        if ":" not in raw:
            raise ValueError(f"--extra entry {raw!r} not of the form '<run_name>:<step>'.")
        rn, step_s = raw.rsplit(":", 1)
        try:
            step_i = int(step_s)
        except ValueError as e:
            raise ValueError(f"--extra entry {raw!r}: step component {step_s!r} not an int.") from e
        if step_i <= 0:
            raise ValueError(f"--extra entry {raw!r}: step must be > 0, got {step_i}.")
        out.append((rn, step_i))
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--run-name",
        required=True,
        help="Training run name (e.g. i471_route_a_cond1_withneg).",
    )
    ap.add_argument(
        "--step",
        required=True,
        type=int,
        help="Checkpoint step (e.g. 45). Must exist at adapters/<run-name>/checkpoint-<step>/.",
    )
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Additional <run_name>:<step> pairs to mirror+upload in the same invocation. "
        "Example: --extra i471_route_a_cond1_posonly:38 i471_route_a_cond2_k0_step45:45",
    )
    ap.add_argument(
        "--skip-upload",
        action="store_true",
        help="Mirror only — skip the HF upload step. Useful for debugging the mirror logic.",
    )
    args = ap.parse_args(argv)

    targets: list[tuple[str, int]] = [(args.run_name, args.step), *_parse_extra(args.extra)]

    for run_name, step in targets:
        mirrored_dir = _mirror_checkpoint(run_name, step)
        adapter_id = f"{run_name}_step{step}"
        if args.skip_upload:
            logger.info("skipping HF upload for %s (--skip-upload).", adapter_id)
            continue
        _upload_and_verify(mirrored_dir, adapter_id=adapter_id)

    logger.info("all anchor adapters mirrored + uploaded.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("FATAL: i471_upload_anchor_adapter failed.")
        sys.exit(2)
