"""Issue #464 ``minimal_content`` — verify (and repair) a train cell's HF adapter upload.

``train_lora``'s HF adapter upload is SOFT-FAIL (it warns and returns 0 on
an upload exception), while the downstream cross-eval / logit-capture
phases download adapters FROM HF. Without this check, a failed upload
marks the train cell "ok" and the pipeline crashes a phase later (or,
worse with a local-adapter skip path, never notices at all).

Contract (fail-loud):
  1. Verify ``adapters/i464_<cell>/adapter_model.safetensors`` AND
     ``adapter_config.json`` resolve on the HF model repo via
     ``huggingface_hub.list_repo_files`` (the Python Hub API — NOT the
     ``hf`` CLI, which has no ``api`` subcommand and reads as a false
     "0 files"; CLAUDE.md Upload Policy / #458 post-mortem).
  2. On missing: re-upload the LOCAL adapter dir via
     ``HfApi.upload_folder``, then RE-VERIFY via ``list_repo_files``.
  3. If the local dir is absent too, or the re-verify still misses:
     raise — the runner marks the cell failed.

Exit code 0 ⇔ the adapter verifiably resolves on HF.

CLI:
    uv run python scripts/i464_min_verify_upload.py --cell system_minimal_seed42
    uv run python scripts/i464_min_verify_upload.py --cell role_bare_seed137 \
        --local-dir adapters/i464_role_bare_seed137
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i464.min_verify_upload")

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")


def _missing_on_hub(repo: str, subpath: str) -> list[str]:
    """Return the REQUIRED_FILES not currently resolving under ``subpath`` on ``repo``."""
    from huggingface_hub import list_repo_files

    files = set(list_repo_files(repo, repo_type="model", revision="main"))
    return [f for f in REQUIRED_FILES if f"{subpath}/{f}" not in files]


def verify_or_reupload(
    cell: str, local_dir: Path, repo: str = HF_MODEL_REPO, prefix: str = "i464"
) -> str:
    """Verify the cell's adapter on HF; re-upload from ``local_dir`` if missing.

    ``prefix`` selects the per-run HF subpath prefix the train script
    writes under (matches the train's ``hf_path_in_repo`` prefix):
    ``adapters/{prefix}_{cell}``. Default ``"i464"`` preserves the
    parent #464 / minimal_content_cn behavior. Pass ``"i533bw"`` for
    the #533 bare-word follow-up.

    Returns a one-word status: ``"verified"`` (already on HF) or
    ``"reuploaded"`` (was missing, re-upload + re-verify succeeded).

    Raises:
        RuntimeError if the adapter is missing on HF AND the local dir
        cannot supply it, or if the re-upload does not verify.
    """
    subpath = f"adapters/{prefix}_{cell}"
    missing = _missing_on_hub(repo, subpath)
    if not missing:
        logger.info("cell=%s adapter verified on HF: %s/%s", cell, repo, subpath)
        return "verified"

    logger.warning(
        "cell=%s adapter MISSING on HF (%s/%s lacks %s); attempting re-upload from %s",
        cell,
        repo,
        subpath,
        missing,
        local_dir,
    )
    if not (local_dir / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"cell={cell}: adapter missing on HF ({missing}) AND no local copy at "
            f"{local_dir}/adapter_model.safetensors — cannot repair; re-train the cell."
        )
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        folder_path=str(local_dir),
        path_in_repo=subpath,
        repo_id=repo,
        repo_type="model",
        commit_message=f"task {prefix}: re-upload adapter {subpath}",
    )
    still_missing = _missing_on_hub(repo, subpath)
    if still_missing:
        raise RuntimeError(
            f"cell={cell}: re-upload completed but {still_missing} STILL not resolving "
            f"on {repo}/{subpath} via list_repo_files — refusing to mark the cell ok."
        )
    logger.info("cell=%s adapter re-uploaded + verified on HF: %s/%s", cell, repo, subpath)
    return "reuploaded"


def main(argv: list[str] | None = None) -> None:
    """Entry point: verify one cell's adapter upload (repairing if possible)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cell",
        required=True,
        help="Cell label, e.g. 'system_minimal_seed42' (HF subpath adapters/<prefix>_<cell>).",
    )
    ap.add_argument(
        "--local-dir",
        default=None,
        help="Local adapter dir for the repair path (default: adapters/<prefix>_<cell>).",
    )
    ap.add_argument("--repo", default=HF_MODEL_REPO, help="HF model repo id.")
    ap.add_argument(
        "--prefix",
        default="i464",
        help=(
            "HF subpath prefix matching the train script's hf_path_in_repo "
            "(``i464`` for parent / min_cn; ``i533bw`` for the #533 "
            "bare-word follow-up)."
        ),
    )
    args = ap.parse_args(argv)

    local_dir = (
        Path(args.local_dir) if args.local_dir else Path(f"adapters/{args.prefix}_{args.cell}")
    )
    status = verify_or_reupload(args.cell, local_dir, repo=args.repo, prefix=args.prefix)
    print(f"upload-verify cell={args.cell}: {status}")


if __name__ == "__main__":
    main()
