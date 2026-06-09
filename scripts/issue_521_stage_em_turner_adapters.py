#!/usr/bin/env python3
"""Step 3.6.3 — stage retrained EM adapters from HF to the local layout.

Plan v2 §4 Step 3.6.3. Mirrors ``scripts/issue_521_stage_adapters.py``
but targets the v2 HF subfolder
``adapters/issue_521/em_turner_seed{S}/`` and stages each cell into
``eval_results/issue_521/em_turner_seed{S}/adapter/``.

Per-file ``hf_hub_download`` (not ``snapshot_download``) per the
``feedback_snapshot_download_siblings_truncation`` memory: on
``superkaiba1/explore-persona-space`` (~42k files) the siblings list
truncates and ``snapshot_download(allow_patterns=...)`` silently
returns 0 files for prefixes in the truncated tail.

Run::

    uv run python scripts/issue_521_stage_em_turner_adapters.py \\
        --output-dir eval_results/issue_521 \\
        [--seeds 42 137 256]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = (42, 137, 256)
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_REVISION = "main"
HF_SUBFOLDER_TEMPLATE = "adapters/issue_521/em_turner_seed{seed}"

# The retrain pipeline persists the LoRA adapter as the train script
# default (`_finalize_phase` per `.claude/rules/upload-policy.md`).
# The exact file list mirrors the v1 marker adapter stage but the
# tokenizer files may or may not be present depending on the trainer
# version — README and tokenizer companions are optional.
REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)
OPTIONAL_FILES = (
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _stage_one(*, seed: int, output_dir: Path) -> None:
    """Stage one retrained EM adapter cell via per-file hf_hub_download."""
    from huggingface_hub import hf_hub_download

    target = output_dir / f"em_turner_seed{seed}" / "adapter"
    cfg = target / "adapter_config.json"
    sft = target / "adapter_model.safetensors"
    if cfg.exists() and sft.exists():
        cfg_sz = cfg.stat().st_size
        sft_sz = sft.stat().st_size
        if cfg_sz < 100 or sft_sz < 1024:
            raise RuntimeError(
                f"adapter files at {target} are already present but suspiciously "
                f"small (config={cfg_sz}B, safetensors={sft_sz}B). Looks like a "
                f"stale LFS pointer or partial previous stage — delete {target} "
                f"and re-run to re-download."
            )
        logger.info(
            "[skip] em_turner_seed%d already staged at %s (cfg=%dB, safetensors=%.1fMB)",
            seed,
            target,
            cfg_sz,
            sft_sz / 1e6,
        )
        return
    target.mkdir(parents=True, exist_ok=True)

    hf_subfolder = HF_SUBFOLDER_TEMPLATE.format(seed=seed)
    logger.info(
        "[phase=stage] em_turner_seed%d: hf_hub_download per-file from %s::%s into %s",
        seed,
        HF_MODEL_REPO,
        hf_subfolder,
        target,
    )

    # REQUIRED first — any failure here is fatal.
    for fname in REQUIRED_FILES:
        hf_path = f"{hf_subfolder}/{fname}"
        try:
            local = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=hf_path,
                revision=HF_REVISION,
            )
        except Exception as e:
            raise RuntimeError(
                f"hf_hub_download failed for {HF_MODEL_REPO}@{HF_REVISION} :: "
                f"{hf_path}: {type(e).__name__}: {e}"
            ) from e
        dest = target / fname
        if dest.exists():
            dest.unlink()
        shutil.copy2(local, dest)

    # OPTIONAL — log warnings on miss, never raise.
    for fname in OPTIONAL_FILES:
        hf_path = f"{hf_subfolder}/{fname}"
        try:
            local = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=hf_path,
                revision=HF_REVISION,
            )
        except Exception as e:
            logger.info(
                "[stage] em_turner_seed%d: optional %s missing on HF: %s",
                seed,
                hf_path,
                type(e).__name__,
            )
            continue
        dest = target / fname
        if dest.exists():
            dest.unlink()
        shutil.copy2(local, dest)

    # Fail-loud size sanity per v1 staging pattern.
    cfg_sz = cfg.stat().st_size
    sft_sz = sft.stat().st_size
    if cfg_sz < 100 or sft_sz < 1024:
        raise RuntimeError(
            f"adapter files at {target} are suspiciously small "
            f"(config={cfg_sz}B, safetensors={sft_sz}B) — likely an LFS pointer, "
            f"not the real adapter."
        )
    logger.info(
        "[stage_done] em_turner_seed%d at %s (cfg=%dB, safetensors=%.1fMB)",
        seed,
        target,
        cfg_sz,
        sft_sz / 1e6,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage retrained EM adapters for #521 v2 Phase C",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/issue_521",
        help="Top-level output dir; adapters land under em_turner_seed{S}/adapter/.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="EM-turner seeds to stage (default 42 137 256).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    logger.info("staging %d retrained EM-turner cells: seeds=%s", len(seeds), seeds)

    for seed in seeds:
        _stage_one(seed=seed, output_dir=output_dir)

    logger.info("[phase=done] all %d em_turner cells staged at %s", len(seeds), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
