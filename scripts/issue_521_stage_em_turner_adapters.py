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
# Per-seed PREFIX (no leaf). _finalize_phase (src/.../train/trainer.py:508)
# appends the per-phase leaf (`adapter_dir.name`, e.g. `sft_narrow_adapter`
# for the `issue404_pair_turner_bad_medical` condition) so the actual
# files live at
#   adapters/issue_521/em_turner_seed{S}/<leaf>/adapter_{config.json,model.safetensors,...}
# We discover the leaf via list_repo_files instead of hardcoding it; this
# is robust to the leaf name shifting if the condition's stage name ever
# changes (round-1 hardcoded the no-leaf path → guaranteed 404 after GPU
# spend; round-2 fix Critical #3).
HF_SEED_PREFIX_TEMPLATE = "adapters/issue_521/em_turner_seed{seed}"

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


def _discover_leaf(*, seed_prefix: str, hf_files: list[str]) -> str:
    """Discover the per-seed leaf subfolder by scanning hf_files.

    Expects exactly one leaf subfolder under ``seed_prefix`` containing
    BOTH ``adapter_config.json`` AND ``adapter_model.safetensors``. Fails
    loud on zero matches OR multiple matches: silent ambiguity in the
    leaf name is exactly what Critical #3 (round 2) was about.

    Returns the full HF prefix INCLUDING the leaf, e.g.
    ``adapters/issue_521/em_turner_seed42/sft_narrow_adapter``.
    """
    seed_files = [f for f in hf_files if f.startswith(f"{seed_prefix}/")]
    if not seed_files:
        raise RuntimeError(
            f"HF list_repo_files returned 0 files under prefix {seed_prefix!r}; "
            f"production train may have failed to persist the adapter, or the "
            f"path on HF is unexpected. Re-run the persist-verify step in the "
            f"sweep before staging."
        )

    # Look for adapter_config.json + adapter_model.safetensors co-located in
    # the SAME subdir under seed_prefix. Group files by their immediate
    # parent under the prefix.
    by_leaf: dict[str, set[str]] = {}
    for path in seed_files:
        # path = "<seed_prefix>/<maybe-leaf>/.../<fname>". Trim the prefix +
        # take the FIRST path component as the leaf candidate.
        rel = path[len(seed_prefix) + 1 :]
        parts = rel.split("/")
        if len(parts) < 2:
            # File sits directly at seed_prefix root (no leaf). Use "" as
            # the leaf key so we still pick this up if a future trainer
            # version drops the leaf appendage.
            leaf_key = ""
            fname = parts[0]
        else:
            leaf_key = parts[0]
            fname = parts[-1]
        by_leaf.setdefault(leaf_key, set()).add(fname)

    candidates = [
        leaf
        for leaf, fnames in by_leaf.items()
        if {"adapter_config.json", "adapter_model.safetensors"}.issubset(fnames)
    ]
    if not candidates:
        raise RuntimeError(
            f"No leaf under {seed_prefix!r} carries BOTH adapter_config.json "
            f"AND adapter_model.safetensors. Files found per leaf: "
            f"{ {k: sorted(v) for k, v in by_leaf.items()} }"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous adapter leaf under {seed_prefix!r}: candidates="
            f"{sorted(candidates)}. Expected exactly one (e.g. 'sft_narrow_adapter')."
        )
    leaf = candidates[0]
    return f"{seed_prefix}/{leaf}" if leaf else seed_prefix


def _stage_one(*, seed: int, output_dir: Path, hf_files: list[str]) -> None:
    """Stage one retrained EM adapter cell via per-file hf_hub_download.

    Round-2 fix (Critical #3): discovers the per-seed leaf via
    list_repo_files (cached at the call site as ``hf_files``) instead
    of hardcoding `adapters/issue_521/em_turner_seed{seed}` — which
    misses the `sft_narrow_adapter` leaf that ``_finalize_phase``
    appends at write time.
    """
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

    seed_prefix = HF_SEED_PREFIX_TEMPLATE.format(seed=seed)
    hf_subfolder = _discover_leaf(seed_prefix=seed_prefix, hf_files=hf_files)
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
        if hf_path not in hf_files:
            raise RuntimeError(
                f"required file {hf_path!r} not in HF repo listing "
                f"(list_repo_files turned up {len(hf_files)} files under "
                f"{HF_MODEL_REPO}@{HF_REVISION}). Refusing to attempt the "
                f"download — the leaf discovery may have picked the wrong "
                f"subdir for this seed."
            )
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

    # OPTIONAL — log warnings on miss, never raise. Skip the download
    # entirely when the file isn't in the listing (no 404 round-trip).
    for fname in OPTIONAL_FILES:
        hf_path = f"{hf_subfolder}/{fname}"
        if hf_path not in hf_files:
            logger.info(
                "[stage] em_turner_seed%d: optional %s not in HF listing; skipping",
                seed,
                hf_path,
            )
            continue
        try:
            local = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=hf_path,
                revision=HF_REVISION,
            )
        except Exception as e:
            logger.info(
                "[stage] em_turner_seed%d: optional %s download failed: %s",
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

    # Round-2 fix (Critical #3): list the model repo's files ONCE so
    # _stage_one can discover the per-seed leaf via path scan instead
    # of hardcoding `sft_narrow_adapter`. ``list_repo_files`` is
    # authoritative + paginated under the hood, so it survives the
    # 8k-sibling truncation that bit ``snapshot_download``.
    from huggingface_hub import list_repo_files

    try:
        hf_files = list(
            list_repo_files(
                HF_MODEL_REPO,
                repo_type="model",
                revision=HF_REVISION,
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"list_repo_files failed for {HF_MODEL_REPO}@{HF_REVISION}: "
            f"{type(e).__name__}: {e}. Cannot discover adapter leaf without "
            f"this; refusing to attempt blind downloads."
        ) from e
    logger.info(
        "[phase=hf_list] %d files in %s@%s",
        len(hf_files),
        HF_MODEL_REPO,
        HF_REVISION,
    )

    for seed in seeds:
        _stage_one(seed=seed, output_dir=output_dir, hf_files=hf_files)

    logger.info("[phase=done] all %d em_turner cells staged at %s", len(seeds), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
