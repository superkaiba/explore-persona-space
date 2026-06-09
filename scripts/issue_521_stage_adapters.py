#!/usr/bin/env python3
"""Stage the six #519 LoRA adapters from HF Hub into the local layout
``eval_results/issue_521/{arm}_seed{S}/adapter/`` (v2 M2 path).

The dispatcher's ``phase_c_extract_shifts`` resolves the adapter path
as ``output_dir / f"{arm}_seed{seed}" / "adapter"``. Passing
``--output-dir eval_results/issue_521`` to the dispatcher therefore
expects the adapters under that exact subtree — no extra ``adapters/``
parent. Plan §4 Step 3 + §10 Reproducibility Card row "Adapter
local-stage path (v2 M2)".

The HF repo layout is ``superkaiba1/explore-persona-space@main`` with
files under ``issue_519/{arm}_seed{seed}/...``. We pull each cell into
a tmp dir then flatten the ``issue_519/...`` subtree into the local
target. Asserts ``adapter_config.json`` + ``adapter_model.safetensors``
present per cell post-stage (the v2 M2 fail-loud check).

Run::

    uv run python scripts/issue_521_stage_adapters.py \\
        --output-dir eval_results/issue_521 \\
        [--cells marker_seed42]       # smoke: single cell
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ARMS = ("marker", "em")
SEEDS = (42, 137, 256)
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_REVISION = "main"  # NOT the dispatcher git_commit `c46b8989...`; that
# SHA is the code-repo commit, not an HF revision. Plan §10 Reproducibility
# Card row "Body-cited HF commit" + §12 Assumption #1.


def _stage_one(
    *,
    arm: str,
    seed: int,
    output_dir: Path,
    tmp_root: Path,
) -> None:
    """Stage one adapter cell via per-file `hf_hub_download`.

    Rationale: `snapshot_download(allow_patterns=...)` on
    `superkaiba1/explore-persona-space` (42k+ files) silently returns 0
    files because the HF Hub repo_info siblings list is truncated past
    ~8k entries (memory `feedback_snapshot_download_siblings_truncation`).
    The adapter dirs sit in the truncated tail, so per-pattern downloads
    miss them. Per-file downloads still work because they don't go
    through siblings.
    """
    from huggingface_hub import hf_hub_download

    target = output_dir / f"{arm}_seed{seed}" / "adapter"
    cfg = target / "adapter_config.json"
    sft = target / "adapter_model.safetensors"
    if cfg.exists() and sft.exists():
        # Round-2 reviewer NIT: validate file sizes on the already-staged
        # path so a stale LFS pointer (~134 B) or partial previous stage
        # cannot silently bypass the fail-loud size check at the end of
        # this function. v1 returned on file-presence alone, which let
        # a previously-interrupted stage masquerade as "already done."
        cfg_sz = cfg.stat().st_size
        sft_sz = sft.stat().st_size
        if cfg_sz < 100 or sft_sz < 1024:
            raise RuntimeError(
                f"adapter files at {target} are already present but suspiciously "
                f"small (config={cfg_sz}B, safetensors={sft_sz}B). This looks "
                f"like a stale LFS pointer or a partial previous stage — "
                f"refusing to skip. Delete {target} and re-run to re-download."
            )
        logger.info(
            "[skip] %s_seed%d already staged at %s (cfg=%dB, safetensors=%.1fMB)",
            arm,
            seed,
            target,
            cfg_sz,
            sft_sz / 1e6,
        )
        return
    target.mkdir(parents=True, exist_ok=True)

    # The known per-cell file layout in the HF repo (10 files per cell;
    # verified at planning time via the planner's Hub-API check). Pinned
    # explicitly to bypass the siblings-truncation bug.
    expected_files = (
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    logger.info(
        "[phase=stage] %s_seed%d: hf_hub_download per-file into %s",
        arm,
        seed,
        target,
    )
    for fname in expected_files:
        hf_path = f"issue_519/{arm}_seed{seed}/{fname}"
        try:
            local = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=hf_path,
                revision=HF_REVISION,
            )
        except Exception as e:
            # README.md may be optional; everything else is mandatory.
            if fname == "README.md":
                logger.warning(
                    "[stage] %s_seed%d: optional %s missing on HF: %s",
                    arm,
                    seed,
                    hf_path,
                    e,
                )
                continue
            raise RuntimeError(
                f"hf_hub_download failed for {HF_MODEL_REPO}@{HF_REVISION} :: "
                f"{hf_path}: {type(e).__name__}: {e}"
            ) from e
        dest = target / fname
        if dest.exists():
            dest.unlink()
        # hf cache stores the real file at a hashed path; copy (not move
        # — the cache may be shared) into the target dir.
        shutil.copy2(local, dest)

    # v2 M2 fail-loud: assert adapter files at the resolver-expected path.
    cfg = target / "adapter_config.json"
    sft = target / "adapter_model.safetensors"
    if not cfg.exists():
        raise RuntimeError(
            f"adapter_config.json missing at {cfg} after snapshot_download — "
            f"phase_c_extract_shifts will fail to load this adapter (v2 M2)."
        )
    if not sft.exists():
        raise RuntimeError(f"adapter_model.safetensors missing at {sft}.")
    cfg_sz = cfg.stat().st_size
    sft_sz = sft.stat().st_size
    if cfg_sz < 100 or sft_sz < 1024:
        raise RuntimeError(
            f"adapter files at {target} are suspiciously small "
            f"(config={cfg_sz}B, safetensors={sft_sz}B) — likely an LFS "
            f"pointer file, not the real adapter."
        )
    logger.info(
        "[stage_done] %s_seed%d at %s (cfg=%dB, safetensors=%.1fMB)",
        arm,
        seed,
        target,
        cfg_sz,
        sft_sz / 1e6,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage #519 LoRA adapters for #521 Phase C",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="eval_results/issue_521",
        help="Top-level output dir; adapters land under {arm}_seed{S}/adapter/.",
    )
    p.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help=(
            "Override cell list (default = all 6). Format: `marker_seed42 em_seed42 ...`. "
            "Use a single cell for smoke."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    # Resolve repo root.
    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # `tmp_root` was used by the old snapshot_download path; the
    # per-file `hf_hub_download` path uses the HF cache directly, so
    # the placeholder argument is harmless but unused.
    tmp_root = output_dir / "_hf_tmp_unused"

    if args.cells:
        cells: list[tuple[str, int]] = []
        for spec in args.cells:
            arm, _, rest = spec.partition("_seed")
            try:
                seed = int(rest)
            except ValueError as e:
                raise ValueError(
                    f"--cells spec {spec!r} must look like 'marker_seed42' / 'em_seed137'"
                ) from e
            if arm not in ARMS:
                raise ValueError(f"--cells: unknown arm {arm!r} (expected one of {ARMS})")
            cells.append((arm, seed))
    else:
        cells = [(a, s) for a in ARMS for s in SEEDS]
    logger.info("staging %d cells: %s", len(cells), cells)

    for arm, seed in cells:
        _stage_one(arm=arm, seed=seed, output_dir=output_dir, tmp_root=tmp_root)
    # No tmp directory to clean — per-file `hf_hub_download` writes into
    # the HF cache only; `shutil.copy2` lands the file directly at the
    # target path.
    logger.info("[phase=done] all %d cells staged at %s", len(cells), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
