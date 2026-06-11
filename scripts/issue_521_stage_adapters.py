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

#561 follow-up ``exposure-matched-ckpt300`` (plan v2 §4.0.1): pass
``--hf-path-template`` + ``--hf-revision`` to stage a per-step
CHECKPOINT directory instead of a final adapter, e.g.::

    uv run python scripts/issue_521_stage_adapters.py \\
        --output-dir eval_results/issue_561/exposure-matched-ckpt300 \\
        --cells marker_seed42 marker_seed137 marker_seed256 \\
        --hf-path-template 'issue_561_posonly/{arm}_seed{seed}/checkpoints/checkpoint-300' \\
        --hf-revision c6a4771980ff4f7ff960ae7cd620dcca58668fec

In template mode (any non-default ``--hf-path-template``) the mandatory
file set shrinks to the 3-file checkpoint shape — ``adapter_config.json``
+ ``adapter_model.safetensors`` + ``trainer_state.json`` (the last is
mandatory for the driver's ``global_step`` provenance assert). Tokenizer
files are NOT expected: checkpoint dirs carry only those 3 files
(Hub-verified at the pin), the extraction loads its tokenizer from
``--base-model-id``, and ``PeftModel.from_pretrained`` needs only the
config + safetensors pair. The defaults preserve the original #519
staging behavior verbatim.
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

# Default per-cell path template on the HF repo (the original #519 layout).
# Any NON-default template switches to "checkpoint mode": the mandatory
# file set shrinks to the 3-file checkpoint shape below (#561 follow-up
# `exposure-matched-ckpt300`, plan v2 §4.0.1).
DEFAULT_HF_PATH_TEMPLATE = "issue_519/{arm}_seed{seed}"

# The known final-adapter file layout in the HF repo (10 files per cell;
# verified at #521 planning time via the planner's Hub-API check). Pinned
# explicitly to bypass the siblings-truncation bug.
DEFAULT_EXPECTED_FILES = (
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
# Per-step checkpoint dirs carry exactly these 3 files (Hub-verified at the
# #561 pin). trainer_state.json is mandatory: the ckpt-300 driver's
# provenance assert reads `global_step` from it before any extraction.
CHECKPOINT_EXPECTED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "trainer_state.json",
)


def _stage_one(
    *,
    arm: str,
    seed: int,
    output_dir: Path,
    tmp_root: Path,
    hf_path_template: str = DEFAULT_HF_PATH_TEMPLATE,
    hf_revision: str = HF_REVISION,
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

    checkpoint_mode = hf_path_template != DEFAULT_HF_PATH_TEMPLATE
    expected_files = CHECKPOINT_EXPECTED_FILES if checkpoint_mode else DEFAULT_EXPECTED_FILES
    optional_files = set() if checkpoint_mode else {"README.md"}
    mandatory_files = tuple(f for f in expected_files if f not in optional_files)
    hf_cell_prefix = hf_path_template.format(arm=arm, seed=seed)

    target = output_dir / f"{arm}_seed{seed}" / "adapter"
    cfg = target / "adapter_config.json"
    sft = target / "adapter_model.safetensors"
    if all((target / f).exists() for f in mandatory_files):
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

    logger.info(
        "[phase=stage] %s_seed%d: hf_hub_download per-file from %s@%s into %s%s",
        arm,
        seed,
        hf_cell_prefix,
        hf_revision,
        target,
        " (checkpoint mode)" if checkpoint_mode else "",
    )
    for fname in expected_files:
        hf_path = f"{hf_cell_prefix}/{fname}"
        try:
            local = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=hf_path,
                revision=hf_revision,
            )
        except Exception as e:
            # README.md may be optional (default mode only); everything
            # else is mandatory.
            if fname in optional_files:
                logger.warning(
                    "[stage] %s_seed%d: optional %s missing on HF: %s",
                    arm,
                    seed,
                    hf_path,
                    e,
                )
                continue
            raise RuntimeError(
                f"hf_hub_download failed for {HF_MODEL_REPO}@{hf_revision} :: "
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
    if checkpoint_mode:
        ts = target / "trainer_state.json"
        if not ts.exists() or ts.stat().st_size < 10:
            raise RuntimeError(
                f"trainer_state.json missing/empty at {ts} — checkpoint mode "
                f"requires it for the driver's global_step provenance assert."
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
    p.add_argument(
        "--hf-path-template",
        default=DEFAULT_HF_PATH_TEMPLATE,
        help=(
            "Per-cell path template on the HF repo, formatted with {arm} and {seed}. "
            "Any NON-default value switches to checkpoint mode: the mandatory file "
            "set becomes adapter_config.json + adapter_model.safetensors + "
            "trainer_state.json (no tokenizer files). Default preserves the "
            "original #519 staging behavior verbatim."
        ),
    )
    p.add_argument(
        "--hf-revision",
        default=HF_REVISION,
        help="HF repo revision (commit SHA / branch) to download from. Default: main.",
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
        _stage_one(
            arm=arm,
            seed=seed,
            output_dir=output_dir,
            tmp_root=tmp_root,
            hf_path_template=args.hf_path_template,
            hf_revision=args.hf_revision,
        )
    # No tmp directory to clean — per-file `hf_hub_download` writes into
    # the HF cache only; `shutil.copy2` lands the file directly at the
    # target path.
    logger.info("[phase=done] all %d cells staged at %s", len(cells), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
