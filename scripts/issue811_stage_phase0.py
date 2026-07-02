#!/usr/bin/env python3
"""Issue #811 — stage a COMPLETED phase-0 base-leg store from HF + verify completeness.

The att-20260701-233116 production run completed the phase-0 base leg cleanly (3
behaviors x 16 sources x 30 targets @ L14) and the crash's EXIT trap uploaded the
FULL store (incl. every per-source ``.done`` sentinel) to the HF data repo under
``issue811_partial/att-.../eval_results_issue_811/phase0_base_leg``. Re-extracting the
base leg costs ~5.6h wall on the L4 lane — this stages the uploaded store instead.

Contract (called by ``issue811_dispatch.sh`` when ``EPM_PHASE0_STAGE_PREFIX`` is set):
  1. ``snapshot_download`` the ``--hf-prefix`` subtree into ``--out`` ($PHASE0_DIR),
     flattening the prefix so the result is exactly the phase-0 store layout
     ``{behavior}/{source}_seed42/{target}_L{layer}.npz`` (+ per-source ``.done``).
  2. VERIFY completeness for THIS run's resolved grid (``--sources-spec`` +
     ``--primary-layer`` [+ optional ``--targets``]): each resolved (behavior, source)
     cell dir has its atomic ``.done`` sentinel AND EVERY target npz the ``.done``
     DECLARES at the primary layer (production, ``--targets`` absent — the declared
     ``targets`` list is the ground truth, so a partial HF recovery of ``.done`` +
     a subset of its npz FAILS) OR exactly the requested ``--targets`` subset (smoke).
     FAIL LOUD (``RuntimeError`` → exit 1) naming every missing target npz on a
     shortfall, and on an unparsable / ``targets``-less ``.done``.

Why fail-loud on a shortfall rather than "top up the missing cells": the phase-0
extractor (``issue811_phase0_extract.py``) has NO per-cell resume-skip — it re-runs
every cell it is handed — so a partial stage cannot be safely topped up by falling
through to extraction (that would re-extract ALL cells and clobber the staged good
ones). And stamping a trusted-complete flag over a partial dir without a complement
check is banned. So a shortfall is a hard fail; the operator re-stages a complete
prefix or unsets the env var to extract from scratch.

Usage:
    uv run python scripts/issue811_stage_phase0.py \
        --hf-prefix issue811_partial/att-20260701-233116/eval_results_issue_811/phase0_base_leg \
        --out eval_results/issue_811/phase0_base_leg \
        --primary-layer 14 \
        --sources-spec "em=binst_em,default;sycophancy=...;fact=..." \
        [--targets default,sp_swe,sp_doctor]   # smoke: verify only this target subset
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# uv run python does NOT auto-load .env; snapshot_download below needs HF_TOKEN.
# Project wrapper (analysis-phase script; shell exports also cover pod/GCE/SLURM).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue811.stage_phase0")

REF_REPO = "superkaiba1/explore-persona-space-data"
CELL_DONE_SENTINEL = ".done"


def _parse_sources_spec(spec: str) -> dict[str, list[str]]:
    """Parse ``behavior=src1,src2;behavior2=src3,...`` into ``{behavior: [sources]}``.

    A trailing ``;`` is tolerated. Raises ``ValueError`` on a malformed entry so a
    dispatcher typo fails loud rather than silently verifying an empty grid.
    """
    out: dict[str, list[str]] = {}
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"malformed --sources-spec entry {entry!r} (expected behavior=srcs)")
        beh, srcs = entry.split("=", 1)
        beh = beh.strip()
        src_list = [s.strip() for s in srcs.split(",") if s.strip()]
        if not beh or not src_list:
            raise ValueError(f"empty behavior or sources in --sources-spec entry {entry!r}")
        out[beh] = src_list
    if not out:
        raise ValueError(f"--sources-spec parsed to an empty grid: {spec!r}")
    return out


def _expected_targets_for_cell(cell_dir: Path, layer: int, targets: list[str] | None) -> list[str]:
    """The target cids a complete cell dir must carry at ``layer``.

    Production (``targets`` is None): the ``targets`` list DECLARED in the cell's
    atomic ``.done`` sentinel — the ground-truth grid the phase-0 extractor wrote
    (``issue811_phase0_extract.py`` ~line 296) when the cell completed. The
    completeness check then requires EVERY declared ``{target}_L{layer}.npz`` to be
    present, so a PARTIAL HF recovery (``.done`` + a subset of its declared target
    npz) FAILS LOUD here instead of PASSing on a "≥1 npz present" heuristic and
    skipping the ~5.6h re-extraction. Deriving "expected" from the files that
    happen to be present would make the check circular — any subset would pass. An
    unparsable ``.done``, or one lacking a ``targets`` key, is itself a shortfall
    (a corrupt/legacy sentinel we cannot trust as complete). Smoke (``targets``
    given): exactly the requested target subset must be present.

    Raises ``RuntimeError`` (naming the cell dir) on an unparsable ``.done`` or one
    without a non-empty ``targets`` list, so ``_verify_complete`` fails loud.
    """
    if targets is not None:
        return list(targets)
    # Production: the declared target grid is the source of truth, NOT the files
    # present (which a partial recovery would let pass vacuously).
    done_path = cell_dir / CELL_DONE_SENTINEL
    try:
        meta = json.loads(done_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"unparsable .done sentinel at {done_path} ({exc}) — cannot trust the "
            "staged cell as complete"
        ) from exc
    declared = meta.get("targets")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError(
            f".done sentinel at {done_path} has no non-empty 'targets' list "
            f"(got {declared!r}) — cannot verify staged completeness"
        )
    return [str(t) for t in declared]


def _verify_complete(
    out_root: Path,
    grid: dict[str, list[str]],
    layer: int,
    targets: list[str] | None,
) -> dict[str, tuple[int, int]]:
    """Assert every resolved (behavior, source) cell is complete; raise listing gaps.

    In production mode (``targets`` is None) the "expected" per-cell target set is the
    grid DECLARED in each cell's ``.done`` sentinel (via ``_expected_targets_for_cell``)
    — NOT the files that happen to be present — so a partial HF recovery (``.done`` + a
    subset of its declared target npz) is a shortfall listed here, and EVERY missing
    ``{target}_L{layer}.npz`` is named by name in the raised message. An unparsable /
    ``targets``-less ``.done`` raises inside ``_expected_targets_for_cell``.

    Returns a per-behavior ``{behavior: (n_declared_or_requested, n_present)}`` count
    digest on success (for the caller's completeness log).
    """
    missing: list[str] = []
    per_behavior: dict[str, tuple[int, int]] = {}
    for behavior, sources in grid.items():
        declared_total = present_total = 0
        for source in sources:
            cell_dir = out_root / behavior / f"{source}_seed{42}"
            if not cell_dir.is_dir():
                missing.append(f"{behavior}/{source}: cell dir absent")
                continue
            if not (cell_dir / CELL_DONE_SENTINEL).is_file():
                missing.append(f"{behavior}/{source}: no .done sentinel (partial/truncated cell)")
                continue
            # Production: raises loud on an unparsable / targets-less .done.
            expected = _expected_targets_for_cell(cell_dir, layer, targets)
            if not expected:
                missing.append(f"{behavior}/{source}: zero target npz at L{layer}")
                continue
            declared_total += len(expected)
            for tgt in expected:
                if (cell_dir / f"{tgt}_L{layer}.npz").is_file():
                    present_total += 1
                else:
                    missing.append(f"{behavior}/{source}: target {tgt}_L{layer}.npz missing")
        per_behavior[behavior] = (declared_total, present_total)
    if missing:
        raise RuntimeError(
            "staged phase-0 store INCOMPLETE for this run's grid "
            f"({len(missing)} missing cell/target(s)):\n  " + "\n  ".join(missing) + "\n"
            "The phase-0 extractor has no per-cell resume-skip, so a partial stage "
            "cannot be topped up safely — re-stage a COMPLETE HF prefix or unset "
            "EPM_PHASE0_STAGE_PREFIX to re-extract from scratch."
        )
    return per_behavior


def stage_phase0(
    hf_prefix: str,
    out: Path,
    primary_layer: int,
    sources_spec: str,
    targets: list[str] | None = None,
    *,
    repo: str = REF_REPO,
) -> int:
    """Download the phase-0 store subtree from HF into ``out`` + verify completeness."""
    from huggingface_hub import snapshot_download

    grid = _parse_sources_spec(sources_spec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    hf_prefix = hf_prefix.rstrip("/")
    logger.info(
        "staging phase-0 store: repo=%s prefix=%s -> %s (grid=%s, primary_layer=%d, targets=%s)",
        repo,
        hf_prefix,
        out,
        {b: len(s) for b, s in grid.items()},
        primary_layer,
        targets if targets is not None else "<all present>",
    )
    # snapshot_download into a scratch dir, then flatten the prefix into $PHASE0_DIR
    # (the store layout the dispatcher's parity + gate read is prefix-free).
    scratch = out.parent / ".phase0_stage_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    snapshot_download(
        repo,
        repo_type="dataset",
        revision="main",
        allow_patterns=[f"{hf_prefix}/**"],
        local_dir=str(scratch),
    )
    staged_root = scratch / hf_prefix
    if not staged_root.is_dir():
        raise RuntimeError(
            f"HF prefix {hf_prefix!r} produced no files under {staged_root} — wrong prefix?"
        )
    # CLEAR the destination for every behavior in THIS run's grid BEFORE merging, so
    # a stale local ``out/{behavior}`` subtree left from a prior run can never mask a
    # behavior that is MISSING from the staged prefix: a grid behavior absent from the
    # prefix now has an absent destination dir and surfaces as "cell dir absent" in
    # ``_verify_complete``, rather than silently reusing stale local cells. (A behavior
    # PRESENT in the prefix is re-cleared below and replaced with the fresh subtree.)
    for behavior in grid:
        dest_beh = out / behavior
        if dest_beh.exists():
            shutil.rmtree(dest_beh)
    # Move each staged behavior subtree into $PHASE0_DIR (replace any duplicate).
    n_copied = 0
    for beh_dir in sorted(p for p in staged_root.iterdir() if p.is_dir()):
        dest_beh = out / beh_dir.name
        if dest_beh.exists():
            shutil.rmtree(dest_beh)
        shutil.move(str(beh_dir), str(dest_beh))
        n_copied += 1
    shutil.rmtree(scratch, ignore_errors=True)
    logger.info("staged %d behavior subtrees into %s", n_copied, out)

    per_behavior = _verify_complete(out, grid, primary_layer, targets)
    digest = ", ".join(
        f"{beh}: {present}/{declared} target npz present"
        for beh, (declared, present) in sorted(per_behavior.items())
    )
    logger.info(
        "phase-0 store COMPLETE for grid %s at L%d [%s] — dispatcher may skip re-extraction",
        {b: len(s) for b, s in grid.items()},
        primary_layer,
        digest,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 stage completed phase-0 store from HF")
    ap.add_argument("--hf-prefix", required=True, help="HF dataset prefix of the completed store")
    ap.add_argument("--out", type=Path, required=True, help="$PHASE0_DIR to stage into")
    ap.add_argument("--primary-layer", type=int, default=14)
    ap.add_argument(
        "--sources-spec",
        required=True,
        help="behavior=src1,src2;behavior2=... — the run's resolved (behavior x source) grid",
    )
    ap.add_argument(
        "--targets",
        default=None,
        help="smoke: comma-separated target subset to require (default: all present per cell)",
    )
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets else None
    return stage_phase0(
        args.hf_prefix,
        args.out,
        args.primary_layer,
        args.sources_spec,
        targets=targets,
    )


if __name__ == "__main__":
    raise SystemExit(main())
