#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (v0, →, ×, ρ) in scientific docstrings + logs.
"""Issue #811 — re-extracted-mean-reproduces-#667 parity check (plan §13 check 1).

The #811 position-aware reader is a CODE CHANGE to the teacher-force extractor, so
the mean-vs-turn_nl comparison must NOT be confounded by drift between #667's
extraction environment and this run's. Plan §13 check 1 asserts the re-extracted
``mean`` ``v0`` reproduces #667's STORED ``v0`` on a few cells within float
tolerance — a FAIL-LOUD guard, run BEFORE the upload (so a drifted store never
propagates). It is framed in the plan as "run before the sweep, NOT a decision
gate", but it MUST hard-fail on real drift (otherwise the parent's ``mean``
reference is silently invalidated) — a drift is a CODE bug, so a FAIL routes to
``failure_class: code`` (round-2 CONCERN mean-parity-check-not-wired).

What it compares: the base-leg ``v0`` (mean-over-response, base θ0) freshly
re-extracted into the Phase-0 store (``phase0_base_leg/{behavior}/{source}_seed42/
{target}_L{layer}.npz`` key ``v0``) against #667's stored ``v0`` at the SAME
(behavior, source, target, layer) cell. Both are the base-model mean-over-response
residual — the SAME quantity — so a valid re-extraction reproduces #667 up to
bf16-vs-stored-fp32 rounding. Tolerance is a per-cell COSINE floor (default 0.999;
direction-preserving, robust to bf16 element rounding on a 3584-d vector) plus a
relative-L2 ceiling; either violated on any cell FAILS LOUD.

The Phase-0 store carries only ``v0`` (base mean) — it does NOT re-extract the
adapter leg — so the parity check is on the base ``mean`` summary only. ``turn_nl``
has no #667 counterpart to parity-check against (it does not exist in the #667
mean-only store), so this check is scoped to ``mean`` by construction (the plan's
"re-extracted-mean-reproduces-#667").

Usage (a few cells, run inline in the dispatcher AFTER the first Phase-0 cells land
and BEFORE upload):
    uv run python scripts/issue811_mean_parity_check.py \
        --phase0-root eval_results/issue_811/phase0_base_leg \
        --behavior em --layer 14 --n-cells 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# uv run python does NOT auto-load .env; the #667-store hf_hub_download needs
# HF_TOKEN. Project wrapper (analysis-phase script; shell exports also cover
# pod/GCE/SLURM).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue811.parity")

# #667's mean-only store (the reference the re-extracted mean must reproduce).
REF_REPO = "superkaiba1/explore-persona-space-data"
REF_PREFIX = "issue667_gate_chain_preview/analysis_tensors"

# Parity tolerances (plan §13 "within float tolerance"; no numeric atol was pinned
# in the plan, so grounded here on the re-extraction being bit-equivalent up to
# bf16-vs-fp32 rounding). Cosine is the primary direction-preserving floor; the
# relative-L2 ceiling additionally catches a magnitude drift a cosine misses.
# bf16 has ~3 decimal digits of mantissa, so per-element relative error is ~1e-2,
# but averaged over a 3584-d vector the direction (cosine) stays >0.999 and the
# aggregate relative-L2 stays well under 0.05 for a faithful re-extraction; a real
# reader-logic drift (wrong span, wrong layer) blows both far past these.
COS_FLOOR = 0.999
REL_L2_CEIL = 0.05


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / nb)


def _load_ref_v0(behavior: str, source_cid: str, target_cid: str, layer: int) -> np.ndarray:
    """#667's stored base-leg ``v0`` (mean) for one cell (the reference)."""
    from huggingface_hub import hf_hub_download

    rel = f"{REF_PREFIX}/{behavior}/{source_cid}_seed42/{target_cid}_L{layer}.npz"
    path = hf_hub_download(REF_REPO, rel, repo_type="dataset")
    d = np.load(path, allow_pickle=True)
    if "v0" not in d.files:
        raise KeyError(f"#667 ref {rel} has no 'v0' key; keys={sorted(d.files)}")
    return np.asarray(d["v0"], dtype=np.float64)


def _iter_phase0_cells(phase0_root: Path, behavior: str, layer: int, n_cells: int):
    """Yield up to ``n_cells`` (source_cid, target_cid, v0) from the Phase-0 store.

    Walks ``{phase0_root}/{behavior}/{source}_seed42/{target}_L{layer}.npz`` in
    sorted order (deterministic cell selection). Re-reads source/target from
    inside each file (robust to filename drift). Yields the base ``v0`` (mean).
    """
    beh_dir = phase0_root / behavior
    if not beh_dir.is_dir():
        raise FileNotFoundError(
            f"phase0 store dir missing: {beh_dir} — run Phase-0 extraction first"
        )
    yielded = 0
    for src_dir in sorted(p for p in beh_dir.iterdir() if p.is_dir()):
        for npz in sorted(src_dir.glob(f"*_L{layer}.npz")):
            d = np.load(npz, allow_pickle=True)
            if "v0" not in d.files:
                raise KeyError(f"phase0 {npz} has no 'v0' key; keys={sorted(d.files)}")
            src = str(np.asarray(d["source_cid"]).item())
            tgt = str(np.asarray(d["target_cid"]).item())
            yield src, tgt, np.asarray(d["v0"], dtype=np.float64)
            yielded += 1
            if yielded >= n_cells:
                return
    if yielded == 0:
        raise FileNotFoundError(
            f"no phase0 cells found under {beh_dir} for L{layer} — run Phase-0 first"
        )


def check_mean_parity(
    phase0_root: Path,
    behavior: str,
    layer: int,
    n_cells: int,
    *,
    cos_floor: float = COS_FLOOR,
    rel_l2_ceil: float = REL_L2_CEIL,
) -> list[dict]:
    """Assert re-extracted mean v0 reproduces #667's stored v0 on ``n_cells`` cells.

    Returns the per-cell comparison records. Raises ``RuntimeError`` (fail-loud) on
    the FIRST cell whose cosine < ``cos_floor`` OR whose relative-L2 > ``rel_l2_ceil``
    — a real drift means the position-aware reader changed the mean read, which
    invalidates the parent's mean reference (a CODE bug, plan §13 / §4.2).
    """
    recs: list[dict] = []
    for src, tgt, v0_new in _iter_phase0_cells(phase0_root, behavior, layer, n_cells):
        v0_ref = _load_ref_v0(behavior, src, tgt, layer)
        assert v0_new.shape == v0_ref.shape, (
            f"shape mismatch {src}->{tgt} L{layer}: new {v0_new.shape} vs ref {v0_ref.shape}"
        )
        cos = _cos(v0_new, v0_ref)
        rel = _rel_l2(v0_new, v0_ref)
        rec = {
            "source_cid": src,
            "target_cid": tgt,
            "layer": layer,
            "cosine": cos,
            "rel_l2": rel,
            "ok": cos >= cos_floor and rel <= rel_l2_ceil,
        }
        recs.append(rec)
        logger.info(
            "[parity] %s->%s L%d: cosine=%.6f rel_l2=%.6f %s",
            src,
            tgt,
            layer,
            cos,
            rel,
            "OK" if rec["ok"] else "DRIFT",
        )
        if not rec["ok"]:
            raise RuntimeError(
                f"MEAN-PARITY DRIFT {src}->{tgt} L{layer}: cosine={cos:.6f} "
                f"(floor {cos_floor}) rel_l2={rel:.6f} (ceil {rel_l2_ceil}) — the "
                f"re-extracted mean v0 does NOT reproduce #667's stored v0; the "
                f"position-aware reader changed the mean read (plan §13, failure_class: code)."
            )
    logger.info("[parity] PASS: %d cells reproduce #667's stored mean v0", len(recs))
    return recs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Issue #811 mean-parity check vs #667 (plan §13)")
    ap.add_argument(
        "--phase0-root",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_811/phase0_base_leg",
        help="the Phase-0 base-leg store root ({behavior}/{source}_seed42/{target}_L{layer}.npz)",
    )
    ap.add_argument("--behavior", default="em", help="behavior to parity-check (default: em)")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n-cells", type=int, default=3, help="number of cells to compare (2-3)")
    ap.add_argument("--cos-floor", type=float, default=COS_FLOOR)
    ap.add_argument("--rel-l2-ceil", type=float, default=REL_L2_CEIL)
    args = ap.parse_args()
    check_mean_parity(
        args.phase0_root,
        args.behavior,
        args.layer,
        args.n_cells,
        cos_floor=args.cos_floor,
        rel_l2_ceil=args.rel_l2_ceil,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
