#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# (math/scientific notation — Δv, ŵ, σ, ρ — intentional in docstrings + labels)
"""Issue #683 Phase B — A7 scalar-gated-write precondition read (CPU, off-pod).

Plan §4 Phase B. BEFORE any key fitting, decide whether the off-source write
is actually ONE direction scaled per target (the scalar g_real is a faithful
DV) or genuinely multi-directional (the scalar summary is invalid → low-rank
fallback). Computed per behavior from the {ŵ, Δv(C'_i)} banks the Phase-A
extractors wrote, exactly as the paper's A7 Testable block prescribes:

  1. Scalarity residual per target:
       ρ_i = Δv(C'_i) − ŵ·g_real(C'_i),   ‖ρ_i‖ / ‖Δv(C'_i)‖
     (small ⇒ the target's shift IS the source write scaled by g_real).
  2. Stacked SVD of ΔV = [Δv(C'_1) … Δv(C'_n)]:
       σ₁²/Σσ²   (energy in the top component; ≥ ~0.5 ⇒ rank-1-ish)
       σ₂/σ₁     (gap to the second component)
       cos(u₁, ŵ) (does the top singular direction align with the source write?)
  3. Branch verdict:
       rank-1 holds (σ₁²/Σσ² ≥ thresh AND median scalarity residual small)
         → the scalar g_real is the DV; the scorer scores keys against it.
       rank-1 fails → the scorer fits the low-rank fallback Δv ≈ Σ wⱼ gⱼ and
         scores keys against the dominant component(s).

The verdict is recorded as a FINDING regardless of outcome — the
marker-vs-sycophancy precondition contrast IS part of the headline (plan §0).

Input: the per-source / per-seed Δv banks under
``eval_results/issue_683/analysis_tensors/dv/<behavior>/``. The source's own
condition is EXCLUDED from the held-out target stack (g_real(C)=1 is the
diagonal, never a held-out target).

Output: ``eval_results/issue_683/<behavior>/a7_precondition.json``.

CLI:
    uv run python scripts/issue683_a7_precondition.py --behavior marker
    uv run python scripts/issue683_a7_precondition.py --behavior sycophancy
    # CPU smoke over a dv_smoke bank:
    uv run python scripts/issue683_a7_precondition.py --behavior marker \
        --dv-dir eval_results/issue_683/analysis_tensors/dv_smoke/marker \
        --out eval_results/issue_683/smoke/a7_marker_smoke.json --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_a7_precondition")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_683 import (  # noqa: E402
    DEFAULT_LAYER,
    cosine,
    repro_metadata,
)

RANK1_ENERGY_THRESHOLD = 0.5  # σ₁²/Σσ² ≥ this ⇒ rank-1-ish (plan §4)
SCALARITY_RESIDUAL_THRESHOLD = 0.5  # median ‖ρ‖/‖Δv‖ below this ⇒ scalar-faithful


def _load_dv_banks(dv_dir: Path) -> list[dict]:
    """Load every per-source / per-seed Δv bank .pt under ``dv_dir``."""
    import torch

    banks = []
    for p in sorted(dv_dir.glob("*_L*.pt")):
        banks.append({"path": p, "payload": torch.load(p, map_location="cpu", weights_only=False)})
    if not banks:
        raise FileNotFoundError(f"no Δv bank .pt files under {dv_dir}")
    return banks


def a7_read_for_bank(payload: dict) -> dict:
    """Scalarity residual + stacked-SVD spectrum for one source/seed Δv bank."""
    import numpy as np
    import torch

    source = payload["source"]
    per_context = payload["per_context"]
    g_real = payload["g_real"]
    w_hat = torch.as_tensor(payload["w_hat"]).flatten().double()
    w_norm = float(w_hat.norm())

    # Held-out targets = every context EXCEPT the source's own condition.
    targets = [c for c in per_context if c != source]
    if not targets:
        raise ValueError(f"bank for source {source!r} has no held-out targets")

    delta_stack = []
    scal_resid: dict[str, float] = {}
    for c in targets:
        dv = torch.as_tensor(per_context[c]["Delta_v"]).flatten().double()
        delta_stack.append(dv)
        g = g_real.get(c)
        dv_norm = float(dv.norm())
        if g is None or g != g or w_norm == 0 or dv_norm == 0:
            scal_resid[c] = float("nan")
            continue
        rho = dv - w_hat * float(g)
        scal_resid[c] = float(rho.norm()) / dv_norm

    dv_matrix = torch.stack(delta_stack, dim=1).double()  # (H, n_targets)
    # economy SVD; singular values descending.
    try:
        s = torch.linalg.svdvals(dv_matrix)
    except RuntimeError:
        # tiny / degenerate stacks (CPU smoke): fall back to numpy.
        s = torch.as_tensor(np.linalg.svd(dv_matrix.numpy(), compute_uv=False))
    s = s.double()
    s2 = s**2
    sigma1_sq_frac = float(s2[0] / s2.sum()) if float(s2.sum()) > 0 else float("nan")
    sigma2_over_1 = float(s[1] / s[0]) if s.numel() >= 2 and float(s[0]) > 0 else float("nan")

    # cos(u1, ŵ): the top LEFT singular vector vs the source write.
    cos_u1_what = float("nan")
    if w_norm > 0:
        try:
            u, _s, _vt = torch.linalg.svd(dv_matrix, full_matrices=False)
            u1 = u[:, 0]
            cos_u1_what = abs(cosine(u1, w_hat))
        except RuntimeError:
            cos_u1_what = float("nan")

    resid_vals = [v for v in scal_resid.values() if v == v]  # drop NaN
    median_resid = float(np.median(resid_vals)) if resid_vals else float("nan")

    rank1_holds = bool(
        sigma1_sq_frac == sigma1_sq_frac
        and sigma1_sq_frac >= RANK1_ENERGY_THRESHOLD
        and median_resid == median_resid
        and median_resid <= SCALARITY_RESIDUAL_THRESHOLD
    )
    return {
        "source": source,
        "seed": payload.get("seed"),
        "n_targets": len(targets),
        "w_hat_norm": w_norm,
        "scalarity_residual_per_target": scal_resid,
        "scalarity_residual_median": median_resid,
        "sigma1_sq_frac": sigma1_sq_frac,
        "sigma2_over_sigma1": sigma2_over_1,
        "cos_u1_what": cos_u1_what,
        "singular_values_top8": [float(x) for x in s[:8].tolist()],
        "rank1_holds": rank1_holds,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--behavior", required=True, choices=("marker", "sycophancy"))
    ap.add_argument("--dv-dir", default=None, help="default analysis_tensors/dv/<behavior>")
    ap.add_argument(
        "--layer", type=int, default=None, help="for metadata only; default per behavior"
    )
    ap.add_argument(
        "--out", default=None, help="default eval_results/issue_683/<behavior>/a7_precondition.json"
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    layer = args.layer if args.layer is not None else DEFAULT_LAYER[args.behavior]
    dv_dir = Path(
        args.dv_dir or (PROJECT_ROOT / "eval_results/issue_683/analysis_tensors/dv" / args.behavior)
    )
    out_path = Path(
        args.out
        or (PROJECT_ROOT / "eval_results/issue_683" / args.behavior / "a7_precondition.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[phase=a7_start] behavior=%s dv_dir=%s", args.behavior, dv_dir)
    banks = _load_dv_banks(dv_dir)
    reads = []
    for b in banks:
        r = a7_read_for_bank(b["payload"])
        r["bank_path"] = str(b["path"])
        reads.append(r)
        logger.info(
            "[phase=a7_bank] source=%s seed=%s n_targets=%d σ1²/Σ=%.3f median_resid=%.3f "
            "cos(u1,ŵ)=%.3f rank1_holds=%s",
            r["source"],
            r["seed"],
            r["n_targets"],
            r["sigma1_sq_frac"],
            r["scalarity_residual_median"],
            r["cos_u1_what"],
            r["rank1_holds"],
        )

    # Behavior-level verdict: rank-1 holds iff it holds for the majority of banks.
    n_hold = sum(1 for r in reads if r["rank1_holds"])
    behavior_rank1_holds = n_hold > len(reads) / 2
    payload = {
        "behavior": args.behavior,
        "layer": layer,
        "rank1_energy_threshold": RANK1_ENERGY_THRESHOLD,
        "scalarity_residual_threshold": SCALARITY_RESIDUAL_THRESHOLD,
        "n_banks": len(reads),
        "n_banks_rank1_holds": n_hold,
        "behavior_rank1_holds": behavior_rank1_holds,
        "verdict": ("rank1_scalar_gate" if behavior_rank1_holds else "low_rank_fallback_required"),
        "per_bank": reads,
        "reproducibility": repro_metadata({"behavior": args.behavior, "layer": layer}),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "[phase=a7_done] behavior=%s verdict=%s (%d/%d banks rank-1) -> %s",
        args.behavior,
        payload["verdict"],
        n_hold,
        len(reads),
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
