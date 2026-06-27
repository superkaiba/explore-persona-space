#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# (math/scientific notation — Δv, ŵ, σ, ρ, g₁ — intentional in docstrings + labels)
"""Issue #683 — g₁-vs-g_real tracking read (CPU, off-pod, 0 GPU).

The A7 precondition FAILED for both behaviors (the off-source write is
low-rank, not strictly rank-1), so the key-ablation scorer scored the keys
against the dominant SVD component ``g₁`` (``scored_dv ==
lowrank_dominant_component``) rather than the scalar ``g_real``. The clean-
result body claims the dominant component "tracks g_real tightly" — this
script SUBSTANTIATES (or refutes) that claim by computing, per source bank,
the held-out Spearman of the dominant-component projection ``g₁(C')`` against
the scalar realized gate ``g_real(C')``.

Definitions (mirroring scripts/issue683_a7_precondition.py exactly):
  - Held-out targets = every context EXCEPT the source's own condition
    (g_real(C)=1 is the diagonal, never a held-out target).
  - ΔV = [Δv(C'_1) … Δv(C'_n)]  (H × n_targets), stacked over held-out targets.
  - u₁ = top LEFT singular vector of ΔV (the A7 dominant SVD direction).
  - g₁(C'_i) = ⟨Δv(C'_i), u₁_signed⟩, where u₁_signed = sign(⟨u₁, ŵ⟩)·u₁ so
    the projection polarizes the same way as the source write ŵ (and hence as
    g_real, whose denominator ⟨ŵ,ŵ⟩ > 0). The sign flip is monotone, so it
    does not change |Spearman|, only its sign — but it makes a POSITIVE ρ the
    "tracks" outcome unambiguously.
  - g_real(C'_i) = the bank's stored scalar gate ⟨ŵ, Δv(C'_i)⟩ / ⟨ŵ, ŵ⟩.
  - g1_vs_greal_spearman = spearmanr(g₁, g_real) over the held-out targets.

This is the dominant-component read corresponding to scored_dv ==
lowrank_dominant_component; it does NOT re-fit any key. 0 GPU — reads the
committed Δv banks only.

Output: writes a ``g1_vs_greal`` block into each behavior's existing
``eval_results/issue_683/<behavior>/a7_precondition.json`` (adds a
``g1_vs_greal_spearman`` field to each ``per_bank`` entry + a top-level
``g1_vs_greal`` summary), preserving every existing field.

CLI:
    uv run python scripts/issue683_compute_g1_tracking.py --behavior marker
    uv run python scripts/issue683_compute_g1_tracking.py --behavior sycophancy
    uv run python scripts/issue683_compute_g1_tracking.py --behavior both
"""

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_compute_g1_tracking")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_683 import DEFAULT_LAYER  # noqa: E402


def _g1_tracking_for_bank(payload: dict) -> dict:
    """g₁(C') vs g_real(C') held-out Spearman for one source/seed Δv bank."""
    import torch
    from scipy.stats import spearmanr

    source = payload["source"]
    per_context = payload["per_context"]
    g_real = payload["g_real"]
    w_hat = torch.as_tensor(payload["w_hat"]).flatten().double()

    # Held-out targets = every context EXCEPT the source's own condition,
    # in a fixed order so the stack columns align with g_real lookups.
    targets = [c for c in per_context if c != source]
    if not targets:
        raise ValueError(f"bank for source {source!r} has no held-out targets")

    delta_stack = []
    g_real_vals = []
    kept_targets = []
    for c in targets:
        g = g_real.get(c)
        if g is None or g != g:  # skip NaN/missing g_real (kept identical to A7)
            continue
        dv = torch.as_tensor(per_context[c]["Delta_v"]).flatten().double()
        delta_stack.append(dv)
        g_real_vals.append(float(g))
        kept_targets.append(c)

    n = len(kept_targets)
    if n < 3:
        return {
            "source": source,
            "seed": payload.get("seed"),
            "n_targets": n,
            "g1_vs_greal_spearman": float("nan"),
            "note": "fewer than 3 held-out targets with finite g_real — Spearman undefined",
        }

    dv_matrix = torch.stack(delta_stack, dim=1).double()  # (H, n_targets)
    # Top LEFT singular vector u₁ (the A7 dominant direction).
    u, _s, _vt = torch.linalg.svd(dv_matrix, full_matrices=False)
    u1 = u[:, 0]
    # Sign-align u₁ to ŵ so g₁ polarizes like g_real (monotone flip; |ρ| unchanged).
    sgn = 1.0 if float(u1 @ w_hat) >= 0 else -1.0
    u1_signed = sgn * u1

    # g₁(C'_i) = <Δv(C'_i), u₁_signed>  (column-wise projection onto the dominant dir).
    g1 = (dv_matrix.transpose(0, 1) @ u1_signed).tolist()  # (n_targets,)

    rho, _p = spearmanr(g1, g_real_vals)
    return {
        "source": source,
        "seed": payload.get("seed"),
        "n_targets": n,
        "u1_dot_what_sign": sgn,
        "g1_vs_greal_spearman": float(rho),
    }


def _run_behavior(behavior: str) -> dict:
    import torch

    dv_dir = PROJECT_ROOT / "eval_results/issue_683/analysis_tensors/dv" / behavior
    a7_path = PROJECT_ROOT / "eval_results/issue_683" / behavior / "a7_precondition.json"
    if not a7_path.exists():
        raise FileNotFoundError(f"a7_precondition.json missing for {behavior}: {a7_path}")

    a7 = json.loads(a7_path.read_text())
    bank_paths = sorted(dv_dir.glob("*_L*.pt"))
    if not bank_paths:
        raise FileNotFoundError(f"no Δv bank .pt files under {dv_dir}")

    # Index existing per_bank entries by (source, seed) so we merge in-place.
    by_key = {(b["source"], b.get("seed")): b for b in a7["per_bank"]}

    reads = []
    for p in bank_paths:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        r = _g1_tracking_for_bank(payload)
        reads.append(r)
        key = (r["source"], r["seed"])
        if key in by_key:
            by_key[key]["g1_vs_greal_spearman"] = r["g1_vs_greal_spearman"]
            if "u1_dot_what_sign" in r:
                by_key[key]["g1_vs_greal_u1_dot_what_sign"] = r["u1_dot_what_sign"]
        logger.info(
            "[phase=g1_track] behavior=%s source=%s seed=%s n=%d g1_vs_greal_spearman=%.4f",
            behavior,
            r["source"],
            r["seed"],
            r["n_targets"],
            r["g1_vs_greal_spearman"],
        )

    finite = [
        r["g1_vs_greal_spearman"]
        for r in reads
        if r["g1_vs_greal_spearman"] == r["g1_vs_greal_spearman"]
    ]
    summary = {
        "definition": (
            "spearman(g1(C'), g_real(C')) over held-out targets, "
            "g1(C')=<Delta_v(C'), u1_signed>, u1=top-left-SVD-dir of [Delta_v(C')], "
            "u1_signed sign-aligned to w_hat; g_real(C')=<w_hat,Delta_v(C')>/<w_hat,w_hat>"
        ),
        "layer": DEFAULT_LAYER[behavior],
        "n_banks": len(reads),
        "per_bank_spearman": {f"{r['source']}": r["g1_vs_greal_spearman"] for r in reads},
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "median": float(sorted(finite)[len(finite) // 2]) if finite else None,
    }
    a7["g1_vs_greal"] = summary
    a7_path.write_text(json.dumps(a7, indent=2))
    logger.info(
        "[phase=g1_done] behavior=%s range=[%.3f, %.3f] -> %s",
        behavior,
        summary["min"] if summary["min"] is not None else float("nan"),
        summary["max"] if summary["max"] is not None else float("nan"),
        a7_path,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--behavior", required=True, choices=("marker", "sycophancy", "both"))
    args = ap.parse_args(argv)

    behaviors = ("marker", "sycophancy") if args.behavior == "both" else (args.behavior,)
    for b in behaviors:
        _run_behavior(b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
