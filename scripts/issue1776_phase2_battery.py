"""#1776 Phase 2c(ii): operator battery J_last vs W_{M'} (plan §4 / §6.5).

Runs the ``issue1345_operator_comparison`` conventions on the slot-matched
operator pair — the averaged causal Jacobian J_last and the fitted ridge
comparator M' (``m_ridge_x50k``), both mapping cx_last(14) -> v(19):

  - raw operator cosine (flattened primal betas) + the random two-sided
    rotation chance band (``raw_cosine_with_rotation_null``),
  - Procrustes-aligned cosine = the closed-form two-sided operator-Procrustes
    optimum (``spectrum_cosine``; von Neumann bound), reported against the
    #825 base<->instruct calibration anchor 0.6864 (its own rotation null is
    degenerate by invariance — descriptive, per the #1345 metadata note),
  - singular-spectrum cosine rows (descriptive only).

All direction-aware reads are RESTRICTED to the on-support subspace: the
top-``--n-pcs`` input PCs of cx_last(14) over the J-pair corpus (a fitted map
is undefined off-support; a full-space comparison manufactures fake
disagreement — plan §4 2c(ii) measurement note). The shipped 963k M (input
space = layer 19) enters ONLY the rotation-invariant spectrum-cosine row,
labeled cross-slot reference — direction-aware comparisons against it are not
defined across input slots.

Writes the §6.5 primary deliverable ``phase2/operator_battery.json``.

CPU smoke (``--smoke``): planted identical-operator pair reads raw cosine
~1.0 far above the null band; an independent random operator reads inside it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Plan §10 symbol greps name these helpers; the module import also pulls the
# #825 map-alignment stack it builds on (verified importable by the smoke).
from issue1345_operator_comparison import (  # noqa: E402
    raw_cosine_with_rotation_null,
    spectrum_cosine,
)

# #825 base<->instruct Procrustes-aligned cosine — the calibration anchor the
# plan's "Procrustes-aligned cosine with the #825-calibrated null" names
# (plan § Instruments; a KNOWN-different-but-related operator pair's reading
# of the same statistic).
PROCRUSTES_CALIBRATION_ANCHOR_825 = 0.6864


def _primal_beta(payload: dict) -> torch.Tensor:
    """Ridge payload -> primal operator beta (H_in, H_out) on RAW inputs:
    pred = (x - xmu)/xsd @ W + b  =>  linear part = W / xsd[:, None]."""
    w = payload["W"].to(torch.float64)
    xsd = payload["xsd"].to(torch.float64)
    assert w.ndim == 2 and xsd.shape == (w.shape[0],), (w.shape, xsd.shape)
    return w / xsd[:, None]


def _j_primal_beta(obj) -> torch.Tensor:
    """J_last payload -> primal beta (H_in, H_out): v ~= J c  =>  beta = J.T."""
    j = (obj["J"] if isinstance(obj, dict) else obj).to(torch.float64)
    assert j.ndim == 2 and j.shape[0] == j.shape[1], (
        f"operator battery needs the FULL-RANK square J, got {tuple(j.shape)}"
    )
    return j.T.contiguous()


def _support_basis(acts: torch.Tensor, n_pcs: int) -> torch.Tensor:
    """Top-``n_pcs`` right-singular vectors of the CENTERED input pool (H, k)."""
    x = acts.to(torch.float64)
    assert x.ndim == 2 and x.shape[0] > n_pcs, (x.shape, n_pcs)
    xc = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    return vh[:n_pcs].T.contiguous()  # (H, n_pcs)


def battery(
    beta_j: torch.Tensor,
    beta_m: torch.Tensor,
    basis: torch.Tensor,
    *,
    n_draws: int,
    seed: int,
    beta_shipped: torch.Tensor | None = None,
) -> dict:
    """The §4 2c(ii) reads; every direction-aware row on-support-restricted."""
    bj_r = (basis.T @ beta_j).contiguous()  # (k, H_out)
    bm_r = (basis.T @ beta_m).contiguous()
    on_support_mass = {
        "J_last": float((bj_r.norm() ** 2 / beta_j.norm() ** 2).item()),
        "mprime_x50k": float((bm_r.norm() ** 2 / beta_m.norm() ** 2).item()),
    }
    raw = raw_cosine_with_rotation_null(bj_r, bm_r, n_draws=n_draws, seed=seed)
    spec_r = spectrum_cosine(bj_r, bm_r)
    out = {
        "pair": "J_last vs mprime_x50k (cx_last(14) -> v(19), on-support)",
        "restriction": {"n_pcs": int(basis.shape[1]), "on_support_frobenius_mass": on_support_mass},
        "raw_cosine_with_rotation_null": raw,
        "procrustes_aligned_cosine": {
            "value": spec_r,
            "definition": "two-sided operator-Procrustes optimum (spectrum_cosine, "
            "von Neumann bound); its own rotation null is degenerate by invariance",
            "calibration_anchor_825_base_instruct": PROCRUSTES_CALIBRATION_ANCHOR_825,
        },
        "spectrum_cosine_descriptive": {"J_last_vs_mprime_x50k_on_support": spec_r},
    }
    if beta_shipped is not None:
        out["spectrum_cosine_descriptive"]["J_last_vs_m_shipped_full_space"] = {
            "value": spectrum_cosine(beta_j, beta_shipped),
            "label": "cross-slot reference (shipped M input = layer 19; "
            "rotation-invariant row ONLY — direction-aware reads undefined across slots)",
        }
    return out


def _regime_inputs(args) -> dict:
    """Output-affecting regime keys — the fingerprint manifest the skip keys on
    (sibling convention: jacobian try_resume / phase3 manifest, #722 r3 rule)."""
    return {
        "jlast": str(args.jlast),
        "mprime_weights": str(args.mprime_weights),
        "acts14": str(args.acts14),
        "shipped_m": str(args.shipped_m) if args.shipped_m else None,
        "n_pcs": args.n_pcs,
        "n_draws": args.n_draws,
        "seed": args.seed,
    }


def run(args) -> int:
    # Idempotency (concern p2c-battery-no-idempotency-skip): skip when the
    # output exists with a MATCHING regime fingerprint; --force recomputes;
    # a mismatched/unreadable prior output is recomputed (never mixed).
    fp = _regime_inputs(args)
    if args.out.exists() and not args.force:
        try:
            prior = json.loads(args.out.read_text()).get("inputs")
        except (json.JSONDecodeError, OSError):
            prior = None
        if prior == fp:
            print(
                f"[phase2-battery] output exists with MATCHING fingerprint — skip "
                f"(resume; --force to recompute): {args.out}",
                flush=True,
            )
            return 0
        print(
            "[phase2-battery] output exists but fingerprint MISMATCH/unreadable -> recompute",
            flush=True,
        )
    j_obj = torch.load(args.jlast, map_location="cpu", weights_only=True)
    # weights_only=False: sha-pinned SELF/parent-produced ridge payloads whose
    # metadata carries non-primitives (the documented carve-out; #1073 entry).
    m_obj = torch.load(args.mprime_weights, map_location="cpu", weights_only=False)
    acts = torch.load(args.acts14, map_location="cpu", weights_only=True)
    beta_j = _j_primal_beta(j_obj)
    beta_m = _primal_beta(m_obj)
    assert beta_j.shape == beta_m.shape, (beta_j.shape, beta_m.shape)
    basis = _support_basis(acts, args.n_pcs)
    beta_shipped = None
    if args.shipped_m:
        beta_shipped = _primal_beta(
            torch.load(args.shipped_m, map_location="cpu", weights_only=False)
        )
    report = battery(
        beta_j,
        beta_m,
        basis,
        n_draws=args.n_draws,
        seed=args.seed,
        beta_shipped=beta_shipped,
    )
    report["inputs"] = fp  # the regime fingerprint the resume skip keys on
    report["repro"] = C76.repro_meta()
    C76.atomic_write_json(args.out, report)
    obs = report["raw_cosine_with_rotation_null"]
    print(
        f"[phase2-battery] [phase=battery_done] raw_cos={obs['raw_cosine']:.4f} "
        f"null_p975={obs['rotation_null']['null_p975']:.4f} "
        f"procrustes={report['procrustes_aligned_cosine']['value']:.4f} -> {args.out}",
        flush=True,
    )
    return 0


def smoke(args) -> int:
    """Planted-pair probe: identical operators >> null; independent ~ null."""
    rng = torch.Generator().manual_seed(0)
    h, n, k = 48, 300, 12
    w = torch.randn(h, h, generator=rng, dtype=torch.float64) / np.sqrt(h)
    acts = torch.randn(n, h, generator=rng)
    basis = _support_basis(acts, k)
    payload_m = {"W": w.to(torch.float32), "xsd": torch.ones(h)}
    beta_m = _primal_beta(payload_m)
    beta_j_same = beta_m.clone()  # identical operator (already primal-oriented)
    rep_same = battery(beta_j_same, beta_m, basis, n_draws=8, seed=0, beta_shipped=beta_m)
    raw_same = rep_same["raw_cosine_with_rotation_null"]
    assert raw_same["raw_cosine"] > 0.99, raw_same
    assert raw_same["raw_cosine"] > raw_same["rotation_null"]["null_p975"] + 0.5, raw_same
    assert abs(rep_same["procrustes_aligned_cosine"]["value"] - 1.0) < 1e-6, rep_same
    assert (
        rep_same["spectrum_cosine_descriptive"]["J_last_vs_m_shipped_full_space"]["value"] > 0.999
    )
    beta_ind = torch.randn(h, h, generator=rng, dtype=torch.float64) / np.sqrt(h)
    rep_ind = battery(beta_ind, beta_m, basis, n_draws=16, seed=1)
    raw_ind = rep_ind["raw_cosine_with_rotation_null"]
    band = 4 * max(raw_ind["rotation_null"]["null_std"], 1e-6)
    assert abs(raw_ind["raw_cosine"] - raw_ind["rotation_null"]["null_mean"]) < band, raw_ind
    # Degenerate gate: a sketch-shaped (non-square) J is refused loudly.
    try:
        _j_primal_beta({"J": torch.randn(5, h, generator=rng)})
        raise RuntimeError("non-square J must be refused")
    except AssertionError as e:
        assert "FULL-RANK" in str(e)
    # Idempotency leg (run()-level, real files): fresh run writes; matching
    # re-run SKIPS (mtime unchanged); --force recomputes; a regime-key change
    # (n_draws) recomputes (fingerprint MISMATCH branch).
    import tempfile

    with tempfile.TemporaryDirectory(prefix="i1776_battery_smoke.") as td:
        tdp = Path(td)
        torch.save({"J": w.to(torch.float32)}, tdp / "J_last.pt")
        torch.save(payload_m, tdp / "m.pt")
        torch.save(acts, tdp / "acts14.pt")
        rargs = argparse.Namespace(
            jlast=tdp / "J_last.pt",
            mprime_weights=tdp / "m.pt",
            acts14=tdp / "acts14.pt",
            shipped_m=None,
            n_pcs=4,
            n_draws=8,
            seed=0,
            out=tdp / "operator_battery.json",
            force=False,
        )
        assert run(rargs) == 0
        m1 = rargs.out.stat().st_mtime_ns
        assert run(rargs) == 0  # MATCH -> skip
        assert rargs.out.stat().st_mtime_ns == m1, "skip branch rewrote the output"
        rargs.force = True
        assert run(rargs) == 0  # --force -> recompute
        m2 = rargs.out.stat().st_mtime_ns
        assert m2 != m1, "--force did not recompute"
        rargs.force = False
        rargs.n_draws = 9
        assert run(rargs) == 0  # regime-key change -> MISMATCH recompute
        assert rargs.out.stat().st_mtime_ns != m2, "fingerprint mismatch did not recompute"
        assert json.loads(rargs.out.read_text())["inputs"]["n_draws"] == 9
    print(
        "[phase2-battery] [phase=smoke_done] PASS (identical >> null; independent in band; "
        "non-square-J refusal; resume skip/--force/mismatch exercised)",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jlast", type=Path, help="merged full-rank J_last.pt")
    ap.add_argument("--mprime-weights", type=Path, help="m_ridge_x50k.pt payload")
    ap.add_argument("--acts14", type=Path, help="J-pair cx_last(14) pool (n, H)")
    ap.add_argument("--shipped-m", type=Path, default=None, help="n1m L19 ridge (cross-slot row)")
    ap.add_argument("--n-pcs", type=int, default=256)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="recompute even if output exists")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke(args)
    for req in ("jlast", "mprime_weights", "acts14", "out"):
        assert getattr(args, req) is not None, f"--{req.replace('_', '-')} is required"
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
