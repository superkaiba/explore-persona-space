"""Issue #2479 round-4 diagnostics — answer-length-controlled recount (zero GPU).

Reads only the committed artifact ``eval_results/issue_2479/gradient_verdict.json``
and writes ``eval_results/issue_2479/r4_length_control.json``. Pure
counting/statistics — no story or answer text is read or persisted.

Question (screened not-redundant follow-up): does the headline association
(rank correlation +0.703 between the frozen AI-likeness axis and rung-4
recovery fraction, 16 characters) survive controlling for per-character mean
kept-answer length (the named open mediator: length tracks the axis at +0.53,
p = 0.017)?

Computes:
  (1) partial Spearman rho(axis, recovery | mean answer length) by the
      rank-residual method, with a 10,000-draw label-permutation null for the
      partial statistic (seed 0; the axis labels are permuted and the partial
      statistic is recomputed per draw; one-sided add-one p — the verdict
      script's convention);
  (2) a length-matched subsample recount: the largest contiguous window in
      sorted mean answer length whose max/min ratio is <= 1.15 (deterministic
      greedy caliper; ties broken by smaller ratio, then lower window start),
      with the plain rank correlation + permutation p recomputed on that
      subset — plus an IQR-band companion subset (lengths within [Q1, Q3]);
  (3) the zero-order pieces for context: axis-recovery (headline replay,
      asserted equal to the committed verdict), length-axis (replay, asserted
      equal to the committed ``answer_length_vs_axis`` secondary read), and
      length-recovery (computed here).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from issue2479_gradient_verdict import (  # noqa: E402
    _pearson_rows,
    _rankdata,
    _spearman,
    spearman_perm_read,
)

REPO = _HERE.parent
EVAL = REPO / "eval_results/issue_2479"
N_PERM = 10_000
SEED = 0
CALIPER_RATIO = 1.15


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def _residualize_rows(v_rows: np.ndarray, rz_c: np.ndarray, denom_z: float) -> np.ndarray:
    """OLS-residualize each row of ``v_rows`` (B, n) on the centered covariate ``rz_c`` (n,).

    Intercept + slope per row; returned residuals are mean-zero by construction.
    """
    v_c = v_rows - v_rows.mean(axis=1, keepdims=True)
    b = (v_c @ rz_c) / denom_z
    return v_c - b[:, None] * rz_c


def partial_spearman_perm_read(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_perm: int,
    seed: int,
    label: str,
) -> dict:
    """Partial Spearman rho(x, y | z) via rank residuals, with an x-label permutation null.

    Ranks all three vectors (average-tie ranks), OLS-residualizes rank(x) and
    rank(y) on rank(z), and takes the Pearson correlation of the residuals.
    The null permutes the x (axis) labels via a single (n_perm, n) index matrix
    — no Python loop over draws — re-residualizing each permuted draw on
    rank(z), so the PARTIAL statistic is recomputed per draw. One-sided
    add-one Monte-Carlo p: ``(1 + #{stat_null >= stat_obs}) / (n_perm + 1)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    assert x.shape == y.shape == z.shape, (x.shape, y.shape, z.shape)
    n = int(len(x))
    rx, ry, rz = _rankdata(x), _rankdata(y), _rankdata(z)
    assert np.ptp(rx) > 0 and np.ptp(ry) > 0 and np.ptp(rz) > 0, "zero rank variance"
    rz_c = rz - rz.mean()
    denom_z = float((rz_c**2).sum())
    assert denom_z > 0.0
    rx_res = _residualize_rows(rx[None, :], rz_c, denom_z)[0]
    ry_res = _residualize_rows(ry[None, :], rz_c, denom_z)[0]
    rho = float(_pearson_rows(rx_res[None, :], ry_res)[0])
    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((n_perm, n)), axis=1)
    null_rows = _residualize_rows(rx[perm_idx], rz_c, denom_z)
    null = _pearson_rows(null_rows, ry_res)
    n_ge = int((null >= rho).sum())
    return {
        "label": label,
        "n": n,
        "n_perm": int(n_perm),
        "seed": int(seed),
        "rho_partial": rho,
        "n_null_ge": n_ge,
        "p_add_one": float((1 + n_ge) / (n_perm + 1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_mean": float(null.mean()),
        "method": (
            "rank-residual partial Spearman: average-tie ranks of all three variables; "
            "rank(axis) and rank(recovery) OLS-residualized (intercept + slope) on "
            "rank(length); Pearson correlation of the residuals"
        ),
        "permutation": (
            "axis labels permuted (single vectorized index matrix); the partial statistic "
            "is re-residualized and recomputed per draw; one-sided add-one p, matching the "
            "verdict script's spearman_perm_read convention"
        ),
        "status": "ok",
    }


def largest_tight_window(
    lengths_by_name: dict[str, float], ratio_cap: float
) -> tuple[list[str], float]:
    """Largest contiguous window in sorted mean answer length with max/min <= ratio_cap.

    Deterministic: characters sorted by (length, name); ties between candidate
    windows broken by smaller realized max/min ratio, then lower window start.
    Returns (subset names in length order, realized max/min ratio).
    """
    items = sorted(lengths_by_name.items(), key=lambda kv: (kv[1], kv[0]))
    vals = [v for _, v in items]
    best: tuple[int, float, int] | None = None  # (size, ratio, start)
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] <= vals[i] * ratio_cap:
            j += 1
        size, ratio = j - i + 1, vals[j] / vals[i]
        if best is None or (size, -ratio, -i) > (best[0], -best[1], -best[2]):
            best = (size, ratio, i)
    assert best is not None
    size, ratio, i = best
    return [name for name, _ in items[i : i + size]], float(ratio)


def _subset_read(
    subset: list[str],
    pc: dict,
    all_names: list[str],
    *,
    rule: str,
    label: str,
) -> dict:
    """Plain rank-correlation recount (axis vs recovery) on a length-matched subset."""
    axis = np.array([pc[n]["axis_score"] for n in subset])
    rec = np.array([pc[n]["recovery_fraction"] for n in subset])
    length = np.array([pc[n]["mean_answer_len"] for n in subset])
    read = spearman_perm_read(axis, rec, n_perm=N_PERM, seed=SEED, label=label)
    return {
        "matching_rule": rule,
        "subset": subset,
        "n": len(subset),
        "excluded": [n for n in all_names if n not in subset],
        "length_min": _r(float(length.min())),
        "length_max": _r(float(length.max())),
        "length_max_over_min": _r(float(length.max() / length.min())),
        "rho": _r(read["rho"]),
        "p_add_one": _r(read["p_add_one"]),
        "n_null_ge": read["n_null_ge"],
        "n_perm": read["n_perm"],
        "seed": read["seed"],
        "within_subset_diagnostics": {
            "rho_length_recovery": _r(_spearman(length, rec)),
            "rho_axis_length": _r(_spearman(axis, length)),
            "note": "descriptive within-subset rank correlations (no permutation band)",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=EVAL / "r4_length_control.json")
    args = ap.parse_args()

    verdict = json.loads((EVAL / "gradient_verdict.json").read_text())
    pc = verdict["per_character"]
    names = sorted(pc)
    axis = np.array([pc[n]["axis_score"] for n in names])
    rec = np.array([pc[n]["recovery_fraction"] for n in names])
    length = np.array([pc[n]["mean_answer_len"] for n in names])

    out: dict = {
        "issue": 2479,
        "characters": names,
        "note": (
            "answer-length-controlled recount of the headline gradient: does "
            "rho(axis, rung-4 recovery) = +0.703 survive controlling for per-character "
            "mean kept-answer length (per_character.mean_answer_len — the committed "
            "answer_length_vs_axis mediation diagnostic's exact values)? All permutation "
            f"reads: n_perm={N_PERM}, seed={SEED}, one-sided add-one p (verdict convention)."
        ),
    }

    # --- (3) zero-order pieces -------------------------------------------------
    head = spearman_perm_read(axis, rec, n_perm=N_PERM, seed=SEED, label="headline replay")
    committed = verdict["headline"]
    assert abs(head["rho"] - committed["rho"]) < 1e-12, (head["rho"], committed["rho"])
    assert abs(head["p_add_one"] - committed["p_add_one"]) < 1e-12

    la = spearman_perm_read(axis, length, n_perm=N_PERM, seed=SEED, label="length-axis replay")
    committed_la = verdict["secondary_reads"]["answer_length_vs_axis"]
    assert abs(la["rho"] - committed_la["rho"]) < 1e-12, (la["rho"], committed_la["rho"])
    assert abs(la["p_add_one"] - committed_la["p_add_one"]) < 1e-12

    lr = spearman_perm_read(
        length,
        rec,
        n_perm=N_PERM,
        seed=SEED,
        label="rho(mean kept-answer length, rung-4 recovery fraction) — computed this round",
    )
    out["zero_order"] = {
        "axis_recovery": {
            "rho": _r(head["rho"]),
            "p_add_one": _r(head["p_add_one"]),
            "n": head["n"],
            "note": "headline replay — asserted equal to the committed verdict",
        },
        "length_axis": {
            "rho": _r(la["rho"]),
            "p_add_one": _r(la["p_add_one"]),
            "n": la["n"],
            "note": "replay — asserted equal to the committed answer_length_vs_axis read",
        },
        "length_recovery": {
            "rho": _r(lr["rho"]),
            "p_add_one": _r(lr["p_add_one"]),
            "n_null_ge": lr["n_null_ge"],
            "n": lr["n"],
            "n_perm": lr["n_perm"],
            "seed": lr["seed"],
            "note": "new read this round; permutation permutes the length labels",
        },
    }

    # --- (1) partial Spearman: axis -> recovery, controlling mean answer length -
    part = partial_spearman_perm_read(
        axis,
        rec,
        length,
        n_perm=N_PERM,
        seed=SEED,
        label="partial rho(axis, rung-4 recovery | mean kept-answer length)",
    )
    part_out = dict(part)
    for k in ("rho_partial", "p_add_one", "null_q95", "null_mean"):
        part_out[k] = _r(part_out[k])
    out["partial_spearman"] = part_out

    # --- (2) length-matched subsample recount -----------------------------------
    lengths_by_name = {n: float(pc[n]["mean_answer_len"]) for n in names}
    subset, _ratio = largest_tight_window(lengths_by_name, CALIPER_RATIO)
    primary = _subset_read(
        subset,
        pc,
        names,
        rule=(
            "greedy caliper: largest contiguous window in sorted mean answer length with "
            f"max/min <= {CALIPER_RATIO} (ties: smaller realized ratio, then lower window "
            "start; characters ordered by (length, name))"
        ),
        label="length-matched subsample recount (greedy caliper window)",
    )

    q1, q3 = (float(q) for q in np.quantile(length, [0.25, 0.75]))
    iqr_subset = [n for n in names if q1 <= lengths_by_name[n] <= q3]
    companion = _subset_read(
        iqr_subset,
        pc,
        names,
        rule=(
            "IQR band: characters with mean answer length inside [Q1, Q3] of the 16-character "
            f"length distribution (numpy linear-interpolation quantiles; Q1={_r(q1)}, "
            f"Q3={_r(q3)})"
        ),
        label="length-matched subsample recount (IQR band companion)",
    )
    out["length_matched_subsample"] = {"primary_caliper": primary, "companion_iqr_band": companion}

    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(
        "partial rho(axis, recovery | length) =",
        out["partial_spearman"]["rho_partial"],
        "| p_add_one =",
        out["partial_spearman"]["p_add_one"],
    )
    print(
        "caliper subset n =",
        primary["n"],
        "| rho =",
        primary["rho"],
        "| p_add_one =",
        primary["p_add_one"],
        "| subset =",
        primary["subset"],
    )
    print(
        "IQR subset n =",
        companion["n"],
        "| rho =",
        companion["rho"],
        "| p_add_one =",
        companion["p_add_one"],
    )
    print(
        "zero-order: axis-recovery rho =",
        out["zero_order"]["axis_recovery"]["rho"],
        "| length-axis rho =",
        out["zero_order"]["length_axis"]["rho"],
        "| length-recovery rho =",
        out["zero_order"]["length_recovery"]["rho"],
        "( p =",
        out["zero_order"]["length_recovery"]["p_add_one"],
        ")",
    )


if __name__ == "__main__":
    main()
