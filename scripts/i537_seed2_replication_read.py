"""Issue #537 same-issue follow-up `seed2-marker-fact-replication`: registered reads.

Compares the marker + fact rows of the G grid across two training seeds
(42 = parent, 1042 = replication; LoRA init + data-order shuffle only — frozen
data, frozen eval protocol, base model unchanged). CPU-only, zero GPU.

Registered criteria (epm:followup-scope v1):
  R1. Per-row diagonal-normalized breadth rank-correlates >= 0.7 across seeds
      over surviving rows (marker row AND fact row; falsify below ~0.5).
  R2. sp_ph1 stays below the broad-family breadth minimum on both rows.
  R3. Worked-example / format-instruction rows stay contained; the inoculation
      sign-flip (binst off-diag mean < default off-diag mean) reproduces.
  Plus the honest seed-variance read: per-cell seed42-vs-seed1042 scatter.

Conventions mirror scripts/i537_assemble_tensor.py + i537_registered_reads.py:
  - marker cell G = mean per-question delta_logp (g_mean_delta_logp);
  - fact cell G = TAUGHT rate (parse_verdict_fact) minus the frozen base rate
    for the eval context (p0/headroom_rates/fact.json; base model unchanged);
  - implant_failed: diagonal G < 4.0 nat (marker) / < 0.05 (fact) -> whole row
    excluded ("surviving rows"), applied identically per seed;
  - saturated (marker): >90% of trained slot logps > -0.1 AND emission >= 0.92.

Output: eval_results/issue_537/analysis/seed2_replication.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

EVAL = Path("eval_results/issue_537")
OUT = EVAL / "analysis"
OUT.mkdir(exist_ok=True)

SEEDS = (42, 1042)
IMPLANT_THRESHOLDS = {"marker": 4.0, "fact": 0.05}
BROAD_FAMILY = [
    "sp_swe",
    "sp_doctor",
    "sp_ph2",  # sp_ph1 deliberately excluded: it is the row under test (R2)
    "wc_short_code",
    "wc_short_advice",
    "wc_long_write",
    "reph_imp",
    "reph_polite",
    "reph_casual",
    "default",
]
CONTAINED_FAMILY = ["icl_k2", "icl_k8", "fmt_json", "fmt_code"]

from explore_persona_space.experiments.i537_contexts import (  # noqa: E402
    eval_cids_for,
    train_cids_for,
)
from explore_persona_space.experiments.i537_judging import parse_verdict_fact  # noqa: E402


def marker_grid(seed: int) -> tuple[dict[tuple[str, str], dict], list[str], list[str]]:
    """Load all marker G cells for one seed -> {(train,eval): summary}."""
    tcids = train_cids_for("marker")
    ecids = eval_cids_for("marker")
    cells: dict[tuple[str, str], dict] = {}
    for t in tcids:
        for e in ecids:
            p = EVAL / f"G_cells/marker/{t}__{e}__seed{seed}.json"
            cell = json.loads(p.read_text())
            trained_logp = np.array([r["trained"]["logp"] for r in cell["per_question"]])
            assert len(trained_logp) == cell["n_questions"], (t, e, seed)
            cells[(t, e)] = {
                "g": float(cell["g_mean_delta_logp"]),
                "saturated": bool(
                    (trained_logp > -0.1).mean() > 0.9 and cell["emission_rate_trained"] >= 0.92
                ),
                "emission_rate_trained": float(cell["emission_rate_trained"]),
            }
    return cells, tcids, ecids


def fact_grid(seed: int) -> tuple[dict[tuple[str, str], dict], list[str], list[str]]:
    """Load all fact judge cells for one seed -> {(train,eval): summary}."""
    tcids = train_cids_for("fact")
    ecids = eval_cids_for("fact")
    base_rates = json.loads((EVAL / "p0/headroom_rates/fact.json").read_text())["rates"]
    cells: dict[tuple[str, str], dict] = {}
    for t in tcids:
        for e in ecids:
            p = EVAL / f"judgments/fact/{t}_seed{seed}/{e}.json"
            payload = json.loads(p.read_text())
            vals = []
            for r in payload["verdicts"]:
                cat = parse_verdict_fact(r["raw_verdict"])
                if cat is None:
                    continue
                vals.append(1.0 if cat == "TAUGHT" else 0.0)
            base = base_rates[e]
            assert len(vals) >= 4, (t, e, seed, len(vals))
            rate = float(np.mean(vals))
            cells[(t, e)] = {"g": rate - base, "rate_trained": rate, "n_verdicts": len(vals)}
    return cells, tcids, ecids


def row_stats(
    cells: dict[tuple[str, str], dict], tcids: list[str], ecids: list[str], thresh: float
) -> dict[str, dict]:
    """Per-train-row diag, off-diag mean, diag-normalized breadth, implant flag."""
    rows = {}
    for t in tcids:
        diag = cells[(t, t)]["g"]
        off = [cells[(t, e)]["g"] for e in ecids if e != t]
        rows[t] = {
            "diag": diag,
            "offdiag_mean": float(np.mean(off)),
            "breadth_diagnorm": float(np.mean(off) / diag) if diag != 0 else None,
            "implant_failed": bool(diag < thresh),
        }
    return rows


def replication_reads(behavior: str, rows_by_seed: dict[int, dict[str, dict]]) -> dict:
    """R1 breadth rank correlation + R2 sp_ph1 + R3 containment/inoculation, per behavior."""
    r42, r1042 = rows_by_seed[42], rows_by_seed[1042]
    tcids = list(r42)
    surviving = [
        t for t in tcids if not r42[t]["implant_failed"] and not r1042[t]["implant_failed"]
    ]
    failed = {s: [t for t in tcids if rows_by_seed[s][t]["implant_failed"]] for s in SEEDS}
    b42 = [r42[t]["breadth_diagnorm"] for t in surviving]
    b1042 = [r1042[t]["breadth_diagnorm"] for t in surviving]
    rho, p = spearmanr(b42, b1042)
    binst = next(t for t in tcids if t.startswith("binst"))
    surv_no_binst = [t for t in surviving if t != binst]
    rho_nb, p_nb = spearmanr(
        [r42[t]["breadth_diagnorm"] for t in surv_no_binst],
        [r1042[t]["breadth_diagnorm"] for t in surv_no_binst],
    )

    def r2_sp_ph1(rows: dict[str, dict]) -> dict:
        broad = {
            t: rows[t]["breadth_diagnorm"] for t in BROAD_FAMILY if not rows[t]["implant_failed"]
        }
        v = rows["sp_ph1"]["breadth_diagnorm"]
        return {
            "sp_ph1_breadth": v,
            "broad_family_min": min(broad.values()),
            "broad_family_min_cid": min(broad, key=broad.get),
            "pass": bool(v < min(broad.values())),
        }

    def r3_containment(rows: dict[str, dict]) -> dict:
        broad_min = min(
            rows[t]["breadth_diagnorm"] for t in BROAD_FAMILY if not rows[t]["implant_failed"]
        )
        contained = {
            t: {
                "breadth_diagnorm": rows[t]["breadth_diagnorm"],
                "below_broad_min": bool(rows[t]["breadth_diagnorm"] < broad_min),
            }
            for t in CONTAINED_FAMILY
            if not rows[t]["implant_failed"]
        }
        return {"broad_family_min": broad_min, "contained_rows": contained}

    def r3_inoculation(rows: dict[str, dict]) -> dict:
        return {
            "binst_offdiag_mean": rows[binst]["offdiag_mean"],
            "default_offdiag_mean": rows["default"]["offdiag_mean"],
            "sign_flip_pass": bool(rows[binst]["offdiag_mean"] < rows["default"]["offdiag_mean"]),
        }

    return {
        "surviving_rows": surviving,
        "implant_failed_rows": failed,
        "r1_breadth_rank_corr": {
            "spearman_rho": float(rho),
            "p": float(p),
            "n_rows": len(surviving),
            "pass_ge_0p7": bool(rho >= 0.7),
            "falsify_lt_0p5": bool(rho < 0.5),
            "excl_binst_row": {
                "spearman_rho": float(rho_nb),
                "p": float(p_nb),
                "n": len(surv_no_binst),
            },
        },
        "r2_sp_ph1_containment": {s: r2_sp_ph1(rows_by_seed[s]) for s in SEEDS},
        "r3_contained_family": {s: r3_containment(rows_by_seed[s]) for s in SEEDS},
        "r3_inoculation": {s: r3_inoculation(rows_by_seed[s]) for s in SEEDS},
        "per_row_breadth": {s: rows_by_seed[s] for s in SEEDS},
    }


def cell_scatter(
    cells_by_seed: dict[int, dict[tuple[str, str], dict]], tcids: list[str], ecids: list[str]
) -> dict:
    """Per-cell seed42-vs-seed1042 agreement over ALL 480 cells (no masks): the
    honest seed-noise band. Also split diag / off-diag."""
    keys = [(t, e) for t in tcids for e in ecids]
    g42 = np.array([cells_by_seed[42][k]["g"] for k in keys])
    g1042 = np.array([cells_by_seed[1042][k]["g"] for k in keys])
    off = np.array([k[0] != k[1] for k in keys])
    d = g1042 - g42
    pear_r, pear_p = pearsonr(g42, g1042)
    sp_rho, sp_p = spearmanr(g42, g1042)
    return {
        "n_cells": len(keys),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho": float(sp_rho),
        "spearman_p": float(sp_p),
        "rms_diff_all": float(np.sqrt(np.mean(d**2))),
        "rms_diff_offdiag": float(np.sqrt(np.mean(d[off] ** 2))),
        "rms_diff_diag": float(np.sqrt(np.mean(d[~off] ** 2))),
        "mean_diff_offdiag": float(np.mean(d[off])),
        "sd_g42_offdiag": float(np.std(g42[off], ddof=1)),
    }


def main() -> None:
    results: dict = {
        "seeds": list(SEEDS),
        "single_seed_inputs_note": (
            "seed = training stochasticity only (LoRA init + data order); training data, "
            "eval protocol, base caches, and judge prompts are frozen and shared across seeds"
        ),
    }

    # ----- marker row -----
    mcells, mt, me = {}, None, None
    for s in SEEDS:
        mcells[s], mt, me = marker_grid(s)
    mrows = {s: row_stats(mcells[s], mt, me, IMPLANT_THRESHOLDS["marker"]) for s in SEEDS}
    results["marker"] = replication_reads("marker", mrows)
    results["marker"]["per_cell_scatter"] = cell_scatter(mcells, mt, me)
    results["marker"]["saturated_cells"] = {
        s: [f"{t}__{e}" for (t, e), c in mcells[s].items() if c["saturated"]] for s in SEEDS
    }
    # Asymmetry noise bound (finding 6): can training noise produce the
    # antisymmetric variance? Var(0.5*(n_ij - n_ji)) = var_cell/2 where
    # var_cell = Var(G_s1 - G_s2)/2 over the 16x16 off-diagonal block.
    pairs = [(a, b) for a in mt for b in mt if a != b]
    d_block = np.array([mcells[42][k]["g"] - mcells[1042][k]["g"] for k in pairs])
    g_block = np.array([mcells[42][k]["g"] for k in pairs])
    var_cell = float(np.mean(d_block**2) / 2)
    results["marker"]["asymmetry_noise_bound"] = {
        "var_offdiag_16x16_block_seed42": float(np.var(g_block)),
        "training_noise_var_per_cell": var_cell,
        "implied_antisym_var_from_training_noise": var_cell / 2,
        "parent_antisym_fraction": 0.283,
        "frac_of_measured_antisym_var_explained": (var_cell / 2) / (0.283 * float(np.var(g_block))),
    }
    results["marker"]["stop_steps"] = {}
    for s in SEEDS:
        sd = EVAL / ("p1/stop_steps" if s == 42 else f"p1/stop_steps_seed{s}")
        results["marker"]["stop_steps"][s] = {
            p.stem: json.loads(p.read_text())["stop_step"] for p in sorted(sd.glob("*.json"))
        }

    # ----- fact row -----
    fcells, ft, fe = {}, None, None
    for s in SEEDS:
        fcells[s], ft, fe = fact_grid(s)
    frows = {s: row_stats(fcells[s], ft, fe, IMPLANT_THRESHOLDS["fact"]) for s in SEEDS}
    results["fact"] = replication_reads("fact", frows)
    results["fact"]["per_cell_scatter"] = cell_scatter(fcells, ft, fe)

    # cache full grids for the figure script
    results["_grids"] = {
        "marker": {s: {f"{t}__{e}": mcells[s][(t, e)]["g"] for t in mt for e in me} for s in SEEDS},
        "fact": {s: {f"{t}__{e}": fcells[s][(t, e)]["g"] for t in ft for e in fe} for s in SEEDS},
    }

    out = OUT / "seed2_replication.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "_grids"}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
