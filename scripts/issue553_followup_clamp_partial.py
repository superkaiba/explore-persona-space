# ruff: noqa: RUF002, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — Step 9a-ter free-analysis follow-up: clamp-routing beyond-shrinkage partial.

The surviving title clause reads "only the clamp's base-state association
transfers". Its evidence is the pair of raw context-level reads
FE(Δz(EOS)) vs context-mean base margin: +0.65 on the #478 persona panel
(``transfer_478.json`` → ``clamp_routing.vs_persona_mean_margin_base``) and
+0.70 on the #532 bystander panel (``transfer_478.json`` →
``i532_side_points.clamp_bystander_fe_vs_prior_margin_rho``). But trained-side
z(EOS) FE correlates 0.91–0.97 with the base z(EOS) LEVEL on both panels, so
the margin association could be pure arithmetic persistence of the base EOS
level rather than a routing signal carried by the margin. The diagnostic here:

(a) reproduce the raw +0.65/+0.70 reads (1e-6 gates against the committed
    parent JSONs — fail loud, no new number on mismatch);
(b) PARTIAL rank Spearman of the FE vs context-mean base margin CONTROLLING
    for context-mean base z(EOS) level (the ``issue531_base_prior_reanalysis
    .partial_spearman`` convention: rank-residualize both sides on the rank
    control + intercept, Spearman of the residuals);
(c) the complementary partial (vs base z(EOS) level controlling base margin).

Inference mirrors the parent reads: unit-axis (persona n=35 / bystander n=16)
percentile bootstrap with 2,000 reps seed 42 (rank-residualization re-fit
inside every resample; degenerate resamples dropped + counted) and an MC
permutation p (10,000 reps; for partials the control-residualized ranks are
permuted — Freedman–Lane-style residual permutation, documented in-method).

#532 regime note: the registered +0.70 read's X is the bystander's OWN-response
base margin (``base_prior_logp.json``), so the PRIMARY #532 triple controls the
OWN-response base z(EOS) mean (same regime as X); the matched-slot triple
(X = bystander-mean ``margin_base_matched``, control = bystander-mean matched-
slot base z(EOS)) is reported as a sensitivity — its complementary raw read is
gated against the committed ``channel_anatomy.json`` −0.86. Regimes are never
mixed within one triple. The #478 panel has only the matched-slot base regime
(no own-response base data exists there — named limitation in the parent).

Outputs ``eval_results/issue_553/followup_clamp_partial.json`` (with an
explicit ``interpretation_guard``) + one raw-vs-partial scatter figure.
Smoke = this exact script with reduced ``--n-marginal-boot/--n-perm``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue539_corrected_reads_inference as i539inf
import issue539_residual_per_cohort as i539
import issue553_panel as p553
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

# ── Partial-Spearman machinery (issue531 convention, array form) ─────────────


def _rank_residual_pair(
    x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-residualize x and y on [intercept | rank controls] (531 convention)."""
    xr = rankdata(x, method="average")
    yr = rankdata(y, method="average")
    design = np.column_stack([np.ones(len(x))] + [rankdata(c, method="average") for c in controls])
    cx, *_ = np.linalg.lstsq(design, xr, rcond=None)
    cy, *_ = np.linalg.lstsq(design, yr, rcond=None)
    return xr - design @ cx, yr - design @ cy


def partial_point(
    x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Partial Spearman rho + the residual pair (Spearman OF the residuals)."""
    rx, ry = _rank_residual_pair(x, y, controls)
    r, _ = spearmanr(rx, ry)
    return float(r), rx, ry


def _selfcheck_partial_vs_531(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> None:
    """Drift assert: the array-form partial equals the panel module's
    ``_partial_spearman_531`` (the issue531 DataFrame mirror) on observed data."""
    df = pd.DataFrame({"y": y, "x": x, "c": control})
    ref = p553._partial_spearman_531(df, "y", "x", ["c"])
    got, _, _ = partial_point(x, y, [control])
    assert abs(got - ref) < 1e-9, f"partial-Spearman drift vs issue531 convention: {got} vs {ref}"


def partial_unit_bootstrap(
    x: np.ndarray, y: np.ndarray, controls: list[np.ndarray], n_boot: int, seed: int
) -> dict:
    """Unit-axis percentile bootstrap of the partial rho.

    Resamples the (x, y, controls) unit rows with replacement (the unit IS the
    cluster at this aggregation level — persona / bystander axis, matching the
    parent reads' ``ci95_boot_personas`` / ``ci95_boot_bystanders``), re-ranks
    and re-fits the rank-residualization inside every resample. Degenerate
    resamples (constant x, y, or residuals) are dropped AND counted.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if i539._is_degenerate(xb) or i539._is_degenerate(yb):
            n_deg += 1
            continue
        rxb, ryb = _rank_residual_pair(xb, yb, [c[idx] for c in controls])
        if i539._is_degenerate(rxb) or i539._is_degenerate(ryb):
            n_deg += 1
            continue
        rhos.append(i539._fast_spearman(rankdata(rxb), rankdata(ryb)))
    return i539inf._percentile_summary(rhos, n_boot, n_deg)


def raw_read(x: np.ndarray, y: np.ndarray, args, axis_label: str) -> dict:
    """Raw Spearman read with the parent's unit-axis bootstrap CI + perm p."""
    return {
        "rho": i539._spearman_rho(x, y),
        "n_units": len(x),
        "ci95_boot_units": i539._bootstrap_spearman_ci(x, y, args.n_marginal_boot, args.seed),
        "p_perm": {
            **i539._permutation_p(x, y, args.n_perm, args.seed),
            "method": f"MC permutation of the {len(x)} {axis_label} labels",
        },
    }


def partial_read(
    x: np.ndarray,
    y: np.ndarray,
    controls: dict[str, np.ndarray],
    args,
    axis_label: str,
) -> dict:
    """Partial Spearman read: point + re-fit unit bootstrap + residual perm p."""
    ctl = list(controls.values())
    rho_p, rx, ry = partial_point(x, y, ctl)
    return {
        "rho_partial": rho_p,
        "n_units": len(x),
        "controls": list(controls),
        "ci95_boot_units": partial_unit_bootstrap(x, y, ctl, args.n_marginal_boot, args.seed),
        "p_perm_residuals": {
            **i539._permutation_p(rx, ry, args.n_perm, args.seed),
            "method": f"Freedman-Lane-style residual permutation: the {len(x)} "
            f"control-residualized rank residuals of the FE ({axis_label} axis) "
            "permuted against the fixed control-residualized X residuals",
        },
    }


_VERDICT_RULE = (
    "'survives' when the partial rho keeps the raw read's sign AND its 95% unit-axis "
    "bootstrap CI excludes 0; 'sign-flipped-significant' when the CI excludes 0 with the "
    "opposite sign; otherwise 'does-not-survive' (CI spans 0 — at n=16/35 this is 'bounded "
    "below the CI half-width', NOT proof of absence; power note carried from the parent reads)"
)


def _verdict(raw_rho: float, partial_block: dict) -> str:
    """Pre-stated verdict rule (see _VERDICT_RULE)."""
    lo = partial_block["ci95_boot_units"]["low"]
    hi = partial_block["ci95_boot_units"]["high"]
    rho_p = partial_block["rho_partial"]
    if np.isnan(lo) or np.isnan(hi) or (lo <= 0.0 <= hi):
        return "does-not-survive"
    return "survives" if np.sign(rho_p) == np.sign(raw_rho) else "sign-flipped-significant"


def triple_block(
    fe: np.ndarray,
    x_margin: np.ndarray,
    x_zeos: np.ndarray,
    args,
    axis_label: str,
    margin_name: str,
    zeos_name: str,
) -> dict:
    """The full diagnostic triple for one (panel, base-state regime)."""
    raw_margin = raw_read(x_margin, fe, args, axis_label)
    raw_zeos = raw_read(x_zeos, fe, args, axis_label)
    part_margin = partial_read(x_margin, fe, {zeos_name: x_zeos}, args, axis_label)
    part_zeos = partial_read(x_zeos, fe, {margin_name: x_margin}, args, axis_label)
    return {
        f"raw_fe_vs_{margin_name}": raw_margin,
        f"raw_fe_vs_{zeos_name}": raw_zeos,
        "x_vs_control_rho": i539._spearman_rho(x_margin, x_zeos),
        f"partial_fe_vs_{margin_name}_given_{zeos_name}": part_margin,
        f"partial_fe_vs_{zeos_name}_given_{margin_name}": part_zeos,
        "margin_partial_verdict": _verdict(raw_margin["rho"], part_margin),
    }


# ── Panel builders ───────────────────────────────────────────────────────────


def build_i478_inputs(args) -> dict:
    """#478 persona-level inputs: FE(dz_eos) + persona-mean base-state columns."""
    df = p553.load_i478_panel(args.i478_parquet)
    step0 = p553.step0_i478(df, args.i478_parquet.parent / "summary_logit.json")
    agg = p553.aggregate_run_persona(df)
    y = agg["dz_eos"].to_numpy(dtype=np.float64)
    per_l = agg["held_out_persona"].to_numpy()
    run_l = agg["run_id"].to_numpy()
    per_u, pc = np.unique(per_l, return_inverse=True)
    _, rc = np.unique(run_l, return_inverse=True)
    fe = p553.fe_vector(y, pc, rc, len(per_u), int(rc.max()) + 1)
    pm = agg.groupby("held_out_persona").agg(
        margin_base=("margin_base", "mean"), z_eos_base=("z_eos_base", "mean")
    )
    return {
        "step0": step0,
        "units": [str(u) for u in per_u],
        "fe": fe,
        "x_margin": pm.loc[per_u, "margin_base"].to_numpy(dtype=np.float64),
        "x_zeos": pm.loc[per_u, "z_eos_base"].to_numpy(dtype=np.float64),
    }


def build_i532_inputs(args) -> dict:
    """#532 bystander-level inputs (ordinary_cross): FE(dz_eos) + both base regimes."""
    panel = p553.build_margin_panel(args.i532_dir)
    step0 = p553.step0_i532(panel, args.i532_dir)
    masks = p553.cohort_masks_553(panel)
    m = masks["ordinary_cross"]
    src, byst = panel["source_cid"][m], panel["bystander_label"][m]
    _, sc = np.unique(src, return_inverse=True)
    byst_u, bc = np.unique(byst, return_inverse=True)
    fe = p553.fe_vector(panel["dz_eos"][m], bc, sc, len(byst_u), int(sc.max()) + 1)
    # OWN-response base state (the registered +0.70 read's regime): margin from
    # the loader; the z(EOS) level re-derived from base_prior_logp.json per_q
    # with the margin identity asserted per bystander.
    a1 = json.loads((args.i532_dir / "logp_slot_followup" / "base_prior_logp.json").read_text())
    assert a1["schema_version"] == p553.FOLLOWUP_SCHEMA, a1["schema_version"]
    assert a1["phase"] == "A1_base_prior_slots", a1["phase"]
    prior_zeos_own: dict[str, float] = {}
    for b, blk in a1["per_bystander"].items():
        qs = blk["per_q"]
        assert len(qs) == 50, (b, len(qs))
        zm = float(np.mean([q["z_marker"] for q in qs]))
        ze = float(np.mean([q["z_eos"] for q in qs]))
        assert abs((zm - ze) - panel["_prior_margin_own_by_bystander"][b]) <= p553.IDENTITY_TOL, b
        prior_zeos_own[b] = ze
    # Matched-slot base state (sensitivity regime).
    ze_b_cell = panel["q_ze_b"].mean(axis=1)
    mb_cell = panel["margin_base_matched"]
    byst_m = byst  # alias for readability in the comprehensions below
    return {
        "step0": step0,
        "units": [str(u) for u in byst_u],
        "fe": fe,
        "x_margin_own": np.array(
            [panel["_prior_margin_own_by_bystander"][b] for b in byst_u], dtype=np.float64
        ),
        "x_zeos_own": np.array([prior_zeos_own[b] for b in byst_u], dtype=np.float64),
        "x_margin_matched": np.array(
            [float(mb_cell[m][byst_m == b].mean()) for b in byst_u], dtype=np.float64
        ),
        "x_zeos_matched": np.array(
            [float(ze_b_cell[m][byst_m == b].mean()) for b in byst_u], dtype=np.float64
        ),
    }


# ── Reproduction gates (fail loud BEFORE any new number ships) ───────────────


def reproduction_gates(i478: dict, i532: dict, args) -> list[dict]:
    """1e-6 gates against the committed parent JSONs (#553 step-0 pattern)."""
    transfer = json.loads((args.parent_553_dir / "transfer_478.json").read_text())
    anatomy = json.loads((args.parent_553_dir / "channel_anatomy.json").read_text())
    gates = [
        {
            "name": "i478 raw FE(dz_eos) vs persona-mean margin_base (the +0.65 read)",
            "got": i539._spearman_rho(i478["x_margin"], i478["fe"]),
            "want": float(transfer["clamp_routing"]["vs_persona_mean_margin_base"]["rho"]),
        },
        {
            "name": "i532 bystander-FE(dz_eos) vs own-response prior margin (the +0.70 read)",
            "got": i539._spearman_rho(i532["x_margin_own"], i532["fe"]),
            "want": float(transfer["i532_side_points"]["clamp_bystander_fe_vs_prior_margin_rho"]),
        },
        {
            "name": "i532 bystander-FE(dz_eos) vs matched-slot base z(EOS) (the -0.86 read)",
            "got": i539._spearman_rho(i532["x_zeos_matched"], i532["fe"]),
            "want": float(
                anatomy["absolute_z_eos_anatomy"]["ordinary_cross"][
                    "rho_bystFE_dz_eos_vs_bystmean_z_eos_base"
                ]
            ),
        },
    ]
    for g in gates:
        g["pass"] = bool(abs(g["got"] - g["want"]) <= p553.GATE_TOL)
    failed = [g for g in gates if not g["pass"]]
    if failed:
        print(
            "REPRODUCTION GATE FAILED — rebuilt reads diverge from the committed", file=sys.stderr
        )
        print("parent JSONs. NOT computing any new number.", file=sys.stderr)
        for g in failed:
            print(f"  FAIL {g['name']}: got {g['got']!r}, want {g['want']!r}", file=sys.stderr)
        sys.exit(1)
    print(
        f"[gates] reproduction PASS ({len(gates)} committed reads reproduced to {p553.GATE_TOL:g})"
    )
    return gates


# ── Figure ───────────────────────────────────────────────────────────────────


def make_figure(i478: dict, i532: dict, blk478: dict, blk532: dict, fig_dir: Path) -> None:
    """Raw-vs-partial scatter pair per panel (raw alongside processed)."""
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
    rows = [
        (
            "#478 persona panel (n=35)",
            i478["x_margin"],
            i478["fe"],
            blk478["raw_fe_vs_persona_mean_margin_base"]["rho"],
            blk478["partial_fe_vs_persona_mean_margin_base_given_persona_mean_z_eos_base"],
            [i478["x_zeos"]],
            "persona-mean base margin (matched-slot)",
        ),
        (
            "#532 bystander panel, ordinary cross (n=16)",
            i532["x_margin_own"],
            i532["fe"],
            blk532["raw_fe_vs_prior_margin_own"]["rho"],
            blk532["partial_fe_vs_prior_margin_own_given_prior_z_eos_own"],
            [i532["x_zeos_own"]],
            "bystander own-response base margin",
        ),
    ]
    for r, (panel_name, x, fe, raw_rho, part_blk, controls, x_label) in enumerate(rows):
        ax_raw, ax_part = axes[r, 0], axes[r, 1]
        ax_raw.plot(x, fe, "o", ms=4, alpha=0.7, color=colors[0])
        ax_raw.set_title(f"{panel_name}\nraw rho={raw_rho:+.2f}", fontsize=8)
        ax_raw.set_xlabel(x_label, fontsize=7)
        ax_raw.set_ylabel("FE of EOS-side change Δz(EOS)", fontsize=7)
        rx, ry = _rank_residual_pair(x, fe, controls)
        ax_part.plot(rx, ry, "o", ms=4, alpha=0.7, color=colors[1])
        ax_part.set_title(
            f"partial | base z(EOS) level: rho={part_blk['rho_partial']:+.2f}\n"
            f"95% CI [{part_blk['ci95_boot_units']['low']:+.2f}, "
            f"{part_blk['ci95_boot_units']['high']:+.2f}]",
            fontsize=8,
        )
        ax_part.set_xlabel("base-margin rank residual | base z(EOS)", fontsize=7)
        ax_part.set_ylabel("FE rank residual | base z(EOS)", fontsize=7)
    fig.suptitle(
        "Clamp routing beyond shrinkage: FE(Δz(EOS)) vs base margin, raw vs partial", fontsize=9
    )
    fig.tight_layout()
    savefig_paper(fig, "followup_clamp_partial_raw_vs_partial", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote followup_clamp_partial_raw_vs_partial to {fig_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = p553.common_parser(
        "Task #553 follow-up: partial check that the clamp's base-margin association "
        "survives controlling the base z(EOS) level (both panels; zero new data)."
    )
    parser.add_argument(
        "--parent-553-dir",
        type=Path,
        default=Path("eval_results/issue_553"),
        dest="parent_553_dir",
        help="directory holding the committed transfer_478.json / channel_anatomy.json "
        "(separate from --out-dir so smoke runs can write elsewhere)",
    )
    args = parser.parse_args()

    print("[load] #478 panel ...")
    i478 = build_i478_inputs(args)
    print("[load] #532 panel ...")
    i532 = build_i532_inputs(args)
    gates = reproduction_gates(i478, i532, args)
    _selfcheck_partial_vs_531(i478["x_margin"], i478["fe"], i478["x_zeos"])

    print("[reads] #478 triple ...")
    blk478 = triple_block(
        i478["fe"],
        i478["x_margin"],
        i478["x_zeos"],
        args,
        "persona",
        "persona_mean_margin_base",
        "persona_mean_z_eos_base",
    )
    print("[reads] #532 primary (own-response regime) ...")
    blk532 = triple_block(
        i532["fe"],
        i532["x_margin_own"],
        i532["x_zeos_own"],
        args,
        "bystander",
        "prior_margin_own",
        "prior_z_eos_own",
    )
    print("[reads] #532 sensitivity (matched-slot regime) ...")
    blk532_matched = triple_block(
        i532["fe"],
        i532["x_margin_matched"],
        i532["x_zeos_matched"],
        args,
        "bystander",
        "matched_slot_margin_base",
        "matched_slot_z_eos_base",
    )

    make_figure(i478, i532, blk478, blk532, args.fig_dir)

    v478 = blk478["margin_partial_verdict"]
    v532 = blk532["margin_partial_verdict"]
    v532m = blk532_matched["margin_partial_verdict"]
    out = {
        "metadata": p553.result_metadata(args, "issue553_followup_clamp_partial.py"),
        "question": (
            "does context-mean base MARGIN carry clamp-routing signal beyond the arithmetic "
            "persistence of the base z(EOS) LEVEL (trained-side z(EOS) FE correlates 0.91-0.97 "
            "with base z(EOS) on these panels)?"
        ),
        "step0_i478": i478["step0"],
        "step0_i532": i532["step0"],
        "reproduction_gates": gates,
        "i478_panel": {
            "unit": "persona (n=35); FE of dz_eos from the two-way (run + persona) fit on the "
            "2,800 run x persona aggregates; base columns are persona means (matched-slot — "
            "the only base regime existing on this panel)",
            "units": i478["units"],
            **blk478,
        },
        "i532_panel": {
            "unit": "bystander (n=16, ordinary cross-context cohort); FE of dz_eos from the "
            "two-way (source + bystander) fit on the 240 ordinary cross cells",
            "units": i532["units"],
            "primary_own_response_regime": {
                "note": "X mirrors the registered +0.70 read (own-response base margin from "
                "base_prior_logp.json); control is the own-response base z(EOS) mean — same "
                "regime as X, never mixed",
                **blk532,
            },
            "sensitivity_matched_slot_regime": {
                "note": "X = bystander-mean margin_base_matched, control = bystander-mean "
                "matched-slot base z(EOS) (the regime the 0.91/-0.86 persistence reads use)",
                **blk532_matched,
            },
        },
        "interpretation_guard": {
            "verdict_rule": _VERDICT_RULE,
            "if_partial_survives": (
                "base margin carries routing signal beyond the base EOS level's arithmetic "
                "persistence — the title clause 'only the clamp's base-state association "
                "transfers' stands as scoped"
            ),
            "if_partial_does_not_survive": (
                "the margin association is not separable from the base z(EOS) level on that "
                "panel — the title clause overstates and must be weakened to arithmetic "
                "persistence of the base EOS level (no channel claim transfers beyond it)"
            ),
            "verdict_i478": v478,
            "verdict_i532_primary_own_response": v532,
            "verdict_i532_sensitivity_matched_slot": v532m,
        },
    }
    p553.write_json(args.out_dir / "followup_clamp_partial.json", out)

    pm478 = blk478["partial_fe_vs_persona_mean_margin_base_given_persona_mean_z_eos_base"]
    pm532 = blk532["partial_fe_vs_prior_margin_own_given_prior_z_eos_own"]
    print(
        f"[headline] #478: raw rho={blk478['raw_fe_vs_persona_mean_margin_base']['rho']:+.3f} → "
        f"partial rho={pm478['rho_partial']:+.3f} "
        f"CI [{pm478['ci95_boot_units']['low']:+.3f}, {pm478['ci95_boot_units']['high']:+.3f}] "
        f"→ {v478}"
    )
    print(
        f"[headline] #532: raw rho={blk532['raw_fe_vs_prior_margin_own']['rho']:+.3f} → "
        f"partial rho={pm532['rho_partial']:+.3f} "
        f"CI [{pm532['ci95_boot_units']['low']:+.3f}, {pm532['ci95_boot_units']['high']:+.3f}] "
        f"→ {v532} (matched-slot sensitivity: {v532m})"
    )


if __name__ == "__main__":
    main()
