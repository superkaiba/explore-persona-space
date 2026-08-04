#!/usr/bin/env python3
"""Task #1491 Phase-4: paired-bootstrap contrasts + hero figure.

Consumes the per-scale fits + per-context test preds/targets Unit 3
produces:
  eval_results/issue_1491/scale_ladder/fits_<slug>.json
  data/issue_1491/preds/<slug>_test_preds_{ridge,registered_nonlinear}.npz

Computes the plan §3 registered contrasts:

- **PRIMARY:** Δ = ridge R²(32B) − ridge R²(0.5B), context arm, primary
  layer, matched-n 25k, with a paired bootstrap (1,000 draws) over the
  1,000 shared pinned test contexts.
- **SECONDARY (H3):** ΔΓ = Γ(32B) − Γ(0.5B), where Γ(s) = [registered
  nonlinear test R² (pre-registered MLP-w8192)] − [ridge
  test R²] at the same layer/n; same paired bootstrap.
- **Descriptive:** Spearman monotonicity across all 6 scales (paired-
  bootstrap CI) — not a registered verdict, per plan §3.
- **Ceiling-normalized:** R²(s) / reliability_ceiling(s) per scale — a
  companion read the plan §4 H4 screen consumes.

DISJOINT + exhaustive verdict labels (per plan §3):

- **Predictability-increases** ⇔ Δ > 0 AND Δ's 95% CI excludes 0 on the
  positive side.
- **Predictability-decreases** ⇔ Δ's 95% CI is wholly below 0.
- **Scale-inconclusive** ⇔ otherwise.

Same three-way DISJOINT + exhaustive labels for ΔΓ (Gap-shrinks /
Gap-grows / Gap-inconclusive), per plan §3.

Writes:
  eval_results/issue_1491/scale_ladder/ladder_contrasts.json
  figures/issue_1491/hero_r2_vs_scale.png (+ meta.json)

Uses the /paper-plots conventions (paper_plots.setup_paper_style) for
publication-quality output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# Heavy import (numpy) MUST come AFTER load_dotenv() so the shared-VM thread
# caps (#847) bind in-process. Pinned by tests/test_shared_vm_thread_caps.py
# (test_no_new_torch_before_dotenv_vm_entrypoints).
import numpy as np  # noqa: E402

logger = logging.getLogger("issue1491_ladder_contrasts")

# ---------------------------------------------------------------------------
# Constants — the 6-rung scale ladder (matches issue1491_ladder_fits.LADDER_SCALES)
# ---------------------------------------------------------------------------

SCALE_ORDER = ["scale05", "scale15", "scale3", "scale7_refit", "scale14", "scale32"]

# For the ladder plot: params in billions (for the x-axis).
SCALE_PARAMS_B = {
    "scale05": 0.5,
    "scale15": 1.5,
    "scale3": 3.0,
    "scale7_refit": 7.0,
    "scale14": 14.0,
    "scale32": 32.0,
}

# Display names for tables/figures.
SCALE_DISPLAY = {
    "scale05": "Qwen 0.5B",
    "scale15": "Qwen 1.5B",
    "scale3": "Qwen 3B",
    "scale7_refit": "Qwen 7B (refit)",
    "scale14": "Qwen 14B",
    "scale32": "Qwen 32B",
}

# Paired-bootstrap sample count (plan §3).
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 1491

# CI level (plan §3 registered 95%).
CI_LOW, CI_HIGH = 2.5, 97.5


# ---------------------------------------------------------------------------
# Load fits + per-context preds
# ---------------------------------------------------------------------------


def _load_fits_json(fits_dir: Path, slug: str) -> dict | None:
    """Load one scale's fits JSON if present."""
    path = fits_dir / f"fits_{slug}.json"
    if not path.exists():
        logger.warning("[contrasts] missing fits JSON: %s", path)
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_preds_npz(preds_dir: Path, slug: str, kind: str) -> dict | None:
    """Load one scale's per-context test preds + targets .npz.

    kind = "ridge" or "registered_nonlinear".
    Returns {"ci": (n,), "target": (n, H), "pred": (n, H)} or None."""
    path = preds_dir / f"{slug}_test_preds_{kind}.npz"
    if not path.exists():
        logger.warning("[contrasts] missing preds .npz: %s", path)
        return None
    with np.load(path) as arr:
        return {
            "ci": arr["ci"],
            "target": arr["target"],
            "pred": arr["pred"],
        }


# ---------------------------------------------------------------------------
# Paired bootstrap over shared test contexts
# ---------------------------------------------------------------------------


def _pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Variance-weighted whole-map R² — parent parity (Σ_d SSE_d / Σ_d SST_d)."""
    sse = ((target - pred) ** 2).sum()
    sst = ((target - target.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - sse / (sst + 1e-30))


def _paired_bootstrap_delta_r2(
    preds_a: dict,
    preds_b: dict,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Paired bootstrap over the shared test contexts between preds_a and preds_b.

    Both arms must share the same set of test-context ids (ci). We match
    by ci, resample the shared context ids with replacement, and per
    resample compute (R²_a, R²_b, Δ = R²_a − R²_b) on the identical
    resampled context set.

    Returns:
      r2_a_point, r2_b_point, delta_point,
      delta_ci_low, delta_ci_high, r2_a_ci_low, r2_a_ci_high,
      r2_b_ci_low, r2_b_ci_high, n_shared_contexts, n_bootstrap
    """
    # Align by ci.
    by_ci_a = {int(c): i for i, c in enumerate(preds_a["ci"])}
    by_ci_b = {int(c): i for i, c in enumerate(preds_b["ci"])}
    shared_cis = sorted(set(by_ci_a.keys()) & set(by_ci_b.keys()))
    if not shared_cis:
        return {"available": False, "reason": "no shared ci between the two arms"}
    idx_a = np.array([by_ci_a[c] for c in shared_cis], dtype=np.int64)
    idx_b = np.array([by_ci_b[c] for c in shared_cis], dtype=np.int64)
    tgt_a = preds_a["target"][idx_a]
    prd_a = preds_a["pred"][idx_a]
    tgt_b = preds_b["target"][idx_b]
    prd_b = preds_b["pred"][idx_b]
    # Target-space comparability self-check.
    #
    # NOTE (issue-1491): the two arms of a CROSS-SCALE contrast do NOT share
    # targets. The target v_x is each model's OWN mean-response activation
    # profile, so across scales it lives in a different space entirely
    # (h_dim 896 at 0.5B vs 5120 at 32B) and is a different quantity even
    # when two scales happen to share a width (14B and 32B are both 5120).
    # An unguarded `tgt_a - tgt_b` therefore raises a broadcast ValueError on
    # the registered primary contrast Δ(32B − 0.5B), and silently emits a
    # meaningless number on the equal-width pairs. Both R² values are
    # scale-internal (each arm scored against its own target by _pooled_r2),
    # so the Δ itself is well-posed — only this diagnostic needed guarding.
    #
    # We therefore report the diff ONLY when the two target blocks are
    # actually shape-compatible, and label whether identity was expected.
    if tgt_a.shape == tgt_b.shape:
        tgt_diff: float | None = float(np.max(np.abs(tgt_a - tgt_b)))
        tgt_space = "same_shape"
    else:
        tgt_diff = None
        tgt_space = f"different_shape:{tgt_a.shape[1]}_vs_{tgt_b.shape[1]}"

    # Point estimates over the shared contexts (no resample).
    r2_a_point = _pooled_r2(prd_a, tgt_a)
    r2_b_point = _pooled_r2(prd_b, tgt_b)
    delta_point = r2_a_point - r2_b_point

    n = len(shared_cis)
    rng = np.random.default_rng(seed)
    r2_a_boot = np.empty(n_bootstrap, dtype=np.float64)
    r2_b_boot = np.empty(n_bootstrap, dtype=np.float64)
    delta_boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        r2_a = _pooled_r2(prd_a[sample], tgt_a[sample])
        r2_b = _pooled_r2(prd_b[sample], tgt_b[sample])
        r2_a_boot[b] = r2_a
        r2_b_boot[b] = r2_b
        delta_boot[b] = r2_a - r2_b

    return {
        "available": True,
        "n_shared_contexts": int(n),
        "n_bootstrap": int(n_bootstrap),
        "r2_a_point": r2_a_point,
        "r2_b_point": r2_b_point,
        "delta_point": delta_point,
        "r2_a_ci_low": float(np.percentile(r2_a_boot, CI_LOW)),
        "r2_a_ci_high": float(np.percentile(r2_a_boot, CI_HIGH)),
        "r2_b_ci_low": float(np.percentile(r2_b_boot, CI_LOW)),
        "r2_b_ci_high": float(np.percentile(r2_b_boot, CI_HIGH)),
        "delta_ci_low": float(np.percentile(delta_boot, CI_LOW)),
        "delta_ci_high": float(np.percentile(delta_boot, CI_HIGH)),
        "target_max_abs_diff_a_vs_b": tgt_diff,
        "target_space_relation": tgt_space,
    }


def _verdict_delta(delta_point: float, ci_low: float, ci_high: float) -> str:
    """Plan §3 registered DISJOINT + exhaustive verdict labels for Δ."""
    if delta_point > 0 and ci_low > 0:
        return "predictability-increases"
    if ci_high < 0:
        return "predictability-decreases"
    return "scale-inconclusive"


def _verdict_delta_gamma(delta_point: float, ci_low: float, ci_high: float) -> str:
    """Plan §3 registered DISJOINT + exhaustive labels for ΔΓ (linear-vs-nonlinear gap)."""
    if ci_high < 0:
        return "gap-shrinks"
    if delta_point > 0 and ci_low > 0:
        return "gap-grows"
    return "gap-inconclusive"


# Plan §4 registers THREE protections against the fixed-n / growing-d confound.
# Hidden dim grows 896 -> 5120 across the ladder while train-n is fixed at 25k,
# so the larger models are relatively MORE data-starved — which can depress the
# high-params end and MIMIC a negative scale trend (or, symmetrically, mask a
# positive one). Plan §4: "Any Δ(32B − 0.5B) whose sign is not stable under
# (a)–(c) is reported as sample-efficiency-confounded, not as a scale effect."
_CONFOUND_CONTROLS = {
    # (a) per-scale R²-vs-n sub-ladder — separates sample efficiency from asymptote.
    "n_sub_ladder": ("n_ladder", "r2_vs_n"),
    # (b) random-projection d=896 control — equalizes raw dimensionality.
    "rp896_dim_control": ("rp896",),
}


def _confound_status(fits: dict) -> dict:
    """Which plan §4 d-confound protections actually ran in THIS cut.

    Returns a REQUIRED companion to every registered verdict. The registered
    verdict vocabulary (plan §3) is emitted unchanged — this does not rewrite
    it — but a verdict produced without the protections is not readable as a
    scale effect, so the qualifier travels with it into the JSON and the figure.

    NOTE (issue-1491): `issue1491_ladder_fits.py`'s own module docstring defers
    BOTH (a) the full n-ladder and (b) the random-projection d=896 control from
    the Unit-3 cut, and its result dict emits neither key — so on this cut the
    status is `sample-efficiency-confounded` by construction.

    Detection fails toward confounded, and PRESENCE ALONE NEVER PRODUCES A
    CLEAN VERDICT: once a later round emits the control keys the status becomes
    `controls-present-sign-stability-unchecked`, not `controls-present`, because
    the Δ/ΔΓ sign-stability check plan §4 actually requires is not implemented
    here. Promoting past that string requires implementing that check.
    """
    present: dict[str, bool] = {}
    for control, keys in _CONFOUND_CONTROLS.items():
        present[control] = any(
            any(k in fj for k in keys)
            or any(k in fj.get(section, {}) for k in keys for section in ("predictors", "floors"))
            for fj in fits.values()
        )
    # (c) the 7B truncation-cost read. This requires the COMPUTED value — the
    # n=25k R² against the committed 963k anchor — not merely the existence of
    # the 7B refit cell. Keying on `"scale7_refit" in fits` marked the control
    # satisfied whenever that cell ran, while nothing anywhere computes the
    # quantity; fail toward confounded instead.
    _refit = fits.get("scale7_refit")
    present["truncation_cost_7b"] = isinstance(_refit, dict) and isinstance(
        _refit.get("truncation_cost_r2"), (int, float)
    )

    all_present = all(present.values())

    # PRESENCE IS NOT SUFFICIENT. Plan §4 makes a verdict readable as a scale
    # effect only under Δ/ΔΓ SIGN STABILITY across the controls — that check is
    # not implemented here, and no caller computes it. Emitting a bare
    # "controls-present" on key presence alone would mean that the moment a
    # later round starts emitting n_ladder/rp896, verdicts silently ship as
    # unconfounded with the sign check never having run anywhere. So the
    # strongest status this function can honestly return is the explicit
    # sign-stability-unchecked form; promoting to a clean "controls-present"
    # requires implementing the sign check.
    if all_present:
        status = "controls-present-sign-stability-unchecked"
        note = (
            "All plan §4 d-confound control KEYS are present in this cut, but the "
            "Δ/ΔΓ sign-stability check those controls exist to support is NOT "
            "implemented — this verdict is not yet readable as a scale effect."
        )
    else:
        status = "sample-efficiency-confounded"
        note = (
            "Plan §4: train-n is fixed at 25k while hidden dim grows 896→5120, so at "
            "fixed n the larger models are relatively more data-starved. Absent the "
            "registered protections, a registered Δ/ΔΓ verdict is NOT readable as a "
            "scale effect — report it as sample-efficiency-confounded."
        )
    return {
        "status": status,
        "controls_present": present,
        "sign_stability_checked": False,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Assemble per-scale summary
# ---------------------------------------------------------------------------


def _per_scale_summary(fits: dict[str, dict], preds_ridge: dict[str, dict]) -> list[dict]:
    """Assemble a per-scale summary row for the ladder plot."""
    rows = []
    for slug in SCALE_ORDER:
        fits_j = fits.get(slug)
        if fits_j is None:
            rows.append(
                {
                    "slug": slug,
                    "display": SCALE_DISPLAY[slug],
                    "params_b": SCALE_PARAMS_B[slug],
                    "available": False,
                }
            )
            continue
        preds = fits_j.get("predictors", {})
        floors = fits_j.get("floors", {})
        ceiling = fits_j.get("ceiling_two_draw", {})
        ridge_r2 = preds.get("ridge", {}).get("test_r2")
        # The nonlinear arm is PRE-REGISTERED upstream, not selected here.
        #
        # NOTE (issue-1491): this previously re-selected `max(..., test_r2)`
        # under a comment claiming it "matches the parent's val-selected
        # policy" — it did not; it was selection-on-outcome (picking the arm on
        # the same test split whose R² is then reported), and it duplicated the
        # identical defect in the fits driver. `issue1491_ladder_fits.py` now
        # pre-registers MLP-w8192 (the arm the Goal's own 0.754-vs-0.810 figure
        # is defined against) and records the choice in `preds_paths`, so the
        # only correct thing to do here is READ that choice.
        nl_kind = fits_j.get("preds_paths", {}).get("registered_nonlinear_kind")
        nl_r2 = preds.get(nl_kind, {}).get("test_r2") if nl_kind else None
        if nl_kind is None and any(k in preds for k in ("mlp_w8192", "mlp_w32768", "krr_nystrom")):
            # Nonlinear arms ran but the fits driver recorded no registered
            # choice — a provenance gap, not something to paper over by
            # re-selecting. Report the scale with no Γ rather than inventing one.
            logger.warning(
                "[contrasts] %s: nonlinear arms present but no "
                "registered_nonlinear_kind recorded — Γ omitted for this scale",
                slug,
            )
        gamma = (nl_r2 - ridge_r2) if (ridge_r2 is not None and nl_r2 is not None) else None
        rows.append(
            {
                "slug": slug,
                "display": SCALE_DISPLAY[slug],
                "params_b": SCALE_PARAMS_B[slug],
                "available": True,
                "primary_layer_index": fits_j.get("primary_layer_index"),
                "n_realized_test": fits_j.get("n_realized", {}).get("test_1000"),
                "ridge_test_r2": ridge_r2,
                "registered_nonlinear_test_r2": nl_r2,
                "registered_nonlinear_kind": nl_kind,
                "gap_gamma": gamma,
                "floors": {name: f["test_r2"] for name, f in floors.items()},
                "ceiling": ceiling,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Hero figure — R² vs log-params, one panel, ridge + registered nonlinear
# ---------------------------------------------------------------------------


def _write_hero_figure(rows: list[dict], contrasts: dict, out_path: Path) -> Path | None:
    """R² vs log-params ladder — one panel. Ridge + registered nonlinear
    lines; per-scale bootstrap CIs shaded; floors as light lower bands;
    ceiling as a dashed upper band; annotated 7B anchor point.

    Falls back to a WARN if matplotlib is unavailable (never a hard fail
    of the driver — the JSON is the primary deliverable)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from explore_persona_space.analysis.paper_plots import setup_paper_style  # type: ignore

        setup_paper_style()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[contrasts] hero figure skipped — matplotlib/paper_plots unavailable: %s", e
        )
        return None

    xs, ridge_ys, nl_ys, ceilings, floor_shuffled, floor_mean = [], [], [], [], [], []
    for r in rows:
        if not r["available"]:
            continue
        xs.append(r["params_b"])
        ridge_ys.append(r["ridge_test_r2"])
        nl_ys.append(r["registered_nonlinear_test_r2"])
        c = r.get("ceiling")
        ceilings.append(
            c.get("ceiling_var_weighted_r") if isinstance(c, dict) and c.get("available") else None
        )
        f = r.get("floors", {})
        floor_shuffled.append(f.get("shuffled_pairing"))
        floor_mean.append(f.get("train_mean"))

    if not xs:
        logger.warning("[contrasts] hero figure skipped — no per-scale rows available")
        return None

    xs = np.array(xs)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ridge_arr = np.array(ridge_ys, dtype=float)
    nl_arr = np.array([y if y is not None else np.nan for y in nl_ys], dtype=float)
    ax.plot(xs, ridge_arr, "o-", label="ridge", color="#1f77b4", lw=2)
    if not np.all(np.isnan(nl_arr)):
        ax.plot(xs, nl_arr, "s-", label="registered nonlinear (MLP-w8192)", color="#d62728", lw=2)
    # Floors + ceiling as light bands.
    fs = np.array([y if y is not None else np.nan for y in floor_shuffled], dtype=float)
    fm = np.array([y if y is not None else np.nan for y in floor_mean], dtype=float)
    if not np.all(np.isnan(fs)):
        ax.plot(xs, fs, ":", label="shuffled-pairing null", color="gray", alpha=0.7)
    if not np.all(np.isnan(fm)):
        ax.plot(xs, fm, ":", label="train-mean floor", color="lightgray", alpha=0.9)
    cs = np.array([y if y is not None else np.nan for y in ceilings], dtype=float)
    if not np.all(np.isnan(cs)):
        ax.plot(xs, cs, "--", label="two-draw ceiling", color="darkgreen", alpha=0.6)
    # 7B anchor annotation (the committed n1M ridge 0.754 read).
    ax.axhline(y=0.754, color="#1f77b4", linestyle="--", alpha=0.3)
    ax.annotate(
        "#779 n1M anchor\nridge 0.754 @ n=963k",
        xy=(7.0, 0.754),
        xytext=(3.5, 0.68),
        fontsize=8,
        color="#1f77b4",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "alpha": 0.4},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Params (B, log scale)")
    ax.set_ylabel("Held-out variance-weighted test R²")
    ax.set_title("Context → answer map fidelity vs model scale (Qwen-2.5-Instruct)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Primary contrast — carried into the figure's meta sidecar below, NOT
    # rendered onto the plot (see the note directly following).
    delta = contrasts.get("delta_r2_ridge_32B_vs_05B", {})

    # NO on-plot verdict annotation, deliberately.
    #
    # It used to render "Δ = ... / Verdict: {v}" as a text box here. Two
    # reasons that is wrong. (1) It printed the registered verdict WITHOUT the
    # sample-efficiency confound qualifier, on the single surface a human reads
    # first — so on this cut, which is confounded by construction (fixed n with
    # d growing 896→5120), a PNG reader saw a bare "predictability-increases".
    # That is exactly the leak the qualifier exists to prevent, surviving on the
    # most-read surface. (2) The project figure convention bans effect-size
    # labels and explanatory text overlays on plots (see the sibling note
    # below). The numbers live in the JSON and in the figure's meta sidecar,
    # both of which carry `confound_controls` alongside them.

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Write a small meta.json sidecar (paper_plots convention).
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "figure": str(out_path),
                "x_axis": "params (B, log scale)",
                "y_axis": "held-out variance-weighted test R²",
                # Travels with the figure so a caption cannot present the trend
                # without the plan §4 qualifier (no on-plot annotation — the
                # project figure convention keeps overlays off the axes).
                "confound_controls": contrasts.get("confound_controls", {}),
                "series": {
                    "ridge": ridge_arr.tolist(),
                    "registered_nonlinear": nl_arr.tolist(),
                    "shuffled_pairing_floor": fs.tolist(),
                    "train_mean_floor": fm.tolist(),
                    "two_draw_ceiling": cs.tolist(),
                },
                "scales_b": xs.tolist(),
                "anchor_line": {"ridge_at_7B_n963k": 0.754},
                "contrast_delta": delta,
            },
            fh,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fits-dir",
        type=Path,
        default=Path("eval_results/issue_1491/scale_ladder"),
        help="dir holding fits_<slug>.json (Unit 3 output)",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("data/issue_1491/preds"),
        help="dir holding <slug>_test_preds_{ridge,registered_nonlinear}.npz (Unit 3 output)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_results/issue_1491/scale_ladder/ladder_contrasts.json"),
    )
    ap.add_argument(
        "--out-figure",
        type=Path,
        default=Path("figures/issue_1491/hero_r2_vs_scale.png"),
    )
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load all per-scale fits + preds.
    fits: dict[str, dict] = {}
    preds_ridge: dict[str, dict] = {}
    preds_nl: dict[str, dict] = {}
    for slug in SCALE_ORDER:
        fj = _load_fits_json(args.fits_dir, slug)
        if fj is not None:
            fits[slug] = fj
        pr = _load_preds_npz(args.preds_dir, slug, "ridge")
        if pr is not None:
            preds_ridge[slug] = pr
        pn = _load_preds_npz(args.preds_dir, slug, "registered_nonlinear")
        if pn is not None:
            preds_nl[slug] = pn

    logger.info(
        "[contrasts] loaded %d/%d scale fits, %d/%d ridge-preds, %d/%d nl-preds",
        len(fits),
        len(SCALE_ORDER),
        len(preds_ridge),
        len(SCALE_ORDER),
        len(preds_nl),
        len(SCALE_ORDER),
    )

    # 2. Primary contrast — Δ = R²(32B) − R²(0.5B) on ridge.
    delta_primary = {"available": False, "reason": "missing 0.5B or 32B ridge preds"}
    if "scale05" in preds_ridge and "scale32" in preds_ridge:
        b = _paired_bootstrap_delta_r2(
            preds_ridge["scale32"],  # A = the high-scale arm
            preds_ridge["scale05"],  # B = the low-scale arm
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        b["arm"] = "ridge"
        b["definition"] = (
            "Δ = ridge R²(32B) − ridge R²(0.5B), primary layer, matched-n 25k, context arm"
        )
        if b.get("available"):
            b["verdict"] = _verdict_delta(b["delta_point"], b["delta_ci_low"], b["delta_ci_high"])
        delta_primary = b

    # 3. Secondary contrast — ΔΓ = Γ(32B) − Γ(0.5B), where Γ = registered
    # nonlinear − ridge. Requires BOTH ridge and nl preds for BOTH scales.
    delta_gamma = {"available": False, "reason": "missing ridge or nl preds for 0.5B or 32B"}
    if all(k in preds_ridge for k in ("scale05", "scale32")) and all(
        k in preds_nl for k in ("scale05", "scale32")
    ):
        # Per-draw gap = nl_R2 − ridge_R2 on the SAME resampled contexts.
        #
        # PAIR-ANCHORING (plan §3: "both arms of every registered pair row come
        # from the same resampled context ids") binds along BOTH axes:
        #   (a) WITHIN a scale, the ridge and nonlinear arms share the resample;
        #   (b) ACROSS scales, draw b of Γ(32B) and draw b of Γ(0.5B) must use
        #       the SAME resampled context ids.
        #
        # NOTE (issue-1491): (a) held, but (b) was VIOLATED — a single `rng` was
        # advanced sequentially through the per-scale loop, so scale32 draw b and
        # scale05 draw b resampled different rows. ΔΓ then differenced two
        # INDEPENDENT bootstrap distributions, whose variances ADD rather than
        # partially cancelling, inflating the registered CI (biasing the verdict
        # toward "Gap-inconclusive"). Fixed by drawing ONE resample matrix over
        # the context ids common to all four pred sets and mapping it into each
        # scale's own row ordering.
        #
        # The two scales' targets live in DIFFERENT spaces (h_dim 896 vs 5120),
        # which is fine here: each _pooled_r2 is scale-internal and only the ROW
        # SELECTION is shared across scales.
        ci_sets = []
        for slug in ("scale05", "scale32"):
            ci_sets.append({int(c) for c in preds_ridge[slug]["ci"]})
            ci_sets.append({int(c) for c in preds_nl[slug]["ci"]})
        shared_all = sorted(set.intersection(*ci_sets))
        n = len(shared_all)

        if n == 0:
            delta_gamma = {
                "available": False,
                "reason": "no context ids shared across all four (scale, arm) pred sets",
            }
        else:
            # ONE resample matrix — shared by both scales AND both arms.
            rng = np.random.default_rng(args.seed + 1)
            samples = rng.integers(0, n, size=(args.n_bootstrap, n))

            gap_draws: dict[str, np.ndarray] = {}
            gap_points: dict[str, float] = {}
            for slug in ("scale05", "scale32"):
                pr = preds_ridge[slug]
                pn = preds_nl[slug]
                by_ci_r = {int(c): i for i, c in enumerate(pr["ci"])}
                by_ci_n = {int(c): i for i, c in enumerate(pn["ci"])}
                idx_r = np.array([by_ci_r[c] for c in shared_all], dtype=np.int64)
                idx_n = np.array([by_ci_n[c] for c in shared_all], dtype=np.int64)
                tgt = pr["target"][idx_r]
                prd_r = pr["pred"][idx_r]
                prd_n = pn["pred"][idx_n]
                # WITHIN a scale the two arms share targets by construction (same
                # model, same test contexts) — unlike the cross-scale case, this
                # identity IS expected, so assert it.
                tgt_n = pn["target"][idx_n]
                assert tgt.shape == tgt_n.shape and np.allclose(tgt, tgt_n, atol=1e-5), (
                    f"{slug}: ridge and nonlinear arms disagree on targets for shared ci"
                )
                # Point estimate on the UNRESAMPLED shared set — never the mean of
                # bootstrap draws (that is a bootstrap-biased estimator, and it is
                # how the primary Δ is already computed).
                gap_points[slug] = _pooled_r2(prd_n, tgt) - _pooled_r2(prd_r, tgt)
                gaps = np.empty(args.n_bootstrap, dtype=np.float64)
                for b_i in range(args.n_bootstrap):
                    s = samples[b_i]
                    gaps[b_i] = _pooled_r2(prd_n[s], tgt[s]) - _pooled_r2(prd_r[s], tgt[s])
                gap_draws[slug] = gaps

            delta_gamma_draws = gap_draws["scale32"] - gap_draws["scale05"]
            gap_point = float(gap_points["scale32"] - gap_points["scale05"])
            lo = float(np.percentile(delta_gamma_draws, CI_LOW))
            hi = float(np.percentile(delta_gamma_draws, CI_HIGH))
            delta_gamma = {
                "available": True,
                "arm": "registered_nonlinear_gap",
                "definition": (
                    "ΔΓ = Γ(32B) − Γ(0.5B), Γ(s) = registered nonlinear R²(s) − ridge R²(s); "
                    "pair-anchored across scales AND arms (one shared resample matrix)"
                ),
                "n_bootstrap": int(args.n_bootstrap),
                "n_shared_contexts": int(n),
                "gamma_point_by_scale": {k: float(v) for k, v in gap_points.items()},
                "delta_gamma_point": gap_point,
                "delta_gamma_ci_low": lo,
                "delta_gamma_ci_high": hi,
                "verdict": _verdict_delta_gamma(gap_point, lo, hi),
            }

    # 4. Descriptive Spearman monotonicity (all 6 scales, ridge).
    spearman = {"available": False}
    ridge_by_slug = {
        slug: fits[slug]["predictors"]["ridge"]["test_r2"]
        for slug in SCALE_ORDER
        if slug in fits and "ridge" in fits[slug].get("predictors", {})
    }
    if len(ridge_by_slug) >= 3:
        xs_b = np.array([SCALE_PARAMS_B[s] for s in ridge_by_slug])
        ys_r = np.array([ridge_by_slug[s] for s in ridge_by_slug])
        # Rank correlation via scipy. (The comment here previously claimed the
        # opposite — "without scipy, compute by hand" — directly above a scipy
        # import.) Descriptive only, never a registered verdict: with at most
        # 6 scales the p-value is not meaningful and is reported for context.
        from scipy.stats import spearmanr  # type: ignore

        rho, p = spearmanr(xs_b, ys_r)
        spearman = {
            "available": True,
            "n_scales": int(len(ridge_by_slug)),
            "spearman_rho": float(rho),
            "spearman_pvalue": float(p),
            "note": "descriptive — never a registered verdict (plan §3)",
        }

    # 5. Ceiling-normalized per-scale ridge R² (companion read for plan §4 H4 screen).
    ceiling_normalized = {}
    for slug in SCALE_ORDER:
        if slug in fits:
            fj = fits[slug]
            ceil = fj.get("ceiling_two_draw", {})
            ridge_r2 = fj.get("predictors", {}).get("ridge", {}).get("test_r2")
            if isinstance(ceil, dict) and ceil.get("available") and ridge_r2 is not None:
                c_val = ceil.get("ceiling_var_weighted_r")
                # The normalizer must be STRICTLY POSITIVE. The prior guard was
                # abs(c_val) > 1e-6, which admitted a NEGATIVE ceiling: the two
                # draws anti-correlating is a broken measurement, not a valid
                # denominator, and dividing by it emits a sign-flipped number
                # presented as a fraction-of-ceiling. Near-zero is excluded for
                # the same reason it always was (the ratio explodes).
                if isinstance(c_val, (int, float)) and c_val > 1e-6:
                    ceiling_normalized[slug] = {
                        "ridge_r2": ridge_r2,
                        "ceiling": c_val,
                        "normalized": ridge_r2 / c_val,
                    }
                else:
                    # Record the skip rather than silently omitting the scale —
                    # an absent row and a non-positive ceiling are different facts.
                    ceiling_normalized[slug] = {
                        "ridge_r2": ridge_r2,
                        "ceiling": c_val,
                        "normalized": None,
                        "skipped": "ceiling not strictly positive — unusable as a normalizer",
                    }

    # 6. Per-scale summary rows (for the hero figure + downstream analyses).
    per_scale_rows = _per_scale_summary(fits, preds_ridge)

    # 7. Assemble the contrasts JSON.
    # Plan §4 d-confound qualifier — a REQUIRED companion to every registered
    # verdict (see _confound_status). Attached to both contrasts AND surfaced
    # top-level so no consumer can read a verdict without it.
    confound = _confound_status(fits)
    # Always true by construction now (a clean "controls-present" is
    # unreachable — see _confound_status), so warn on the ACTUAL status rather
    # than asserting the confounded one: with all control keys present the
    # status is controls-present-sign-stability-unchecked, and claiming
    # "reported as sample-efficiency-confounded" there would misdescribe the
    # JSON the run actually emits.
    logger.warning(
        "[contrasts] d-confound status=%s %s — a registered verdict is NOT readable "
        "as a scale effect until the plan §4 sign-stability check runs",
        confound["status"],
        confound["controls_present"],
    )
    for _c in (delta_primary, delta_gamma):
        if isinstance(_c, dict) and _c.get("available"):
            _c["confound_status"] = confound["status"]

    contrasts = {
        "scale_order": SCALE_ORDER,
        "scale_params_b": SCALE_PARAMS_B,
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.seed),
        "ci_level_percent": [CI_LOW, CI_HIGH],
        "confound_controls": confound,
        "delta_r2_ridge_32B_vs_05B": delta_primary,
        "delta_gamma_32B_vs_05B": delta_gamma,
        "spearman_monotonicity": spearman,
        "ceiling_normalized_r2": ceiling_normalized,
        "per_scale_rows": per_scale_rows,
    }

    # 8. Write.
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(contrasts, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("[contrasts] wrote %s", args.out_json)

    # 9. Hero figure (fail-soft; JSON is the primary deliverable).
    fig_path = _write_hero_figure(per_scale_rows, contrasts, args.out_figure)
    if fig_path is not None:
        logger.info("[contrasts] wrote hero figure: %s", fig_path)

    # 10. One-line stdout summary for the smoke run's per-phase artifact digest.
    d = delta_primary
    dg = delta_gamma
    if d.get("available"):
        v = d["verdict"]
        # The confound qualifier rides EVERY verdict surface, stdout included:
        # a log reader must not see a bare "predictability-increases" on a cut
        # that is sample-efficiency-confounded by construction.
        print(
            f"OK — Δ_primary = {d['delta_point']:+.3f} [{d['delta_ci_low']:+.3f}, {d['delta_ci_high']:+.3f}] "
            f"verdict={v} confound={d.get('confound_status', 'unknown')}; wrote {args.out_json}"
        )
    else:
        print(
            f"OK — insufficient inputs for primary Δ ({d.get('reason', '?')}); wrote {args.out_json}"
        )
    if dg.get("available"):
        print(
            f"    ΔΓ_secondary = {dg['delta_gamma_point']:+.3f} "
            f"[{dg['delta_gamma_ci_low']:+.3f}, {dg['delta_gamma_ci_high']:+.3f}] "
            f"verdict={dg['verdict']} confound={dg.get('confound_status', 'unknown')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
