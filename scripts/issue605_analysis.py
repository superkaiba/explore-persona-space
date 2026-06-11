"""Issue #605 Phase 7 — registered statistics, pinned fits, decision lattice,
figures. Runs OFF-POD on the VM against committed per-cell JSONs.

REGISTERED statistics (plan section 3 — verdicts key ONLY on these), per
(family x DV-class x space):

  A. Pooled matched-similarity partial Spearman
     rho(DV, prior | continuous pair-cosine + band FE + source mean DV),
     1000-rep context-cluster bootstrap, Holm over the 2 shift spaces.
  B. Held-out delta-CV-R^2 — grouped 5-fold CV by context (persona for
     facts): model 1 = {pair cosine + source FE}, model 2 = {+ prior};
     out-of-fold delta-CV-R^2 with context-cluster bootstrap CI re-running
     the FULL CV per resample (unique-context resampling; every duplicated
     cluster's cells stay within a single fold).

PINNED primary fits (sign routing): log-prob space = Tobit/censored over ALL
cells (saturation-aware right censoring); EOS-margin space = rank fit over
ALL cells. The saturation-EXCLUDED fit is a robustness column only.

Decision lattice (plan section 3) is evaluated sign-routed per family; any
non-enumerated outcome ships as indeterminate. Diagnostics per plan 11.5:
base-term correlation + base-covariate fit, split-probe complement,
affordance-class stratified read, prompt-length covariate, cov_negsim (both
families), saturation map + slice coverage per prior tercile, collinearity
gate with tercile-contrast fallback, per-band descriptive partials with
realized CIs + MDE, legacy-anchor drift, centered-bank cosine sensitivity,
fact TF-proxy validation (rho > 0.4 vs judged leak).

``--synthesize-stub DIR`` writes a tiny synthetic artifact tree (schema-true,
planted positive prior->shift effect) so the WHOLE script smoke-runs on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, rankdata, spearmanr  # noqa: E402

from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID  # noqa: E402

logger = logging.getLogger("issue605.analysis")

# Production expected-source set for the marker coverage assert: the SAME
# registered 16 source cids the eval dispatcher trains/evals
# (issue605_eval_marker.py: ``SOURCES_ALL = list(CONDITIONS_BY_ID)``).
# Deriving the denominator from the observed frame let a source with ZERO
# per_cell files vanish silently from the coverage assert (round-5 blocker
# wide-analysis-source-set-blindspot); production now always asserts against
# this registered set, and only ``--synthesize-stub`` gates the stub override.
REGISTERED_SOURCES: list[str] = list(CONDITIONS_BY_ID)
# Stub trees are deliberately 4-source (kept tiny for the CPU smoke).
STUB_SOURCES: list[str] = ["A1", "A2", "B1", "D1"]
# Parent figure home. _marker_figures/_fact_figures write FIXED filenames
# (hero_shift_vs_prior_by_band, ...), so amendment/stub runs must never
# default here (concern wide-analysis-figures-dir-clobber).
PARENT_FIGURES_DIR = Path("figures/issue_605")

ANALYSIS_SEED = 42
SATURATION_LOGP = -0.1
SATURATION_ARGMAX = 0.92
COLLINEARITY_GATE = 0.6
PROXY_RHO_MIN = 0.4
SURVIVES_RHO_UPPER = 0.2  # plan section 3 precision clause


def _repro_meta(extra: dict | None = None) -> dict:
    from issue532_predictor_stress import _reproducibility_metadata

    return _reproducibility_metadata(extra)


# ---------------------------------------------------------------------------
# Statistics machinery
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _design(covars: list[np.ndarray], dummies: list[np.ndarray], n: int) -> np.ndarray:
    cols = [rankdata(c) for c in covars] + list(dummies) + [np.ones(n)]
    return np.column_stack(cols)


def partial_spearman(
    y: np.ndarray, x: np.ndarray, covars: list[np.ndarray], dummies: list[np.ndarray]
) -> float:
    """Partial Spearman = Pearson on rank residuals after OLS on the
    rank-transformed continuous covariates + raw dummies + intercept."""
    n = len(y)
    if n < 5:
        return float("nan")
    C = _design(covars, dummies, n)
    ry, rx = rankdata(y), rankdata(x)
    res_y = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    res_x = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    return _pearson(res_y, res_x)


def _band_dummies(bands: pd.Series) -> list[np.ndarray]:
    cats = sorted(bands.unique())
    return [(bands == c).to_numpy(dtype=float) for c in cats[1:]]  # drop-first


def _fe_dummies(values: pd.Series) -> list[np.ndarray]:
    cats = sorted(values.unique())
    return [(values == c).to_numpy(dtype=float) for c in cats[1:]]


def pooled_partial(frame: pd.DataFrame, dv: str, prior_col: str, sim_col: str) -> float:
    """Registered statistic A on one frame."""
    src_mean = frame.groupby("source")[dv].transform("mean").to_numpy()
    return partial_spearman(
        frame[dv].to_numpy(),
        frame[prior_col].to_numpy(),
        covars=[frame[sim_col].to_numpy(), src_mean],
        dummies=_band_dummies(frame["band"]),
    )


def _grouped_cv_r2(frame: pd.DataFrame, dv: str, x_cols: list[str], group_col: str) -> float:
    """Out-of-fold R^2, grouped K<=5-fold by ``group_col``; FE columns are the
    frame's pre-built source dummies (``fe_*``)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold

    fe_cols = [c for c in frame.columns if c.startswith("fe_")]
    X = frame[x_cols + fe_cols].to_numpy(dtype=float)
    y = frame[dv].to_numpy(dtype=float)
    groups = frame[group_col].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2 or len(y) < 5:
        return float("nan")
    preds = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups=groups):
        m = LinearRegression().fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")


def delta_cv_r2(frame: pd.DataFrame, dv: str, prior_col: str, sim_col: str, group_col: str) -> dict:
    """Registered statistic B (+ the reverse ladder) on one frame."""
    r2_sim = _grouped_cv_r2(frame, dv, [sim_col], group_col)
    r2_full = _grouped_cv_r2(frame, dv, [sim_col, prior_col], group_col)
    r2_prior = _grouped_cv_r2(frame, dv, [prior_col], group_col)
    return {
        "r2_sim_only": r2_sim,
        "r2_sim_plus_prior": r2_full,
        "delta_cv_r2_prior_beyond_sim": r2_full - r2_sim,
        "r2_prior_only": r2_prior,
        "delta_cv_r2_sim_beyond_prior_reverse": r2_full - r2_prior,
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_col: str,
    stat_fn,
    n_boot: int,
    seed: int = ANALYSIS_SEED,
) -> np.ndarray:
    """Unique-cluster resampling. ``_boot_cluster`` ids are per-instance
    bookkeeping ONLY — any grouped-CV statistic run inside a resample MUST
    group folds by the ORIGINAL cluster column, never ``_boot_cluster``:
    fresh ids keep each resampled COPY in one fold but let two copies of the
    same original context straddle train/test folds (identical covariates AND
    DV), which makes the CV bootstrap CI anti-conservative (round-1 review
    concern ``cv-bootstrap-cluster-leakage``)."""
    clusters = np.array(sorted(frame[cluster_col].unique()))
    rng = np.random.default_rng(seed)
    by_cluster = {c: frame[frame[cluster_col] == c] for c in clusters}
    out = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for k, c in enumerate(sampled):
            sub = by_cluster[c].copy()
            sub["_boot_cluster"] = f"bc{k}"
            parts.append(sub)
        out.append(stat_fn(pd.concat(parts, ignore_index=True)))
    return np.array(out, dtype=float)


def _boot_summary(samples: np.ndarray, point: float) -> dict:
    s = samples[~np.isnan(samples)]
    if len(s) < 10:
        return {"point": point, "ci95": [float("nan"), float("nan")], "p_two_sided": float("nan")}
    lo, hi = np.percentile(s, [2.5, 97.5])
    p = 2 * min(float(np.mean(s <= 0)), float(np.mean(s >= 0)))
    return {
        "point": float(point),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(min(1.0, max(p, 1.0 / len(s)))),
        "n_boot_effective": len(s),
    }


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    m = len(items)
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def tobit_prior_coef(frame: pd.DataFrame, dv: str, prior_col: str, sim_col: str) -> float:
    """Right-censored (saturation-aware) Tobit MLE over ALL cells; returns the
    signed prior coefficient (the pinned log-prob-space primary fit)."""
    from scipy.optimize import minimize

    fe_cols = [c for c in frame.columns if c.startswith("fe_")]
    X = np.column_stack(
        [
            np.ones(len(frame)),
            frame[sim_col].to_numpy(dtype=float),
            frame[fe_cols].to_numpy(dtype=float) if fe_cols else np.zeros((len(frame), 0)),
            frame[prior_col].to_numpy(dtype=float),
        ]
    )
    y = frame[dv].to_numpy(dtype=float)
    cens = frame["saturated"].to_numpy(dtype=bool)
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta0
    x0 = np.concatenate([beta0, [np.log(max(np.std(resid), 1e-3))]])

    def nll(params: np.ndarray) -> float:
        beta, log_sigma = params[:-1], params[-1]
        sigma = np.exp(log_sigma)
        z = (y - X @ beta) / sigma
        ll = np.where(cens, norm.logsf(z), norm.logpdf(z) - log_sigma)
        return -float(np.sum(ll))

    res = minimize(nll, x0, method="BFGS", options={"maxiter": 500})
    return float(res.x[-2])  # prior is the LAST design column


def rank_fit_prior_coef(frame: pd.DataFrame, dv: str, prior_col: str, sim_col: str) -> float:
    """Pinned EOS-margin primary: OLS on rank-transformed DV/covariates
    (all cells, non-saturating space); returns the signed prior coefficient."""
    fe_cols = [c for c in frame.columns if c.startswith("fe_")]
    X = np.column_stack(
        [
            np.ones(len(frame)),
            rankdata(frame[sim_col].to_numpy()),
            frame[fe_cols].to_numpy(dtype=float) if fe_cols else np.zeros((len(frame), 0)),
            rankdata(frame[prior_col].to_numpy()),
        ]
    )
    y = rankdata(frame[dv].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[-1])


def per_band_partials(
    frame: pd.DataFrame, dv: str, prior_col: str, sim_col: str, n_boot: int
) -> dict:
    """DESCRIPTIVE per-band partials with realized CIs + Fisher-z MDE."""
    out = {}
    for band in sorted(frame["band"].unique()):
        sub = frame[frame["band"] == band]
        n_ctx = sub["context"].nunique()

        def stat(f: pd.DataFrame) -> float:
            src_mean = f.groupby("source")[dv].transform("mean").to_numpy()
            return partial_spearman(
                f[dv].to_numpy(),
                f[prior_col].to_numpy(),
                covars=[f[sim_col].to_numpy(), src_mean],
                dummies=[],
            )

        point = stat(sub)
        boots = cluster_bootstrap(sub, "context", stat, n_boot) if n_ctx >= 4 else np.array([])
        k_covars = 2
        mde = float(np.tanh(1.96 / np.sqrt(max(n_ctx - 3 - k_covars, 1))))
        out[band] = {
            **_boot_summary(boots, point),
            "n_contexts": int(n_ctx),
            "n_cells": len(sub),
            "mde_abs_rho_context_level": mde,
            "descriptive_only": True,
        }
    return out


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------
def _resolve_selected_panel(sel: dict, what: str) -> set[str]:
    """Panel context set from a selection dict, honoring gate + descope:
    gate_pass=true -> full panel; gate_pass=false -> REFUSE unless the JSON
    carries the recorded pre-registered descope, then the surviving-band
    subset (round-1 blocker ``panel-gate-not-enforced`` /
    ``marker-analysis-panel-filter``)."""
    if sel.get("gate_pass", False):
        return set(sel["panel"])
    desc = sel.get("descope") or {}
    assert desc.get("active"), (
        f"{what} has gate_pass=false and no recorded descope — the selection gate blocks "
        "this analysis (plan section 7 gate 2); re-run selection after the pre-registered "
        "expansion round or with --allow-descope"
    )
    logger.warning(
        "%s: descoped panel in effect (surviving bands: %s, %d contexts)",
        what,
        desc["surviving_bands"],
        len(desc["panel_descoped"]),
    )
    return set(desc["panel_descoped"])


def build_marker_frame(
    out_root: Path,
    selection_path: Path | None = None,
    expected_sources: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Cell-level marker frame from per-cell JSONs + pair table + selection.

    Only cells whose context is in the SELECTED panel (or recorded descope
    subset) enter the frame — stale smoke/expansion cells in the same output
    root never reach the registered statistics — and coverage of
    ``expected_sources x panel`` is asserted (plan section 6). The default
    ``expected_sources=None`` resolves to the REGISTERED 16 source cids the
    eval dispatcher runs (``REGISTERED_SOURCES``), so a source with ZERO
    per_cell files is a named shortfall, never a silently shrunk denominator;
    sources on disk OUTSIDE the expected set also fail loud. The stub path
    passes ``STUB_SOURCES`` explicitly (gated on ``--synthesize-stub`` in
    ``main``). ``selection_path`` overrides the default parent selection JSON
    (the amendment passes the WIDE selection; coverage then demands
    expected_sources x realized wide panel)."""
    sel_path = selection_path or (out_root / "panel" / "marker_panel_selection.json")
    sel = json.loads(sel_path.read_text())
    table = json.loads((out_root / "panel" / "marker_pair_table.json").read_text())
    pair = {(r["source_cid"], r["context_label"]): r for r in table["rows"]}
    edges = sel["band_edges_terciles"]
    panel_set = _resolve_selected_panel(sel, sel_path.name)
    trained_dir = out_root / "marker" / "per_cell_trained"
    base_dir = out_root / "marker" / "per_cell_base"
    gen_dir = out_root / "marker" / "gen"
    prior_dir = out_root / "panel" / "marker_measure" / "prior"

    half_a_cache: dict[str, float] = {}

    def prior_half_a(ctx: str) -> float:
        """Half-A (even-index per-q) prior from the P1 per-q reads; NaN for
        legacy contexts whose prior was reused as a #532 scalar (no per-q)."""
        if ctx not in half_a_cache:
            p = prior_dir / f"{ctx}.json"
            if p.exists():
                per_q = json.loads(p.read_text())["per_q"]
                half_a_cache[ctx] = float(np.mean([r["logp_marker"] for r in per_q[0::2]]))
            else:
                half_a_cache[ctx] = float("nan")
        return half_a_cache[ctx]

    rows = []
    skipped: list[tuple[str, str, str]] = []
    seen_cells: set[tuple[str, str]] = set()
    for f in sorted(trained_dir.glob("*.json")):
        src, ctx = f.stem.split("__", 1)
        if ctx not in panel_set:
            continue  # stale smoke/expansion cell — never enters registered statistics
        b_path = base_dir / f.name
        if not b_path.exists():
            skipped.append((src, ctx, "missing per_cell_base"))
            continue
        if (src, ctx) not in pair:
            skipped.append((src, ctx, "missing pair-table row"))
            continue
        t = json.loads(f.read_text())
        b = json.loads(b_path.read_text())
        pr = pair[(src, ctx)]
        t_q, b_q = t["per_q"], b["per_q"]
        assert len(t_q) == len(b_q), f.name
        for tq, bq in zip(t_q, b_q, strict=True):
            assert tq["slot_kind"] == bq["slot_kind"], f.name
            assert tq["n_truncated_tokens"] == bq["n_truncated_tokens"], f.name
        dlogp_q = [tq["logp_marker"] - bq["logp_marker"] for tq, bq in zip(t_q, b_q, strict=True)]
        dmargin_q = [
            (tq["z_marker"] - tq["z_eos"]) - (bq["z_marker"] - bq["z_eos"])
            for tq, bq in zip(t_q, b_q, strict=True)
        ]
        dz_q = [tq["z_marker"] - bq["z_marker"] for tq, bq in zip(t_q, b_q, strict=True)]
        g_path = gen_dir / f.name
        emit = (
            json.loads(g_path.read_text())["summary"]["in_R_emission_rate"]
            if g_path.exists()
            else float("nan")
        )
        trained_logp = float(np.mean([q["logp_marker"] for q in t_q]))
        argmax_rate = float(np.mean([q["argmax_id"] == 83399 for q in t_q]))
        cos = pr["cos_l21"]
        band = "band_lo" if cos <= edges[0] else ("band_mid" if cos <= edges[1] else "band_hi")
        rows.append(
            {
                "source": src,
                "context": ctx,
                "band": band,
                "cos": cos,
                "gkl": pr["gkl_l22"],
                "cos_centered": pr["cos_centered_bank"],
                "prior": pr["prior_logp"],
                "negsim": pr["sim_to_nearest_negative"],
                "affordance_class": pr["affordance_class"],
                "content_class": pr["content_class"],
                "is_legacy": pr["is_legacy"],
                "dlogp": float(np.mean(dlogp_q)),
                "dmargin": float(np.mean(dmargin_q)),
                "dz_marker": float(np.mean(dz_q)),
                "dlogZ": float(
                    np.mean([tq["logZ"] - bq["logZ"] for tq, bq in zip(t_q, b_q, strict=True)])
                ),
                "trained_logp": trained_logp,
                "base_logp_matched": float(np.mean([q["logp_marker"] for q in b_q])),
                "emission_rate": emit,
                "argmax_rate": argmax_rate,
                "saturated": bool(
                    trained_logp > SATURATION_LOGP and argmax_rate >= SATURATION_ARGMAX
                ),
                "n_emitting_slots": int(sum(q["slot_kind"] == "pre_marker" for q in t_q)),
                "dlogp_oddq": float(np.mean(dlogp_q[1::2])),
                "dmargin_oddq": float(np.mean(dmargin_q[1::2])),
                "prior_half_a": prior_half_a(ctx),
            }
        )
        seen_cells.add((src, ctx))
    frame = pd.DataFrame(rows)
    assert len(frame), "no marker cells found — run the Phase 2 dispatcher first"
    # Coverage assert: every EXPECTED source (registered set by default — NOT
    # the observed frame, which would let a zero-file source vanish) must
    # cover the FULL selected (or descoped) panel — a shortfall names the
    # missing sources/cells instead of silently shrinking the registered
    # statistics' denominator.
    if expected_sources is None:
        expected_sources = REGISTERED_SOURCES
    sources_present = sorted(frame["source"].unique())
    unexpected = sorted(set(sources_present) - set(expected_sources))
    assert not unexpected, (
        f"marker frame contains sources OUTSIDE the expected set: {unexpected} "
        f"(expected {sorted(expected_sources)}) — stale per_cell files in {trained_dir}?"
    )
    absent_sources = sorted(set(expected_sources) - set(sources_present))
    shortfall = sorted({(s, c) for s in expected_sources for c in panel_set} - seen_cells)
    assert not shortfall, (
        f"marker frame coverage shortfall: expected {len(expected_sources)} sources x "
        f"{len(panel_set)} panel contexts, {len(shortfall)} cells absent; "
        f"sources with ZERO per_cell files: {absent_sources or 'none'} "
        f"(first 10 missing cells: {shortfall[:10]}); skipped-with-reason: {skipped[:10]}"
    )
    for c in sources_present[1:]:
        frame[f"fe_{c}"] = (frame["source"] == c).astype(float)
    return frame, sel


def _is_structural_drop(arm_sel: dict) -> bool:
    """Plan section 3 structural-alternative outcome for ONE fact arm:
    gate_pass=false AND no recorded descope AND ZERO bands pass the selection
    gate. In that state ``_descope_record`` returns None, so no descope can
    ever rescue the arm — 'similarity and behavior prior cannot be decoupled'
    for it, and it is structurally DROPPED from the analysis frame instead of
    blocking the whole fact battery (round-2 review note). Every OTHER
    absence (gate-fail with >=1 surviving band but no recorded descope,
    missing strata) falls through to ``_resolve_selected_panel``'s fail-loud
    assert."""
    if arm_sel.get("gate_pass", False):
        return False
    if (arm_sel.get("descope") or {}).get("active"):
        return False
    strata = arm_sel.get("strata") or {}
    return bool(strata) and not any(st["gate"]["verdict"] for st in strata.values())


def build_fact_frame(out_root: Path) -> tuple[pd.DataFrame, dict, list[str]]:
    """(cell x persona)-level fact frame from judged + tf JSONs + selection.

    Arms whose selection is a STRUCTURAL DROP (``_is_structural_drop``) are
    excluded from the frame and returned as the third element so the analysis
    JSON reports the plan-section-3 structural finding; their stale on-disk
    cells (e.g. Phase-0 smoke files) never enter the registered statistics.
    All other absences stay fail-loud: a gate-failed arm WITHOUT zero bands
    still trips ``_resolve_selected_panel``, a surviving arm with missing
    per-cell files trips the coverage shortfall assert, and a surviving arm
    absent from disk entirely trips the missing-arm assert."""
    sel = json.loads((out_root / "panel" / "fact_panel_selection.json").read_text())
    dropped = sorted(a for a, s in sel["per_arm"].items() if _is_structural_drop(s))
    for a in dropped:
        logger.warning(
            "[phase=p7_fact] arm %s STRUCTURALLY DROPPED (gate_pass=false, zero passing "
            "bands, no descope possible) — plan section 3 structural alternative; "
            "excluded from the fact frame",
            a,
        )
    judged_dir = out_root / "fact" / "judged"
    tf_dir = out_root / "fact" / "tf"
    allowed_by_arm = {
        a: _resolve_selected_panel(arm_sel, f"fact_panel_selection.json arm={a}")
        for a, arm_sel in sel["per_arm"].items()
        if a not in dropped
    }
    assert allowed_by_arm, (
        "ALL fact arms structurally dropped (zero passing bands everywhere) — no fact "
        "frame can be built; the family-wide structural finding IS the result "
        "(plan section 3 structural alternative)"
    )
    rows = []
    for f in sorted(judged_dir.glob("*.json")):
        j = json.loads(f.read_text())
        arm, seed, persona = j["arm"], j["seed"], j["persona"]
        if arm in dropped:
            continue  # structurally-dropped arm — stale smoke cells never enter
        arm_sel = sel["per_arm"][arm]
        if persona not in allowed_by_arm[arm] or persona not in arm_sel.get("per_persona", {}):
            continue  # stale / out-of-panel cell-persona — never enters registered statistics
        meta = arm_sel["per_persona"][persona]
        edges = arm_sel["band_edges_terciles"]
        cos = meta["cos_to_teacher"]
        band = "band_lo" if cos <= edges[0] else ("band_mid" if cos <= edges[1] else "band_hi")
        tf_path = tf_dir / f.name
        tf = json.loads(tf_path.read_text())["summary"] if tf_path.exists() else {}
        rows.append(
            {
                "source": f"{arm}_s{seed}",
                "arm": arm,
                "seed": seed,
                "context": persona,
                "band": band,
                "cos": cos,
                "prior": meta["prior_nat_per_tok"],
                "leak_rate": j["summary"]["leak_rate"],
                "tf_delta": tf.get("mean_delta_logprob_per_tok", float("nan")),
                "saturated": False,  # no softmax-ceiling construct for the fact TF delta
            }
        )
    frame = pd.DataFrame(rows)
    assert len(frame), "no fact cell-personas found — run the Phase 5 dispatcher first"
    missing_arms = sorted(set(allowed_by_arm) - set(frame["arm"].unique()))
    assert not missing_arms, (
        f"fact frame missing surviving (non-dropped) arms entirely: {missing_arms} — "
        "run / repair the Phase 5 dispatcher for these arms (only structurally-dropped "
        f"arms may be absent; dropped here: {dropped})"
    )
    shortfall = {}
    for (a, s), grp in frame.groupby(["arm", "seed"]):
        miss = sorted(allowed_by_arm[a] - set(grp["context"]))
        if miss:
            shortfall[f"{a}_s{s}"] = miss
    assert not shortfall, (
        f"fact frame coverage shortfall (cell -> missing panel personas): {shortfall}"
    )
    for c in sorted(frame["arm"].unique())[1:]:
        frame[f"fe_arm_{c}"] = (frame["arm"] == c).astype(float)
    for s in sorted(frame["seed"].unique())[1:]:
        frame[f"fe_seed_{s}"] = (frame["seed"] == s).astype(float)
    return frame, sel, dropped


# ---------------------------------------------------------------------------
# Registered-statistic battery for one (frame x DV x space)
# ---------------------------------------------------------------------------
def registered_battery(
    frame: pd.DataFrame,
    dv: str,
    *,
    sim_col: str,
    prior_col: str,
    group_col: str,
    n_boot: int,
    pinned: str,
) -> dict:
    """Both registered statistics + the pinned primary fit + robustness."""

    def stat_partial(f: pd.DataFrame) -> float:
        return pooled_partial(f, dv, prior_col, sim_col)

    def stat_dcv(f: pd.DataFrame) -> float:
        # Group CV folds by the ORIGINAL cluster id (group_col), NOT the
        # per-copy `_boot_cluster` id: all duplicate copies of a resampled
        # context must share a fold, or the +prior model memorizes the
        # duplicated cluster in train and scores it in test (CI ~2x too
        # narrow, shifted positive under a null — round-1 blocker
        # `cv-bootstrap-cluster-leakage` / `dcv-bootstrap-duplicate-fold-leakage`).
        return delta_cv_r2(f, dv, prior_col, sim_col, group_col)["delta_cv_r2_prior_beyond_sim"]

    point_partial = pooled_partial(frame, dv, prior_col, sim_col)
    boots_partial = cluster_bootstrap(frame, group_col, stat_partial, n_boot)
    dcv = delta_cv_r2(frame, dv, prior_col, sim_col, group_col)
    boots_dcv = cluster_bootstrap(frame, group_col, stat_dcv, n_boot)

    if pinned == "tobit":
        pin_point = tobit_prior_coef(frame, dv, prior_col, sim_col)
        pin_boots = cluster_bootstrap(
            frame, group_col, lambda f: tobit_prior_coef(f, dv, prior_col, sim_col), n_boot
        )
    elif pinned == "rank":
        pin_point = rank_fit_prior_coef(frame, dv, prior_col, sim_col)
        pin_boots = cluster_bootstrap(
            frame, group_col, lambda f: rank_fit_prior_coef(f, dv, prior_col, sim_col), n_boot
        )
    else:
        pin_point, pin_boots = float("nan"), np.array([])

    clean = frame[~frame["saturated"]]
    robustness = {
        "saturation_excluded_partial": (
            pooled_partial(clean, dv, prior_col, sim_col) if len(clean) >= 5 else float("nan")
        ),
        "n_saturated_excluded": int(frame["saturated"].sum()),
        "source_cluster_partial_ci": _boot_summary(
            cluster_bootstrap(frame, "source", stat_partial, n_boot), point_partial
        ),
    }
    return {
        "pooled_partial": _boot_summary(boots_partial, point_partial),
        "delta_cv_r2": {**dcv, **_boot_summary(boots_dcv, dcv["delta_cv_r2_prior_beyond_sim"])},
        "pinned_fit": {
            "kind": pinned,
            **_boot_summary(pin_boots, pin_point),
        },
        "per_band_descriptive": per_band_partials(
            frame, dv, prior_col, sim_col, max(n_boot // 4, 50)
        ),
        "robustness": robustness,
        "n_cells": len(frame),
        "n_contexts": int(frame["context"].nunique()),
    }


def _classify(batt: dict, holm_p: float | None = None) -> str:
    """wins+ / wins- / null / indet from one space's registered battery."""
    pp = batt["pooled_partial"]
    dcv = batt["delta_cv_r2"]
    pin = batt["pinned_fit"]
    p_partial = holm_p if holm_p is not None else pp["p_two_sided"]
    partial_sig = p_partial < 0.05 and not np.isnan(pp["point"])
    dcv_pos = dcv["ci95"][0] > 0
    dcv_null = dcv["ci95"][0] <= 0 <= dcv["ci95"][1]
    partial_null = pp["ci95"][0] <= 0 <= pp["ci95"][1]
    if dcv_pos and partial_sig:
        # Level DVs carry no pinned fit (pinned == "none" -> NaN): sign routes
        # on the pooled partial alone there; shift spaces require the pooled
        # partial AND the pinned primary fit to agree (plan section 3).
        pin_pt = pp["point"] if np.isnan(pin["point"]) else pin["point"]
        if pp["point"] > 0 and pin_pt > 0:
            return "wins+"
        if pp["point"] < 0 and pin_pt < 0:
            return "wins-"
        return "indet"
    if dcv_null and partial_null:
        return "null"
    return "indet"


def evaluate_lattice(
    level_cls: str, shift_logp_cls: str, shift_margin_cls: str, pooled_rho_upper: float
) -> dict:
    """The pre-registered section-3 lattice, sign-routed. Non-enumerated cells
    ship as indeterminate (no analyzer-discretion verdicts)."""
    key = (level_cls, shift_logp_cls, shift_margin_cls)
    if level_cls == "indet":
        verdict = "indeterminate: registered statistics disagree on H-level; no lattice row"
    elif level_cls == "null":
        verdict = (
            "prior signal not reproduced at matched geometry — check realized prior-range "
            "coverage vs the parent's high stratum (range restriction) BEFORE narrating "
            "cohort-confounded"
        )
    elif key == ("wins+", "null", "null"):
        if pooled_rho_upper < SURVIVES_RHO_UPPER:
            verdict = "prior = standing-level term; gate survives similarity-only (MDE stated)"
        else:
            verdict = "no detected overlap effect; underpowered for small terms (CI upper >= 0.2)"
    elif key == ("wins+", "wins+", "null"):
        verdict = "compression channel: prior enters through softmax measurement, not the gate"
    elif key == ("wins+", "wins+", "wins+"):
        verdict = "GATE NEEDS A BEHAVIOR-OVERLAP TERM — rank-1 model revision"
    elif level_cls == "wins+" and ("wins-" in (shift_logp_cls, shift_margin_cls)):
        verdict = (
            "headroom/coupling/shrinkage: negative prior->shift "
            "(base-containment) — NOT gate evidence"
        )
    elif key == ("wins+", "null", "wins+"):
        verdict = "margin-only overlap signal: suggestive mechanism-space evidence, cap MODERATE"
    else:
        verdict = "indeterminate: outcome cell not enumerated — report both statistics, no row"
    return {
        "h_level": level_cls,
        "h_shift_logprob": shift_logp_cls,
        "h_shift_eos_margin": shift_margin_cls,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Family analyses
# ---------------------------------------------------------------------------
PARENT_REGRESSION_TOL = 1e-6  # amendment plan v3 §2.3 free regression check


def _parent_regression_check(frame: pd.DataFrame, sel: dict, out_root: Path) -> dict:
    """FREE REGRESSION CHECK (amendment plan v3 §2.3): restricting the
    combined wide frame to the parent's inherited panel contexts must
    reproduce the parent's registered pooled partials (+0.263 log-prob /
    +0.550 EOS-margin recorded in the parent ``marker/analysis.json``) to
    numerical tolerance. Fail-loud with BOTH values printed — a mismatch
    means the parent cells drifted inside the combined frame (plan v3 §7
    'combined frame drifts from parent values')."""
    parent_panel = set(sel["panel_inherited"])
    parent_path = out_root / "marker" / "analysis.json"
    assert parent_path.exists(), (
        f"parent regression check needs the parent registered analysis at {parent_path}"
    )
    parent = json.loads(parent_path.read_text())
    sub = frame[frame["context"].isin(parent_panel)].reset_index(drop=True)
    out: dict = {
        "n_parent_cells": len(sub),
        "n_parent_contexts": int(sub["context"].nunique()),
        "tolerance": PARENT_REGRESSION_TOL,
        "spaces": {},
    }
    for space, dv in (("logprob", "dlogp"), ("eos_margin", "dmargin")):
        recomputed = pooled_partial(sub, dv, "prior", "cos")
        parent_pt = parent["registered"]["shift"][space]["pooled_partial"]["point"]
        diff = abs(recomputed - parent_pt)
        logger.info(
            "[phase=p7_regression] %s: recomputed=%.12f parent=%.12f |diff|=%.3e",
            space,
            recomputed,
            parent_pt,
            diff,
        )
        assert diff < PARENT_REGRESSION_TOL, (
            f"PARENT REGRESSION CHECK FAIL ({space}): combined frame restricted to the "
            f"parent {len(parent_panel)}-context panel gives pooled partial {recomputed!r} "
            f"vs parent registered {parent_pt!r} (|diff|={diff:.3e} >= "
            f"{PARENT_REGRESSION_TOL}) — parent cells drifted inside the combined frame "
            "(amendment plan v3 §2.3 / §7); do NOT proceed to the wide registered statistics"
        )
        out["spaces"][space] = {
            "recomputed": float(recomputed),
            "parent_registered": float(parent_pt),
            "abs_diff": float(diff),
        }
    return out


def analyze_marker(
    out_root: Path,
    figures_dir: Path,
    n_boot: int,
    selection_path: Path | None = None,
    out_dir: Path | None = None,
    expected_sources: list[str] | None = None,
) -> dict:
    frame, sel = build_marker_frame(out_root, selection_path, expected_sources=expected_sources)
    logger.info(
        "[phase=p7_marker] frame: %d cells, %d contexts", len(frame), frame["context"].nunique()
    )
    regression_check: dict | None = None
    if "panel_inherited" in sel:
        regression_check = _parent_regression_check(frame, sel, out_root)

    res: dict = {"schema_version": "issue605_v1", "family": "marker"}
    common = dict(sim_col="cos", prior_col="prior", group_col="context", n_boot=n_boot)
    # Shift DVs (the gate question proper).
    shift = {
        "logprob": registered_battery(frame, "dlogp", pinned="tobit", **common),
        "eos_margin": registered_battery(frame, "dmargin", pinned="rank", **common),
    }
    holm = _holm({k: v["pooled_partial"]["p_two_sided"] for k, v in shift.items()})
    for k in shift:
        shift[k]["pooled_partial"]["p_holm_over_shift_spaces"] = holm[k]
    # Level DVs.
    level = {
        "emission_rate": registered_battery(frame, "emission_rate", pinned="none", **common),
        "trained_logp": registered_battery(frame, "trained_logp", pinned="none", **common),
    }
    res["registered"] = {"shift": shift, "level": level}

    # H-level lattice ROUTING classifies on `trained_logp` alone (the
    # continuous absolute level DV). Plan section 3 names both level DVs
    # (emission rate / absolute trained log P); the emission-rate battery is
    # computed + reported above but NOT routed — it is zero-inflated at low
    # prior and ceiling-prone at high prior, so routing on it would key the
    # lattice on a saturating readout. Recorded in the analysis JSON below.
    level_cls = _classify(level["trained_logp"])
    # H-level requires POSITIVE sign explicitly.
    if level_cls == "wins-":
        level_cls = "indet"
    shift_logp_cls = _classify(
        shift["logprob"], holm_p=shift["logprob"]["pooled_partial"]["p_holm_over_shift_spaces"]
    )
    shift_margin_cls = _classify(
        shift["eos_margin"],
        holm_p=shift["eos_margin"]["pooled_partial"]["p_holm_over_shift_spaces"],
    )
    rho_upper = max(
        shift["logprob"]["pooled_partial"]["ci95"][1],
        shift["eos_margin"]["pooled_partial"]["ci95"][1],
    )
    res["lattice"] = evaluate_lattice(level_cls, shift_logp_cls, shift_margin_cls, rho_upper)
    res["lattice"]["h_level_routing"] = {
        "dv": "trained_logp",
        "note": "H-level routed on the continuous absolute trained log P; the emission-rate "
        "battery is computed + reported under registered.level but not routed "
        "(zero-inflated / ceiling-prone)",
    }

    # Diagnostics (plan 11.5).
    coll_overall = abs(_pearson(frame["cos"].to_numpy(), frame["prior"].to_numpy()))
    coll_by_band = {
        b: abs(
            _pearson(
                frame[frame["band"] == b]["cos"].to_numpy(),
                frame[frame["band"] == b]["prior"].to_numpy(),
            )
        )
        for b in sorted(frame["band"].unique())
    }
    coll_fired = coll_overall > COLLINEARITY_GATE or any(
        v > COLLINEARITY_GATE for v in coll_by_band.values() if not np.isnan(v)
    )
    tercile_fallback = {}
    if coll_fired:
        pri_terc = pd.qcut(frame["prior"], 3, labels=["pT1", "pT2", "pT3"])
        tercile_fallback = {
            str(t): float(frame.loc[pri_terc == t, "dlogp"].median()) for t in ("pT1", "pT2", "pT3")
        }
    prompt_lens = frame["context"].map(_marker_prompt_token_len_map(frame))
    frame["prompt_len"] = prompt_lens
    src_mean_d = frame.groupby("source")["dlogp"].transform("mean").to_numpy()
    diagnostics = {
        "corr_prior_vs_base_term_at_trained_slot": float(
            spearmanr(frame["prior"], frame["base_logp_matched"]).statistic
        ),
        "base_covariate_partial_dlogp": partial_spearman(
            frame["dlogp"].to_numpy(),
            frame["prior"].to_numpy(),
            covars=[frame["cos"].to_numpy(), src_mean_d, frame["base_logp_matched"].to_numpy()],
            dummies=_band_dummies(frame["band"]),
        ),
        "split_probe_complement_partial": _split_probe_complement(frame),
        "prompt_length_covariate_partial": partial_spearman(
            frame["dlogp"].to_numpy(),
            frame["prior"].to_numpy(),
            covars=[frame["cos"].to_numpy(), src_mean_d, frame["prompt_len"].to_numpy()],
            dummies=_band_dummies(frame["band"]),
        ),
        "cov_negsim_partial": partial_spearman(
            frame["dlogp"].to_numpy(),
            frame["prior"].to_numpy(),
            covars=[frame["cos"].to_numpy(), src_mean_d, frame["negsim"].to_numpy()],
            dummies=_band_dummies(frame["band"]),
        ),
        "affordance_class_stratified": {
            cls: pooled_partial(frame[frame["affordance_class"] == cls], "dlogp", "prior", "cos")
            for cls in sorted(frame["affordance_class"].unique())
            if (frame["affordance_class"] == cls).sum() >= 20
        },
        "centered_bank_cosine_sensitivity": {
            "pooled_partial_shift": pooled_partial(frame, "dlogp", "prior", "cos_centered"),
            "delta_cv_r2": delta_cv_r2(frame, "dlogp", "prior", "cos_centered", "context"),
            "label": "centered-bank cosine (#536) — sensitivity only, never mixed with raw",
        },
        "collinearity_gate": {
            "abs_pearson_overall": coll_overall,
            "by_band": coll_by_band,
            "fired": bool(coll_fired),
            "tercile_median_dlogp_fallback": tercile_fallback,
        },
        "saturation_map": _saturation_map(frame),
        "gkl_secondary_partial_shift": pooled_partial(frame, "dlogp", "prior", "gkl"),
        "space_agreement_rho_dlogp_dmargin": float(
            spearmanr(frame["dlogp"], frame["dmargin"]).statistic
        ),
        "legacy_anchor_drift": _legacy_drift(out_root),
    }
    res["diagnostics"] = diagnostics
    res["panel_gate"] = {"gate_pass": sel["gate_pass"], "strata": sel["strata"]}
    if regression_check is not None:
        res["parent_regression_check"] = regression_check
        res["amendment_label"] = sel.get("amendment_label")
        res["panel_inherited_n"] = len(sel["panel_inherited"])
        res["panel_new_n"] = len(sel.get("panel_new", []))
    res["metadata"] = _repro_meta({"n_boot": n_boot, "analysis_seed": ANALYSIS_SEED})

    _marker_figures(frame, res, figures_dir)
    out = (out_dir or (out_root / "marker")) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    logger.info("[phase=p7_marker] written %s — lattice: %s", out, res["lattice"]["verdict"])
    return res


def _split_probe_complement(frame: pd.DataFrame) -> dict:
    """Split-probe diagnostic (plan 11.5): prior from probe half A (even-index
    per-q P1 prior reads, NOT the full-probe scalar) vs the half-B (odd-index)
    DV — removes shared-probe-sampling noise coupling. Legacy contexts whose
    prior is a reused #532 scalar carry no per-q rows and are excluded."""
    m = frame["prior_half_a"].notna() & frame["dlogp_oddq"].notna()
    n = int(m.sum())
    if n < 8:
        rho = float("nan")
    else:
        sub = frame[m]
        sub_src_mean = sub.groupby("source")["dlogp_oddq"].transform("mean").to_numpy()
        rho = partial_spearman(
            sub["dlogp_oddq"].to_numpy(),
            sub["prior_half_a"].to_numpy(),
            covars=[sub["cos"].to_numpy(), sub_src_mean],
            dummies=_band_dummies(sub["band"]),
        )
    return {
        "rho": rho,
        "n_cells": n,
        "prior_basis": "probe half A (even-index per-q P1 prior reads); DV = half-B "
        "(odd-index) dlogp; legacy scalar-prior contexts excluded",
    }


def _marker_prompt_token_len_map(frame: pd.DataFrame) -> dict[str, int]:
    """Rendered system-prompt token length per context (prompt-length covariate)."""
    from issue532_predictor_stress import _instructed_bystander_panel
    from issue605_contexts import marker_candidates, marker_expansion_candidates
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    cands = marker_candidates()
    cands.update(marker_expansion_candidates())
    prompts = {lb: c["system_prompt"] for lb, c in cands.items()}
    prompts.update(_instructed_bystander_panel())
    out = {}
    for ctx in frame["context"].unique():
        text = prompts.get(ctx, ctx)  # condition cids fall back to label length
        out[ctx] = len(tok.encode(text, add_special_tokens=False))
    return out


def _saturation_map(frame: pd.DataFrame) -> dict:
    pri_terc = pd.qcut(frame["prior"], 3, labels=["pT1", "pT2", "pT3"], duplicates="drop")
    out = {}
    for t in pri_terc.cat.categories:
        sub = frame[pri_terc == t]
        out[str(t)] = {
            "n_cells": len(sub),
            "frac_saturated": float(sub["saturated"].mean()),
            "frac_cells_with_emitting_slots": float((sub["n_emitting_slots"] > 0).mean()),
            "non_emitting_slice_coverage": float((sub["n_emitting_slots"] == 0).mean()),
        }
    return out


def _legacy_drift(out_root: Path) -> dict:
    """Fresh-measured vs #532-recorded pair-similarity drift (QA only)."""
    table = json.loads((out_root / "panel" / "marker_pair_table.json").read_text())
    fresh_c, leg_c, fresh_g, leg_g = [], [], [], []
    for r in table["rows"]:
        if "legacy_cos_532" in r:
            fresh_c.append(r["cos_l21"])
            leg_c.append(r["legacy_cos_532"])
            fresh_g.append(r["gkl_l22"])
            leg_g.append(r["legacy_gkl_532"])
    if not fresh_c:
        return {"n_pairs": 0}
    return {
        "n_pairs": len(fresh_c),
        "cos_rank_corr": float(spearmanr(fresh_c, leg_c).statistic),
        "gkl_rank_corr": float(spearmanr(fresh_g, leg_g).statistic),
        "cos_max_abs_diff": float(np.max(np.abs(np.array(fresh_c) - np.array(leg_c)))),
    }


def analyze_fact(
    out_root: Path, figures_dir: Path, n_boot: int, out_dir: Path | None = None
) -> dict:
    frame, sel, dropped = build_fact_frame(out_root)
    logger.info(
        "[phase=p7_fact] frame: %d cell-personas, %d personas, %d arms (structurally dropped: %s)",
        len(frame),
        frame["context"].nunique(),
        frame["arm"].nunique(),
        dropped or "none",
    )
    res: dict = {"schema_version": "issue605_v1", "family": "fact"}
    res["arms_dropped_structural"] = dropped
    if dropped:
        res["structural_finding"] = (
            "arm(s) " + ", ".join(dropped) + ": similarity and behavior prior cannot be "
            "decoupled under this candidate budget — zero bands passed the Phase-4.5 "
            "gate after the pre-registered expansion round, so no descope exists and "
            "the arm is structurally dropped (plan section 3 structural alternative). "
            "All per-arm reads and the family lattice below cover surviving arms only."
        )
    common = dict(sim_col="cos", prior_col="prior", group_col="context", n_boot=n_boot)
    shift = {"tf_delta": registered_battery(frame, "tf_delta", pinned="rank", **common)}
    level = {"leak_rate": registered_battery(frame, "leak_rate", pinned="none", **common)}
    res["registered"] = {"shift": shift, "level": level}

    # Proxy validation (pre-registered): TF delta must track judged leak.
    m = ~(frame["tf_delta"].isna() | frame["leak_rate"].isna())
    proxy_rho = (
        float(spearmanr(frame.loc[m, "tf_delta"], frame.loc[m, "leak_rate"]).statistic)
        if m.sum() >= 5
        else float("nan")
    )
    proxy_valid = bool(proxy_rho > PROXY_RHO_MIN)
    res["tf_proxy_validation"] = {
        "rho_tf_delta_vs_leak": proxy_rho,
        "threshold": PROXY_RHO_MIN,
        "valid": proxy_valid,
        "n_cell_personas": int(m.sum()),
    }

    level_cls = _classify(level["leak_rate"])
    if level_cls == "wins-":
        level_cls = "indet"
    if proxy_valid:
        shift_cls = _classify(shift["tf_delta"])
        # Single shift space for facts: the lattice's two-space conjunction
        # degenerates to the one validated space (reported as such).
        res["lattice"] = evaluate_lattice(
            level_cls,
            shift_cls,
            shift_cls,
            shift["tf_delta"]["pooled_partial"]["ci95"][1],
        )
        res["lattice"]["note"] = (
            "fact family has ONE shift space (TF delta, proxy-validated); the two-space "
            "conjunction degenerates to it"
        )
    else:
        res["lattice"] = {
            "h_level": level_cls,
            "verdict": "fact H-shift read PROXY-INVALID (rho <= 0.4 vs judged leak) — "
            "level DV stands alone",
        }

    src_mean = frame.groupby("source")["tf_delta"].transform("mean").to_numpy()
    res["diagnostics"] = {
        "cov_negsim_partial": _fact_negsim_partial(out_root, frame, src_mean),
        "per_arm_pooled_partial_shift": {
            a: pooled_partial(frame[frame["arm"] == a], "tf_delta", "prior", "cos")
            for a in sorted(frame["arm"].unique())
        },
        "collinearity_gate": {
            "abs_pearson_overall": abs(
                _pearson(frame["cos"].to_numpy(), frame["prior"].to_numpy())
            ),
        },
    }
    # Per-arm gate record over ALL registered arms (incl. dropped) — the
    # frame / verdicts above cover surviving arms only.
    res["panel_gate"] = {
        a: {
            "gate_pass": s["gate_pass"],
            "descope_active": bool((s.get("descope") or {}).get("active")),
            "surviving_bands": (s.get("descope") or {}).get("surviving_bands"),
            "structurally_dropped": a in dropped,
        }
        for a, s in sel["per_arm"].items()
    }
    res["metadata"] = _repro_meta({"n_boot": n_boot, "analysis_seed": ANALYSIS_SEED})

    _fact_figures(frame, res, figures_dir)
    out = (out_dir or (out_root / "fact")) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    logger.info("[phase=p7_fact] written %s — lattice: %s", out, res["lattice"]["verdict"])
    return res


def _fact_negsim_partial(out_root: Path, frame: pd.DataFrame, src_mean: np.ndarray) -> float:
    """cov_negsim for the fact family: similarity to the nearest #541
    training negative, from the Phase-4 activation cache (free)."""
    acts_dir = out_root / "panel" / "fact_measure" / "acts"
    negatives = ("assistant", "software_engineer", "kindergarten_teacher", "no_system")
    if not all((acts_dir / f"{n}.npz").exists() for n in negatives):
        return float("nan")
    from issue532_predictor_stress import _cosine_predictor

    neg_acts = {}
    for n in negatives:
        with np.load(acts_dir / f"{n}.npz") as z:
            neg_acts[n] = z["21"]
    negsim = []
    for p in frame["context"]:
        path = acts_dir / f"{p}.npz"
        if not path.exists():
            negsim.append(np.nan)
            continue
        with np.load(path) as z:
            a = z["21"]
        negsim.append(max(_cosine_predictor(a, neg_acts[n]) for n in negatives))
    negsim = np.array(negsim, dtype=float)
    if np.isnan(negsim).all():
        return float("nan")
    return partial_spearman(
        frame["tf_delta"].to_numpy(),
        frame["prior"].to_numpy(),
        covars=[frame["cos"].to_numpy(), src_mean, np.nan_to_num(negsim, nan=np.nanmean(negsim))],
        dummies=_band_dummies(frame["band"]),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _set_style():
    import matplotlib

    matplotlib.use("Agg")
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()


#: Reader-facing names for the matched-similarity bands (round-2 critique:
#: figures must not carry band_lo/mid/hi slugs).
BAND_LABELS = {
    "band_lo": "far similarity band",
    "band_mid": "mid similarity band",
    "band_hi": "close similarity band",
}

#: Reader-facing names for the registered (DV-class, space) ladder entries.
LADDER_LABELS = {
    ("shift", "logprob"): "log-prob shift",
    ("shift", "eos_margin"): "EOS-margin logit shift",
    ("level", "emission_rate"): "emission rate (level)",
    ("level", "trained_logp"): "trained log-prob (level)",
}


def _marker_figures(frame: pd.DataFrame, res: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    _set_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bands = ["band_lo", "band_mid", "band_hi"]

    # HERO 1 — within-band shift-vs-prior scatter, log-prob + EOS-margin rows.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for row, dv in enumerate(("dlogp", "dmargin")):
        for col, band in enumerate(bands):
            ax = axes[row][col]
            sub = frame[frame["band"] == band]
            sat = sub["saturated"]
            ax.scatter(sub.loc[~sat, "prior"], sub.loc[~sat, dv], s=10, alpha=0.5, color="#1f77b4")
            ax.scatter(
                sub.loc[sat, "prior"],
                sub.loc[sat, dv],
                s=10,
                alpha=0.5,
                color="#d62728",
                marker="x",
            )
            if row == 0:
                ax.set_title(f"{BAND_LABELS[band]} ({len(sub)} cells)")
            if col == 0:
                ax.set_ylabel(
                    "log-prob shift, trained - base (nats)"
                    if dv == "dlogp"
                    else "marker-vs-EOS logit-margin shift"
                )
            if row == 1:
                ax.set_xlabel("base log P(marker) (graded prior)")
    savefig_paper(fig, "hero_shift_vs_prior_by_band", dir=fig_dir)
    plt.close(fig)

    # HERO 2 — delta-CV-R^2 ladder bars with bootstrap CIs.
    fig, ax = plt.subplots(figsize=(8, 5))
    entries = []
    for cls_name, group in res["registered"].items():
        for space, batt in group.items():
            d = batt["delta_cv_r2"]
            label = LADDER_LABELS.get((cls_name, space), f"{cls_name}/{space}")
            entries.append((label, d["point"], d["ci95"]))
    x = np.arange(len(entries))
    pts = [e[1] for e in entries]
    los = [max(0.0, e[1] - e[2][0]) for e in entries]
    his = [max(0.0, e[2][1] - e[1]) for e in entries]
    ax.bar(x, pts, color="#1f77b4")
    ax.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x, [e[0] for e in entries], rotation=20, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("held-out gain in out-of-fold R² from adding the prior")
    savefig_paper(fig, "hero_delta_cv_r2_ladder", dir=fig_dir)
    plt.close(fig)

    # Panel-construction QA: realized similarity x prior grid with band edges.
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, label, color in (
        ("near_twin", "near-twin paraphrase", "#1f77b4"),
        ("related", "related profession", "#2ca02c"),
        ("unrelated", "unrelated profession", "#7f7f7f"),
        ("symbol_flavored", "symbol-flavored persona", "#9467bd"),
        ("legacy", "legacy instructed", "#d62728"),
    ):
        sub = frame[frame["content_class"] == cls]
        if len(sub):
            ax.scatter(sub["cos"], sub["prior"], s=10, alpha=0.5, color=color, label=label)
    for e in res.get("panel_gate", {}).get("strata", {}).values():
        ax.axvspan(e["window"][0], e["window"][1], alpha=0.08, color="orange")
    ax.set_xlabel("cosine similarity to source (layer-21 last-prompt-token)")
    ax.set_ylabel("base log P(marker) at response end (graded prior, nats)")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "panel_grid_similarity_x_prior", dir=fig_dir)
    plt.close(fig)

    # Level-DV hero + per-space leaderboard + raw-vs-residualized.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(frame["prior"], frame["emission_rate"], s=10, alpha=0.4)
    axes[0].set_xlabel("base log P(marker) (graded prior)")
    axes[0].set_ylabel("on-policy emission rate (level)")
    axes[1].scatter(frame["prior"], frame["trained_logp"], s=10, alpha=0.4, color="#2ca02c")
    axes[1].set_xlabel("base log P(marker) (graded prior)")
    axes[1].set_ylabel("trained log P(marker) at response end (level)")
    savefig_paper(fig, "level_dvs_vs_prior", dir=fig_dir)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ry = rankdata(frame["dlogp"])
    rx = rankdata(frame["prior"])
    src_mean = frame.groupby("source")["dlogp"].transform("mean").to_numpy()
    C = _design([frame["cos"].to_numpy(), src_mean], _band_dummies(frame["band"]), len(frame))
    res_y = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    res_x = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ax.scatter(res_x, res_y, s=10, alpha=0.4)
    ax.set_xlabel("prior rank, residualized on similarity + band + source level")
    ax.set_ylabel("log-prob shift rank, residualized the same way")
    savefig_paper(fig, "residualized_shift_vs_prior", dir=fig_dir)
    plt.close(fig)


def _fact_figures(frame: pd.DataFrame, res: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper

    _set_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    arm_labels = {
        "courthouse_architecture_historian": "courthouse-historian teacher",
        "top_prior_wooden_furniture_carpenter": "furniture-carpenter teacher",
        "wooden_furniture_carpenter": "furniture-carpenter teacher",
        "marine_biologist": "marine-biologist teacher",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arm, marker in zip(sorted(frame["arm"].unique()), "o^s", strict=False):
        sub = frame[frame["arm"] == arm]
        label = arm_labels.get(arm, arm)
        axes[0].scatter(sub["prior"], sub["tf_delta"], s=12, alpha=0.5, marker=marker, label=label)
        axes[1].scatter(sub["prior"], sub["leak_rate"], s=12, alpha=0.5, marker=marker, label=label)
    axes[0].set_xlabel("persona's base fact prior (nats per token)")
    axes[0].set_ylabel("teacher-forced fact shift (nats/token)")
    axes[1].set_xlabel("persona's base fact prior (nats per token)")
    axes[1].set_ylabel("judged leak rate (fraction stating the taught fact)")
    axes[0].legend(frameon=False, fontsize=7)
    savefig_paper(fig, "fact_shift_and_level_vs_prior", dir=fig_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic stub (CPU end-to-end smoke; schema-true, planted effect)
# ---------------------------------------------------------------------------
def synthesize_stub(root: Path, fact_variant: str = "clean") -> None:
    """Write a tiny artifact tree matching the production schemas with a
    PLANTED positive prior->shift effect (smoke assertion target).

    ``fact_variant``:
      - ``clean`` (default): all three fact arms gate_pass=true (the original
        stub shape).
      - ``descoped``: the post-Phase-4.5 production shape — every arm
        gate_pass=false; two arms carry recorded descopes to disjoint band
        subsets; one arm (marine_biologist) has ZERO passing bands and no
        descope (the structural-drop path), plus stale smoke cells on disk
        that the frame builder must skip."""
    assert fact_variant in ("clean", "descoped"), fact_variant
    rng = np.random.default_rng(7)
    sources = list(STUB_SOURCES)  # single source of truth with main()'s coverage override
    contexts = [
        f"m605_stub_{i:02d}__{a}"
        for i, a in enumerate(["none"] * 4 + ["soft"] * 4 + ["explicit"] * 4)
    ]
    cos_vals = np.linspace(0.62, 0.95, len(contexts))
    prior_vals = np.tile([-20.0, -16.0, -12.0, -8.0], 3) + rng.normal(0, 0.5, len(contexts))
    n_probes = 6

    pair_rows = []
    for s_i, s in enumerate(sources):
        for c_i, c in enumerate(contexts):
            pair_rows.append(
                {
                    "source_cid": s,
                    "context_label": c,
                    "cos_l21": float(np.clip(cos_vals[c_i] + 0.01 * s_i, 0, 0.999)),
                    "gkl_l22": float(10 * (1 - cos_vals[c_i]) + rng.normal(0, 0.2)),
                    "cos_centered_bank": float(cos_vals[c_i] - 0.5),
                    "prior_logp": float(prior_vals[c_i]),
                    "prior_source": "measured_605",
                    "sim_to_nearest_negative": float(cos_vals[c_i] * 0.9),
                    "content_class": "near_twin" if c_i < 6 else "unrelated",
                    "affordance_class": c.rsplit("__", 1)[1],
                    "is_legacy": False,
                }
            )
    panel_dir = root / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    (panel_dir / "marker_pair_table.json").write_text(
        json.dumps(
            {"schema_version": "issue605_v1", "n_contexts": len(contexts), "rows": pair_rows},
            indent=1,
        )
    )
    cos_all = np.array([r["cos_l21"] for r in pair_rows])
    edges = np.quantile(cos_all, [1 / 3, 2 / 3])
    (panel_dir / "marker_panel_selection.json").write_text(
        json.dumps(
            {
                "schema_version": "issue605_v1",
                "band_edges_terciles": [float(edges[0]), float(edges[1])],
                "panel": contexts,
                "panel_size": len(contexts),
                "gate_pass": True,
                "strata": {
                    "band_lo": {"window": [0.62, 0.68], "gate": {"verdict": True}},
                    "band_mid": {"window": [0.74, 0.80], "gate": {"verdict": True}},
                    "band_hi": {"window": [0.88, 0.94], "gate": {"verdict": True}},
                },
            },
            indent=1,
        )
    )

    # Per-q P1 prior reads (half-A prior basis for the split-probe diagnostic).
    prior_dir = root / "panel" / "marker_measure" / "prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    for c_i, c in enumerate(contexts):
        per_q = [
            {"logp_marker": float(prior_vals[c_i] + rng.normal(0, 0.3))} for _ in range(n_probes)
        ]
        (prior_dir / f"{c}.json").write_text(
            json.dumps(
                {
                    "schema_version": "issue605_v1",
                    "phase": "p1_marker_prior",
                    "context_label": c,
                    "n_probes": n_probes,
                    "per_q": per_q,
                    "summary": {
                        "mean_logp_marker": float(np.mean([r["logp_marker"] for r in per_q]))
                    },
                },
                indent=1,
            )
        )

    def slot_read(logp: float, margin: float) -> dict:
        z_eos = 10.0
        return {
            "logp_marker": float(logp),
            "z_marker": float(z_eos + margin),
            "z_eos": z_eos,
            "logZ": float(z_eos + margin - logp),
            "logp_bare_marker": float(logp - 2),
            "argmax_id": 83399 if logp > -0.05 else 151645,
            "slot_kind": "end_of_response",
            "emitted_id": None,
            "n_truncated_tokens": 0,
        }

    for d in ("per_cell_trained", "per_cell_base", "gen"):
        (root / "marker" / d).mkdir(parents=True, exist_ok=True)
    for s_i, s in enumerate(sources):
        for c_i, c in enumerate(contexts):
            pr = pair_rows[s_i * len(contexts) + c_i]
            base_logp = pr["prior_logp"] + rng.normal(0, 0.3)
            # PLANTED effect: shift rises with prior AND cos.
            shift = 3.0 + 0.3 * (pr["prior_logp"] + 14) + 4 * (pr["cos_l21"] - 0.75)
            shift += 0.5 * s_i + rng.normal(0, 0.4)
            t_q = [
                slot_read(
                    min(base_logp + shift + rng.normal(0, 0.3), -0.01), shift + rng.normal(0, 0.5)
                )
                for _ in range(n_probes)
            ]
            b_q = [
                slot_read(base_logp + rng.normal(0, 0.3), rng.normal(-8, 0.5))
                for _ in range(n_probes)
            ]
            for d, per_q, phase in (
                ("per_cell_trained", t_q, "p2_trained_on_own_R"),
                ("per_cell_base", b_q, "p2_base_on_trained_R"),
            ):
                (root / "marker" / d / f"{s}__{c}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "issue532_followup_logp_v1",
                            "phase": phase,
                            "source_cid": s,
                            "context_label": c,
                            "n_probes": n_probes,
                            "per_q": per_q,
                        },
                        indent=1,
                    )
                )
            emit = float(np.clip((np.mean([q["logp_marker"] for q in t_q]) + 8) / 8, 0, 1))
            (root / "marker" / "gen" / f"{s}__{c}.json").write_text(
                json.dumps(
                    {"summary": {"in_R_emission_rate": emit, "in_R_emit_at_end_rate": emit * 0.9}},
                    indent=1,
                )
            )

    _synthesize_fact_stub(root, panel_dir, rng, fact_variant)
    logger.info("synthetic stub written under %s (fact_variant=%s)", root, fact_variant)


def _synthesize_fact_stub(
    root: Path, panel_dir: Path, rng: np.random.Generator, fact_variant: str
) -> None:
    """Fact half of the stub: 3 arms x 1 seed x 8 personas. In the
    ``descoped`` variant the band membership / surviving-band sets below
    DERIVE the descoped panels and the zero-band structural drop (nothing
    hard-coded downstream)."""
    arms = ["marine_biologist", "courthouse_architecture_historian", "wooden_furniture_carpenter"]
    personas = [f"f605_stub_{i}" for i in range(8)]
    surviving_by_arm = {
        "marine_biologist": [],  # zero passing bands -> structural drop
        "courthouse_architecture_historian": ["band_mid", "band_hi"],
        "wooden_furniture_carpenter": ["band_lo"],
    }
    per_arm = {}
    write_personas: dict[str, list[str]] = {}
    for a in arms:
        cosv = np.linspace(0.55, 0.9, len(personas))
        priv = np.linspace(-3.5, -2.6, len(personas))[np.argsort(rng.random(len(personas)))]
        e = np.quantile(cosv, [1 / 3, 2 / 3])

        def band_of(c: float, _e=e) -> str:
            return "band_lo" if c <= _e[0] else ("band_mid" if c <= _e[1] else "band_hi")

        per_arm[a] = {
            "band_edges_terciles": [float(e[0]), float(e[1])],
            "panel": personas,
            "gate_pass": True,
            "per_persona": {
                p: {"prior_nat_per_tok": float(priv[i]), "cos_to_teacher": float(cosv[i])}
                for i, p in enumerate(personas)
            },
        }
        if fact_variant == "clean":
            write_personas[a] = personas
            continue
        surv = surviving_by_arm[a]
        per_arm[a]["gate_pass"] = False
        per_arm[a]["strata"] = {
            b: {"gate": {"verdict": b in surv}} for b in ("band_lo", "band_mid", "band_hi")
        }
        descoped = [p for i, p in enumerate(personas) if band_of(cosv[i]) in surv]
        if surv:
            per_arm[a]["descope"] = {
                "active": True,
                "surviving_bands": surv,
                "panel_descoped": descoped,
                "note": "stub descope (fact_variant=descoped)",
            }
            write_personas[a] = descoped
        else:
            # Structurally-dropped arm: NO descope key; leave two stale
            # smoke cells on disk that build_fact_frame must skip.
            write_personas[a] = personas[:2]
    (panel_dir / "fact_panel_selection.json").write_text(
        json.dumps({"schema_version": "issue605_v1", "per_arm": per_arm}, indent=1)
    )
    for d in ("judged", "tf"):
        (root / "fact" / d).mkdir(parents=True, exist_ok=True)
    for a in arms:
        for p in write_personas[a]:
            meta = per_arm[a]["per_persona"][p]
            delta = 0.15 + 0.3 * (meta["prior_nat_per_tok"] + 3.0) + rng.normal(0, 0.03)
            leak = float(np.clip(0.3 + 1.5 * delta + rng.normal(0, 0.05), 0, 1))
            tag = f"arm_{a}_seed42__{p}"
            (root / "fact" / "judged" / f"{tag}.json").write_text(
                json.dumps(
                    {
                        "arm": a,
                        "seed": 42,
                        "persona": p,
                        "summary": {
                            "n_rows": 10,
                            "stated_seven": int(leak * 10),
                            "leak_rate": leak,
                            "judge_failed_rows": 0,
                        },
                    },
                    indent=1,
                )
            )
            (root / "fact" / "tf" / f"{tag}.json").write_text(
                json.dumps(
                    {
                        "arm": a,
                        "seed": 42,
                        "persona": p,
                        "summary": {
                            "mean_delta_logprob_per_tok": float(delta),
                            "frac_rows_positive_delta": 0.9,
                            "n_scored": 12,
                        },
                    },
                    indent=1,
                )
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Issue #605 Phase 7 analysis (registered statistics + lattice + figures).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--families", default="marker,fact")
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_605"))
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="figure output dir. Default depends on the run shape: parent run -> "
        "figures/issue_605; amendment run (--out / --marker-selection) -> a dedicated "
        "subdir (figures/issue_605/<out-name>), NEVER the parent dir (the fixed figure "
        "filenames would overwrite the parent's committed hero figures); stub run -> "
        "<stub-root>/figures",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument(
        "--marker-selection",
        type=Path,
        default=None,
        help="marker selection JSON override (the amendment passes the WIDE "
        "marker_panel_selection_wide.json; default = <out-root>/panel/"
        "marker_panel_selection.json). When the override carries panel_inherited, the "
        "parent regression check (plan v3 §2.3) runs before the registered statistics.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="analysis.json output dir override (default <out-root>/<family>/). "
        "Single-family runs only — the amendment passes the followup-label dir.",
    )
    ap.add_argument(
        "--synthesize-stub",
        type=Path,
        default=None,
        help="write a synthetic artifact tree here, then analyze it",
    )
    ap.add_argument(
        "--stub-variant",
        choices=("clean", "descoped"),
        default="clean",
        help="fact-family stub shape: 'clean' = all arms gate_pass=true; 'descoped' = "
        "two descoped arms + one zero-band structurally-dropped arm (CPU exercise of "
        "the post-Phase-4.5 production shape); only used with --synthesize-stub",
    )
    args = ap.parse_args()

    if args.synthesize_stub is not None:
        synthesize_stub(args.synthesize_stub, fact_variant=args.stub_variant)
        args.out_root = args.synthesize_stub

    families = args.families.split(",")
    if args.out is not None and len(families) > 1:
        raise SystemExit(
            "--out names ONE analysis output dir; run a single --families with it "
            "(both families would clobber the same analysis.json)"
        )

    # Figures-dir resolution (concern wide-analysis-figures-dir-clobber):
    # _marker_figures/_fact_figures write FIXED filenames, so amendment and
    # stub runs must never default into the parent figures dir.
    if args.figures_dir is None:
        if args.synthesize_stub is not None:
            args.figures_dir = args.synthesize_stub / "figures"
        elif args.out is not None:
            args.figures_dir = PARENT_FIGURES_DIR / args.out.name
        elif args.marker_selection is not None:
            args.figures_dir = PARENT_FIGURES_DIR / args.marker_selection.stem
        else:
            args.figures_dir = PARENT_FIGURES_DIR
    amendment_run = args.marker_selection is not None or args.out is not None
    if amendment_run:
        assert args.figures_dir.resolve() != PARENT_FIGURES_DIR.resolve(), (
            "amendment analysis (--marker-selection/--out) refuses to write figures into "
            f"the parent default {PARENT_FIGURES_DIR} — the fixed figure filenames would "
            "overwrite the parent's committed hero figures; pass a dedicated --figures-dir"
        )
    logger.info("figures dir resolved to %s", args.figures_dir)

    # Coverage expected-source gating: production always asserts the
    # registered set (build_marker_frame default); ONLY the stub path
    # overrides, explicitly and loudly.
    expected_sources: list[str] | None = None
    if args.synthesize_stub is not None:
        expected_sources = STUB_SOURCES
        logger.warning(
            "stub mode: marker coverage assert gated on the %d stub sources %s, NOT the "
            "%d registered sources — production runs (no --synthesize-stub) always assert "
            "the registered set",
            len(STUB_SOURCES),
            STUB_SOURCES,
            len(REGISTERED_SOURCES),
        )

    for fam in families:
        if fam == "marker":
            analyze_marker(
                args.out_root,
                args.figures_dir,
                args.n_boot,
                selection_path=args.marker_selection,
                out_dir=args.out,
                expected_sources=expected_sources,
            )
        elif fam == "fact":
            analyze_fact(args.out_root, args.figures_dir, args.n_boot, out_dir=args.out)
        else:
            raise SystemExit(f"unknown family {fam!r}")
    logger.info("[phase=done] analysis complete")


if __name__ == "__main__":
    main()
