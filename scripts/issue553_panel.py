# ruff: noqa: RUF002, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — shared panel builders + inference helpers (module, no CLI).

Loads the two committed panels this task re-analyzes and enforces the plan
section 3.0 step-0 consistency gates before any new statistic is computed:

* ``build_margin_panel`` / ``step0_i532`` — the #532 ``logp_slot_followup``
  four-float panel (416 trained + 416 base per-cell JSONs, schema
  ``issue532_followup_logp_v1``), with per-cell channel means built from the
  ``per_q`` arrays exactly as ``scripts/issue532_followup_logp_slot.py`` built
  its committed ``analysis_logp.json`` (same arithmetic, asserted to 1e-6
  against the stored summaries AND the committed analysis values).
* ``load_i478_panel`` / ``step0_i478`` — the #478/#531 logit-rescore tidy
  table (``tidy_logit.parquet``, 56,000 x 25), gated on reproducing the
  committed ``summary_logit.json`` raw + partial Spearman reads (the partial
  implementation mirrors ``issue531_base_prior_reanalysis.partial_spearman``:
  rank-residualize on rank controls [min_dist, K], Pearson on residual ranks).

Statistical machinery reused by import (#539 precedent): the corrected-reads
inference helpers live in ``issue539_corrected_reads_inference`` /
``issue539_residual_per_cohort`` and are generic over (x, y, source-codes,
bystander-codes); this module adds only what #539 did not have — the two-way
ANOVA variance shares (Type-I, both orders), their bootstrap / permutation
nulls, the cell-axis (40-cluster) bootstrap for the #478 panel, OLS
cluster-bootstrap plumbing, and the CGM (Cameron-Gelbach-Miller 2011) two-way
plug-in SE cross-check (NEW code; cross-check only, never headline).

Inference defaults (plan section 3.0): percentile bootstrap re-estimating any
FE/residualization inside every resample; drawn cluster copies relabeled as
distinct groups; degenerate resamples dropped AND counted; seed 42;
two-sided add-one permutation p.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import issue539_corrected_reads_inference as i539inf
import issue539_residual_per_cohort as i539
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

# OpenBLAS spawns 32 pthreads on this VM and a 116x116 pinv / 2,800x116 lstsq
# degrades to ~0.6-2 s/call from thread-sync thrash (measured 2026-06-10).
# Every linear-algebra op in this task is small (<=56,000x3 ranks, <=116^2
# grams), so cap BLAS at 1 thread for the whole process. The controller must
# stay referenced at module scope or the limit is restored on GC.
_BLAS_SINGLE_THREAD = threadpool_limits(limits=1, user_api="blas")

# ── Constants ────────────────────────────────────────────────────────────────

MARKER_ID = 83399  # " ※" (leading space), Qwen-2.5-7B
EOS_ID = 151645  # <|im_end|>
GATE_TOL = 1e-6  # step-0 reproduction tolerance (plan section 3.0)
IDENTITY_TOL = 1e-6  # mean_marker_eos_margin identity (plan assumption 15)
DUP_PAIR = ("B1", "C1")  # quasi-duplicate prompts, cosine(B1,C1) = 1.0 exactly
FOLLOWUP_SCHEMA = "issue532_followup_logp_v1"
I478_NEGATIVE_PANEL = (
    "software_engineer",
    "kindergarten_teacher",
    "helpful_assistant",
    "no_persona",
)

I532_CHANNELS = ("dz_marker", "dz_eos", "dmargin", "margin_trained", "margin_base_matched")
CHANNEL_DISPLAY = {
    "dz_marker": "Marker-logit push Δz(※)",
    "dz_eos": "EOS-side change Δz(EOS)",
    "dmargin": "EOS-margin shift Δmargin",
    "margin_trained": "Trained EOS margin (level)",
    "margin_base_matched": "Base matched-slot EOS margin",
    "dz": "Marker-logit push Δz(※)",
}


# ── #532 followup panel ──────────────────────────────────────────────────────


def build_margin_panel(i532_dir: Path) -> dict:
    """Rebuild the 416-cell four-float panel from the committed followup JSONs.

    Returns a dict of aligned 416-long numpy arrays (sources x bystanders,
    bystander-fastest) for every channel + feature + label, plus per-q
    (416, 50) matrices for the split-half / cross-fit slice and the argmax
    composition read. Fails loud on any missing cell, schema drift, per-q
    slot misalignment, or identity violation (plan section 3.1).
    """
    followup = i532_dir / "logp_slot_followup"
    predictors = json.loads((i532_dir / "predictors.json").read_text())
    sources: list[str] = predictors["sources"]
    bystanders: list[str] = predictors["bystanders"]
    assert len(sources) == 16 and len(bystanders) == 26, (len(sources), len(bystanders))
    cos_m = np.asarray(predictors["cosine_matrix"], dtype=np.float64)
    gkl_m = np.asarray(predictors["gauss_kl_matrix"], dtype=np.float64)
    js_m = np.asarray(predictors["js_v1_matrix"], dtype=np.float64)
    assert cos_m.shape == gkl_m.shape == js_m.shape == (16, 26), cos_m.shape
    # B1/C1 quasi-duplicate (plan assumption 5).
    si, bi = sources.index("B1"), bystanders.index("C1")
    sj, bj = sources.index("C1"), bystanders.index("B1")
    assert cos_m[si, bi] == 1.0 and cos_m[sj, bj] == 1.0, "B1/C1 duplicate assert failed"

    a1 = json.loads((followup / "base_prior_logp.json").read_text())
    assert a1["schema_version"] == FOLLOWUP_SCHEMA, a1["schema_version"]
    assert a1["phase"] == "A1_base_prior_slots", a1["phase"]
    per_byst = a1["per_bystander"]
    assert len(per_byst) == 26, len(per_byst)
    prior_margin_own: dict[str, float] = {}
    prior_logp_own: dict[str, float] = {}
    for b, blk in per_byst.items():
        qs = blk["per_q"]
        assert len(qs) == 50, (b, len(qs))
        pm = float(np.mean([q["z_marker"] - q["z_eos"] for q in qs]))
        assert abs(pm - blk["summary"]["mean_marker_eos_margin"]) <= IDENTITY_TOL, b
        prior_margin_own[b] = pm
        prior_logp_own[b] = float(blk["summary"]["mean_logp_marker"])

    keys = [
        "dz_marker",
        "dz_eos",
        "dmargin",
        "margin_trained",
        "margin_base_matched",
        "dlogp",
        "trained_logp",
        "base_logp_matched",
    ]
    rows: dict[str, list] = {k: [] for k in keys}
    labels: dict[str, list] = {k: [] for k in ("source_cid", "bystander_label")}
    qmats: dict[str, list] = {
        k: [] for k in ("q_zm_t", "q_ze_t", "q_zm_b", "q_ze_b", "q_argmax_t", "q_argmax_b")
    }
    for src in sources:
        for byst in bystanders:
            t = json.loads((followup / "per_cell_trained" / f"{src}__{byst}.json").read_text())
            b = json.loads((followup / "per_cell_base" / f"{src}__{byst}.json").read_text())
            assert t["schema_version"] == b["schema_version"] == FOLLOWUP_SCHEMA
            assert t["phase"] == "A3_trained_on_own_R", t["phase"]
            assert b["phase"] == "A2_base_on_trained_R", b["phase"]
            tq, bq = t["per_q"], b["per_q"]
            assert len(tq) == len(bq) == 50, (src, byst, len(tq), len(bq))
            for x, y in zip(tq, bq, strict=True):
                assert x["slot_kind"] == y["slot_kind"], (src, byst)
                assert x["n_truncated_tokens"] == y["n_truncated_tokens"], (src, byst)
            zm_t = np.array([q["z_marker"] for q in tq], dtype=np.float64)
            ze_t = np.array([q["z_eos"] for q in tq], dtype=np.float64)
            zm_b = np.array([q["z_marker"] for q in bq], dtype=np.float64)
            ze_b = np.array([q["z_eos"] for q in bq], dtype=np.float64)
            lp_t = np.array([q["logp_marker"] for q in tq], dtype=np.float64)
            lp_b = np.array([q["logp_marker"] for q in bq], dtype=np.float64)
            mt = float(np.mean(zm_t - ze_t))
            mb = float(np.mean(zm_b - ze_b))
            # Plan assumption 15: mean-of-difference identity vs stored field.
            assert abs(mt - t["summary"]["mean_marker_eos_margin"]) <= IDENTITY_TOL, (src, byst)
            assert abs(mb - b["summary"]["mean_marker_eos_margin"]) <= IDENTITY_TOL, (src, byst)
            assert abs(float(np.mean(lp_t)) - t["summary"]["mean_logp_marker"]) <= IDENTITY_TOL
            rows["dz_marker"].append(float(np.mean(zm_t - zm_b)))
            rows["dz_eos"].append(float(np.mean(ze_t - ze_b)))
            rows["dmargin"].append(float(np.mean((zm_t - ze_t) - (zm_b - ze_b))))
            rows["margin_trained"].append(mt)
            rows["margin_base_matched"].append(mb)
            rows["dlogp"].append(float(np.mean(lp_t - lp_b)))
            rows["trained_logp"].append(float(np.mean(lp_t)))
            rows["base_logp_matched"].append(float(np.mean(lp_b)))
            labels["source_cid"].append(src)
            labels["bystander_label"].append(byst)
            qmats["q_zm_t"].append(zm_t)
            qmats["q_ze_t"].append(ze_t)
            qmats["q_zm_b"].append(zm_b)
            qmats["q_ze_b"].append(ze_b)
            qmats["q_argmax_t"].append(np.array([q["argmax_id"] for q in tq], dtype=np.int64))
            qmats["q_argmax_b"].append(np.array([q["argmax_id"] for q in bq], dtype=np.int64))

    panel: dict = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    panel["source_cid"] = np.asarray(labels["source_cid"])
    panel["bystander_label"] = np.asarray(labels["bystander_label"])
    for k, v in qmats.items():
        panel[k] = np.stack(v)
        assert panel[k].shape == (416, 50), (k, panel[k].shape)
    src_idx = {s: i for i, s in enumerate(sources)}
    byst_idx = {b: j for j, b in enumerate(bystanders)}
    si_arr = np.array([src_idx[s] for s in panel["source_cid"]])
    bj_arr = np.array([byst_idx[b] for b in panel["bystander_label"]])
    panel["cosine"] = cos_m[si_arr, bj_arr]
    panel["gauss_kl"] = gkl_m[si_arr, bj_arr]
    panel["js_v1"] = js_m[si_arr, bj_arr]
    panel["base_prior_binary"] = np.array(
        [float(predictors["base_prior"][b]) for b in panel["bystander_label"]]
    )
    panel["base_prior_extra_logp"] = np.array(
        [float(predictors["base_prior_extra_logp"][b]) for b in panel["bystander_label"]]
    )
    panel["prior_margin_own"] = np.array([prior_margin_own[b] for b in panel["bystander_label"]])
    panel["prior_logp_own"] = np.array([prior_logp_own[b] for b in panel["bystander_label"]])
    panel["is_instructed"] = np.array(
        [b.startswith("instr_") for b in panel["bystander_label"]], dtype=bool
    )
    panel["is_self"] = panel["source_cid"] == panel["bystander_label"]
    panel["is_dup_source"] = np.isin(panel["source_cid"], DUP_PAIR)
    panel["is_dup_bystander"] = np.isin(panel["bystander_label"], DUP_PAIR)
    panel["_n"] = len(panel["dmargin"])
    panel["_sources"] = sources
    panel["_bystanders"] = bystanders
    panel["_prior_margin_own_by_bystander"] = prior_margin_own
    panel["_prior_logp_own_by_bystander"] = prior_logp_own
    assert panel["_n"] == 416, panel["_n"]
    return panel


def cohort_masks_553(panel: dict) -> dict[str, np.ndarray]:
    """The plan section 3.0 cohorts/slices (plain-English names in outputs)."""
    is_ord = ~panel["is_instructed"]
    cross = is_ord & ~panel["is_self"]
    instr = panel["is_instructed"]
    no_dup = ~panel["is_dup_source"] & ~panel["is_dup_bystander"]
    return {
        "ordinary_all": is_ord,
        "ordinary_cross": cross,
        "instructed_strip": instr,
        "pooled_cohort_fe": cross | instr,
        "noB1C1_ordinary_cross": cross & no_dup,
        "noB1C1_instructed_strip": instr & ~panel["is_dup_source"],
        "noB1C1_pooled_cohort_fe": (cross | instr) & no_dup,
    }


def step0_i532(panel: dict, i532_dir: Path) -> dict:
    """Step-0 gate: reproduce committed ``analysis_logp.json`` values to 1e-6.

    Coded against the JSON KEYS (plan section 3.0), never plan prose:
    ``n_cells/n_ordinary/n_instructed``, ``spearman.dmargin.cosine.rho_union``,
    ``spearman.trained_logp.base_prior_logp.rho_union``, and
    ``graded_prior_spread.ordinary_sd_across_bystanders``. sys.exit(1) on any
    mismatch, BEFORE any new statistic.
    """
    committed = json.loads((i532_dir / "logp_slot_followup" / "analysis_logp.json").read_text())
    masks = cohort_masks_553(panel)
    checks: list[dict] = []

    def check(name: str, got: float, want: float, tol: float) -> None:
        checks.append(
            {
                "name": name,
                "got": float(got),
                "want": float(want),
                "pass": bool(abs(got - want) <= tol),
            }
        )

    check("n_cells", panel["_n"], committed["n_cells"], 0)
    check("n_ordinary", int(masks["ordinary_all"].sum()), committed["n_ordinary"], 0)
    check("n_instructed", int(masks["instructed_strip"].sum()), committed["n_instructed"], 0)
    check(
        "spearman.dmargin.cosine.rho_union",
        i539._spearman_rho(panel["cosine"], panel["dmargin"]),
        committed["spearman"]["dmargin"]["cosine"]["rho_union"],
        GATE_TOL,
    )
    # The committed analysis's graded prior is base_prior_logp.json's
    # per-bystander mean_logp_marker (issue532_followup_logp_slot.py line 682).
    check(
        "spearman.trained_logp.base_prior_logp.rho_union",
        i539._spearman_rho(panel["prior_logp_own"], panel["trained_logp"]),
        committed["spearman"]["trained_logp"]["base_prior_logp"]["rho_union"],
        GATE_TOL,
    )
    ord_priors = [
        panel["_prior_logp_own_by_bystander"][b]
        for b in panel["_bystanders"]
        if not b.startswith("instr_")
    ]
    check(
        "graded_prior_spread.ordinary_sd_across_bystanders",
        float(np.std(ord_priors)),
        committed["graded_prior_spread"]["ordinary_sd_across_bystanders"],
        GATE_TOL,
    )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        print(
            "STEP-0 GATE (#532 followup) FAILED — rebuilt panel diverges from the", file=sys.stderr
        )
        print("committed analysis_logp.json. NOT computing any new number.", file=sys.stderr)
        for c in failed:
            print(f"  FAIL {c['name']}: got {c['got']!r}, want {c['want']!r}", file=sys.stderr)
        sys.exit(1)
    print(f"[step0:i532] gate PASS ({len(checks)} checks reproduced to {GATE_TOL:g})")
    return {"pass": True, "checks": checks}


# ── #478 panel ───────────────────────────────────────────────────────────────


def load_i478_panel(parquet: Path) -> pd.DataFrame:
    """Load tidy_logit.parquet, derive ``dz_eos`` + ``run_id``, assert schema."""
    df = pd.read_parquet(parquet)
    assert df.shape == (56_000, 25), df.shape
    assert df["cell_id"].nunique() == 40
    assert sorted(df["seed"].unique().tolist()) == [42, 137]
    assert df["held_out_persona"].nunique() == 35
    assert df["question_idx"].nunique() == 20
    assert set(df["track"].unique()) == {"CORE"}
    # min_dist constant per (cell, persona); K constant per cell (assumptions 8/7).
    assert df.groupby(["cell_id", "held_out_persona"])["min_dist"].nunique().max() == 1
    assert df.groupby("cell_id")["K"].nunique().max() == 1
    df = df.copy()
    df["dz_eos"] = df["z_eos_trained"] - df["z_eos_base"]
    df["run_id"] = df["cell_id"] + "_seed" + df["seed"].astype(str)
    assert df["run_id"].nunique() == 80
    return df


def _partial_spearman_531(
    df: pd.DataFrame, y_col: str, x_col: str, control_cols: list[str]
) -> float:
    """Rank-residual partial Spearman, mirroring issue531_base_prior_reanalysis."""
    sub = df[[y_col, x_col, *control_cols]].dropna()
    yr = rankdata(sub[y_col].to_numpy(), method="average")
    xr = rankdata(sub[x_col].to_numpy(), method="average")
    design = np.column_stack(
        [np.ones(len(sub))] + [rankdata(sub[c].to_numpy(), method="average") for c in control_cols]
    )
    cy, *_ = np.linalg.lstsq(design, yr, rcond=None)
    cx, *_ = np.linalg.lstsq(design, xr, rcond=None)
    r, _ = spearmanr(xr - design @ cx, yr - design @ cy)
    return float(r)


def step0_i478(df: pd.DataFrame, summary_path: Path) -> dict:
    """Step-0 gate: reproduce committed ``summary_logit.json`` reads to 1e-6.

    Coded against the JSON KEYS (statistics-critic correction carried in plan
    section 3.0: ``partial_dmargin_vs_marginbase`` is the PARTIAL −0.4169, the
    raw is −0.6411; plus the +0.70/+0.77 absolute partials). sys.exit(1) on
    any mismatch.
    """
    committed = json.loads(summary_path.read_text())["spearman"]
    checks: list[dict] = []

    def check(name: str, got: float) -> None:
        want = float(committed[name]["rho_point"])
        checks.append(
            {
                "name": name,
                "got": float(got),
                "want": want,
                "pass": bool(abs(got - want) <= GATE_TOL),
            }
        )

    raw, _ = spearmanr(df["margin_base"], df["dmargin"])
    check("raw_dmargin_vs_marginbase", float(raw))
    ctl = ["min_dist", "K"]
    check("partial_dmargin_vs_marginbase", _partial_spearman_531(df, "dmargin", "margin_base", ctl))
    check("partial_ztrained_vs_zbase", _partial_spearman_531(df, "z_trained", "z_base", ctl))
    check(
        "partial_margintrained_vs_marginbase",
        _partial_spearman_531(df, "margin_trained", "margin_base", ctl),
    )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        print("STEP-0 GATE (#478 parquet) FAILED — parquet reads diverge from the", file=sys.stderr)
        print("committed summary_logit.json. NOT computing any new number.", file=sys.stderr)
        for c in failed:
            print(f"  FAIL {c['name']}: got {c['got']!r}, want {c['want']!r}", file=sys.stderr)
        sys.exit(1)
    print(f"[step0:i478] gate PASS ({len(checks)} checks reproduced to {GATE_TOL:g})")
    return {"pass": True, "checks": checks}


def aggregate_run_persona(df: pd.DataFrame) -> pd.DataFrame:
    """(run x persona) means — the #478 anatomy unit (plan section 3.2).

    Question-level variation re-enters through cluster bootstraps; aggregation
    avoids 20x pseudo-replication in the FE decomposition (plan section 11).
    """
    agg = (
        df.groupby(["run_id", "cell_id", "seed", "K", "held_out_persona"], as_index=False)
        .agg(
            dz=("dz", "mean"),
            dz_eos=("dz_eos", "mean"),
            dmargin=("dmargin", "mean"),
            margin_trained=("margin_trained", "mean"),
            margin_base=("margin_base", "mean"),
            z_eos_trained=("z_eos_trained", "mean"),
            z_eos_base=("z_eos_base", "mean"),
            base_prior=("base_prior", "mean"),
            min_dist=("min_dist", "first"),
        )
        .sort_values(["run_id", "held_out_persona"])
        .reset_index(drop=True)
    )
    assert len(agg) == 2_800, len(agg)
    # Complete balanced 80 x 35 panel (plan assumption 7).
    assert agg.groupby("run_id")["held_out_persona"].nunique().eq(35).all()
    return agg


# ── Fast exact two-way FE solver (gram/pinv; drop-in for the #539 lstsq) ─────


def _twoway_gram_coefs(
    rhs: np.ndarray, ac: np.ndarray, bc: np.ndarray, n_a: int, n_b: int
) -> np.ndarray:
    """Min-norm least-squares coefficients of [1 | A dummies | B dummies] on rhs.

    Identical estimand to ``np.linalg.lstsq(design, rhs)`` via the exact
    identity ``X⁺ = (X'X)⁺ X'`` — the pseudoinverse of the (1+n_a+n_b)² gram
    matrix (built from count vectors + the A×B crosstab, never forming the
    n×(1+n_a+n_b) design) applied to X'rhs. The full lstsq on the explicit
    design takes ~2 s/call at 2,800×116 in this environment (gelsd), which
    made the plan's 10,000-rep FE-re-estimating bootstraps wall-infeasible;
    this solver is exact-equivalent (asserted at import on random data AND on
    the observed panels inside every consuming script) at ~1,000× the speed.

    ``rhs`` may be (n,) or (n, k). Returns coefficients shaped like lstsq's.
    """
    rhs2 = rhs if rhs.ndim == 2 else rhs[:, None]
    n = len(ac)
    count_a = np.bincount(ac, minlength=n_a).astype(np.float64)
    count_b = np.bincount(bc, minlength=n_b).astype(np.float64)
    crosstab = np.bincount(ac * n_b + bc, minlength=n_a * n_b).astype(np.float64).reshape(n_a, n_b)
    m = 1 + n_a + n_b
    gram = np.zeros((m, m))
    gram[0, 0] = n
    gram[0, 1 : 1 + n_a] = count_a
    gram[1 : 1 + n_a, 0] = count_a
    gram[0, 1 + n_a :] = count_b
    gram[1 + n_a :, 0] = count_b
    gram[1 : 1 + n_a, 1 : 1 + n_a] = np.diag(count_a)
    gram[1 + n_a :, 1 + n_a :] = np.diag(count_b)
    gram[1 : 1 + n_a, 1 + n_a :] = crosstab
    gram[1 + n_a :, 1 : 1 + n_a] = crosstab.T
    xty = np.empty((m, rhs2.shape[1]))
    for k in range(rhs2.shape[1]):
        v = rhs2[:, k]
        xty[0, k] = v.sum()
        xty[1 : 1 + n_a, k] = np.bincount(ac, weights=v, minlength=n_a)
        xty[1 + n_a :, k] = np.bincount(bc, weights=v, minlength=n_b)
    coef = np.linalg.pinv(gram) @ xty
    return coef if rhs.ndim == 2 else coef[:, 0]


def fast_twoway_resid_pair(
    x: np.ndarray, y: np.ndarray, sc: np.ndarray, bc: np.ndarray, n_s: int, n_b: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop-in for ``i539inf._twoway_resid_pair`` (same signature, same estimand).

    Residuals of a least-squares projection are invariant to the gauge, so the
    gram-solver coefficients reproduce the lstsq residuals exactly (asserted
    at import below + per-script drift asserts against the parent's fail-loud
    ``_twoway_fe_residualize`` on the observed data).
    """
    rhs = np.column_stack([x, y])
    coef = _twoway_gram_coefs(rhs, sc, bc, n_s, n_b)
    fitted = coef[0][None, :] + coef[1 : 1 + n_s][sc] + coef[1 + n_s :][bc]
    resid = rhs - fitted
    return resid[:, 0], resid[:, 1]


def _selfcheck_fast_twoway() -> None:
    """Import-time equivalence assert vs the #539 lstsq implementation.

    Random unbalanced panel WITH an absent group level (exercises the
    min-norm rank-deficient path). Aborts the import on drift > 1e-8.
    """
    rng = np.random.default_rng(0)
    n, n_s, n_b = 320, 9, 13
    sc = rng.integers(0, n_s - 1, n)  # level n_s-1 absent
    bc = rng.integers(0, n_b, n)
    x, y = rng.normal(size=n), rng.normal(size=n)
    rx0, ry0 = _I539_TWOWAY_RESID_PAIR_LSTSQ(x, y, sc, bc, n_s, n_b)
    rx1, ry1 = fast_twoway_resid_pair(x, y, sc, bc, n_s, n_b)
    drift = max(float(np.max(np.abs(rx0 - rx1))), float(np.max(np.abs(ry0 - ry1))))
    assert drift < 1e-8, f"fast two-way gram solver drifts from lstsq: {drift!r}"


# Capture the original, verify equivalence, then patch it in for THIS process
# so the imported #539 bootstrap/permutation loops use the fast exact solver.
_I539_TWOWAY_RESID_PAIR_LSTSQ = i539inf._twoway_resid_pair
_selfcheck_fast_twoway()
i539inf._twoway_resid_pair = fast_twoway_resid_pair


# ── Two-way ANOVA variance shares (Type-I, both orders) ──────────────────────


def _twoway_resid(y: np.ndarray, ac: np.ndarray, bc: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Exact two-way FE residual (single RHS; fast gram solver)."""
    r, _ = fast_twoway_resid_pair(y, y, ac, bc, n_a, n_b)
    return r


def _ss_after_oneway(y: np.ndarray, codes: np.ndarray, n_groups: int) -> float:
    """Residual SS after intercept + one factor (exact: residual = y − group mean)."""
    gm = i539inf._group_mean_by_code(y, codes, n_groups)
    return float(np.sum((y - gm) ** 2))


def anova_shares(y: np.ndarray, ac: np.ndarray, bc: np.ndarray, n_a: int, n_b: int) -> dict:
    """Type-I variance shares of factors A and B in BOTH orders + pair residual.

    On a complete balanced panel the decomposition is orthogonal and the two
    orders agree; on unbalanced panels they differ — the order-swap diff is the
    diagnostic the plan registers (section 3.2.1 / 3.4.1). The pair share is
    order-invariant (residual SS of the full two-way fit).
    """
    yc = y - y.mean()
    ss_tot = float(yc @ yc)
    ss_two = float(np.sum(_twoway_resid(y, ac, bc, n_a, n_b) ** 2))
    ss_after_a = _ss_after_oneway(y, ac, n_a)
    ss_after_b = _ss_after_oneway(y, bc, n_b)
    if ss_tot <= 0:
        nan = float("nan")
        return {
            k: nan
            for k in (
                "a_first_share_a",
                "a_first_share_b",
                "b_first_share_a",
                "b_first_share_b",
                "pair_share",
            )
        }
    return {
        "a_first_share_a": (ss_tot - ss_after_a) / ss_tot,
        "a_first_share_b": (ss_after_a - ss_two) / ss_tot,
        "b_first_share_b": (ss_tot - ss_after_b) / ss_tot,
        "b_first_share_a": (ss_after_b - ss_two) / ss_tot,
        "pair_share": ss_two / ss_tot,
    }


def shares_cell_bootstrap(
    y: np.ndarray,
    a_labels: np.ndarray,
    b_labels: np.ndarray,
    n_boot: int,
    seed: int,
    dominance: bool = False,
) -> dict:
    """Cell-level percentile bootstrap on the variance shares themselves.

    Shares (both orders) are re-derived per resample (FE re-estimated). When
    ``dominance`` is set, also bootstraps the registered D1(a) statistic
    ``a_share − max(b_share, pair_share)`` (A-first order, the analogue of the
    inline source-first read; the order-swap variant is reported alongside).
    """
    rng = np.random.default_rng(seed)
    _, ac = np.unique(a_labels, return_inverse=True)
    _, bc = np.unique(b_labels, return_inverse=True)
    n_a, n_b = int(ac.max()) + 1, int(bc.max()) + 1
    n = len(y)
    cols: dict[str, list[float]] = {
        k: []
        for k in (
            "a_first_share_a",
            "a_first_share_b",
            "b_first_share_a",
            "b_first_share_b",
            "pair_share",
        )
    }
    dom: list[float] = []
    dom_swap: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sh = anova_shares(y[idx], ac[idx], bc[idx], n_a, n_b)
        if any(np.isnan(v) for v in sh.values()):
            n_deg += 1
            continue
        for k in cols:
            cols[k].append(sh[k])
        if dominance:
            dom.append(sh["a_first_share_a"] - max(sh["a_first_share_b"], sh["pair_share"]))
            dom_swap.append(sh["b_first_share_a"] - max(sh["b_first_share_b"], sh["pair_share"]))
    out = {k: i539inf._percentile_summary(v, n_boot, n_deg) for k, v in cols.items()}
    if dominance:
        out["dominance_a_first"] = i539inf._percentile_summary(dom, n_boot, n_deg)
        out["dominance_b_first_order_swap"] = i539inf._percentile_summary(dom_swap, n_boot, n_deg)
    return out


def share_permutation_null(
    y: np.ndarray,
    a_labels: np.ndarray,
    b_labels: np.ndarray,
    permute_axis: str,
    n_perm: int,
    seed: int,
) -> dict:
    """FE-respecting permutation null for ONE factor's Type-I-last share.

    Permutes the ``permute_axis`` labels WITHIN each level of the other axis
    (the other factor's structure — and therefore its one-way residual SS —
    is exactly preserved), recomputing the permuted factor's share beyond the
    preserved factor per rep. Persisted in full per plan concern 13.5 (FE
    shares are upward-biased; the null distribution calibrates the observed
    share). Two-sided add-one p.
    """
    rng = np.random.default_rng(seed)
    _, ac = np.unique(a_labels, return_inverse=True)
    _, bc = np.unique(b_labels, return_inverse=True)
    n_a, n_b = int(ac.max()) + 1, int(bc.max()) + 1
    yc = y - y.mean()
    ss_tot = float(yc @ yc)
    if permute_axis == "a":
        keep_codes, perm_codes, n_keep, n_perm_groups = bc, ac, n_b, n_a
    else:
        keep_codes, perm_codes, n_keep, n_perm_groups = ac, bc, n_a, n_b
    ss_after_keep = _ss_after_oneway(y, keep_codes, n_keep)

    def share_of(pc: np.ndarray) -> float:
        if permute_axis == "a":
            ss_two = float(np.sum(_twoway_resid(y, pc, bc, n_perm_groups, n_b) ** 2))
        else:
            ss_two = float(np.sum(_twoway_resid(y, ac, pc, n_a, n_perm_groups) ** 2))
        return (ss_after_keep - ss_two) / ss_tot

    observed = share_of(perm_codes)
    groups = [np.where(keep_codes == g)[0] for g in range(n_keep)]
    null: list[float] = []
    pc = perm_codes.copy()
    for _ in range(n_perm):
        for g_idx in groups:
            pc[g_idx] = perm_codes[g_idx][rng.permutation(len(g_idx))]
        null.append(share_of(pc))
    null_arr = np.asarray(null)
    count = int((null_arr >= observed).sum())  # share is one-sided-large by construction
    return {
        "observed_share_beyond_other_factor": float(observed),
        "p_perm": float((1 + count) / (n_perm + 1)),
        "null_mean": float(null_arr.mean()),
        "null_sd": float(null_arr.std()),
        "n_perm": n_perm,
        "permuted_axis": permute_axis,
        "method": "labels of the permuted factor shuffled WITHIN each level of the other "
        "factor; share beyond the preserved factor recomputed per rep; one-sided "
        "(share >= observed), add-one",
        "null_distribution": [round(float(v), 6) for v in null_arr],
    }


def fe_vector(
    y: np.ndarray, codes: np.ndarray, other_codes: np.ndarray, n: int, n_other: int
) -> np.ndarray:
    """Centered FE coefficient vector for one factor from the two-way fit.

    Min-norm gauge freedom on a CONNECTED panel is exactly the two uniform
    shifts (factor block vs intercept), so centering the factor's coefficient
    block makes the vector gauge-invariant. Solved via the exact gram solver
    (same min-norm solution as lstsq — see ``_twoway_gram_coefs``).
    """
    coef = _twoway_gram_coefs(y.astype(np.float64), codes, other_codes, n, n_other)
    fe = coef[1 : 1 + n]
    return fe - fe.mean()


# ── Cluster bootstrap plumbing ───────────────────────────────────────────────


def cluster_boot_stat(
    labels: np.ndarray,
    stat_fn,
    n_boot: int,
    seed: int,
) -> tuple[list, int, int]:
    """Generic cluster percentile bootstrap.

    ``stat_fn(idx, copy_codes)`` receives the resampled row indices plus a
    fresh integer code per drawn cluster copy (the #539 relabeling convention,
    so FE re-estimation treats two draws of one cluster as distinct groups);
    returns a float/array or None for a degenerate resample. Returns
    (stats, n_boot, n_degenerate).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    idx_of = {c: np.where(labels == c)[0] for c in uniq}
    stats: list = []
    n_deg = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[c] for c in chosen])
        copy_codes = np.repeat(np.arange(len(chosen)), [len(idx_of[c]) for c in chosen])
        s = stat_fn(idx, copy_codes)
        if s is None:
            n_deg += 1
            continue
        stats.append(s)
    return stats, n_boot, n_deg


def cluster_boot_twoway_spearman_cellaxis(
    x: np.ndarray,
    y: np.ndarray,
    run_labels: np.ndarray,
    persona_labels: np.ndarray,
    cell_labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Cell-axis (40-cluster) bootstrap for the pair-corrected #478 reads.

    Plan section 3.2.3 (statistics-reconciler REVISION): min_dist is constant
    per (cell, persona) and a cell's two seeds share one training mix, so the
    resampling unit for min_dist reads must include the cell axis. Drawn cell
    copies bring BOTH seed-runs; run codes are relabeled per copy (fresh group
    per drawn run), persona codes kept; two-way FE re-estimated per resample.
    """
    rng = np.random.default_rng(seed)
    cells = np.unique(cell_labels)
    idx_of = {c: np.where(cell_labels == c)[0] for c in cells}
    # Local run code within each cell (0/1 by sorted run label) for relabeling.
    local_run: dict = {}
    for c in cells:
        runs = np.unique(run_labels[idx_of[c]])
        m = {r: k for k, r in enumerate(sorted(runs.tolist()))}
        local_run[c] = np.array([m[r] for r in run_labels[idx_of[c]]])
    _, pc_full = np.unique(persona_labels, return_inverse=True)
    n_p = int(pc_full.max()) + 1
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        chosen = rng.choice(cells, size=len(cells), replace=True)
        idx = np.concatenate([idx_of[c] for c in chosen])
        run_codes = np.concatenate([local_run[c] + 2 * k for k, c in enumerate(chosen)])
        n_r = 2 * len(chosen)
        pc = pc_full[idx]
        xt, yt = i539inf._twoway_resid_pair(x[idx], y[idx], run_codes, pc, n_r, n_p)
        if i539._is_degenerate(xt) or i539._is_degenerate(yt):
            n_deg += 1
            continue
        rhos.append(i539._fast_spearman(xt, yt))
    out = i539inf._percentile_summary(rhos, n_boot, n_deg)
    out["n_clusters"] = len(cells)
    return out


def wider_ci(cis: dict[str, dict]) -> dict:
    """Pick the WIDER 95% CI among named one-way cluster CIs (plan section 3.0)."""
    best_name, best = None, None
    for name, ci in cis.items():
        lo, hi = ci.get("low"), ci.get("high")
        if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
            continue
        width = hi - lo
        if best is None or width > best[0]:
            best_name, best = name, (width, lo, hi)
    if best is None:
        return {"axis": None, "low": float("nan"), "high": float("nan")}
    return {"axis": best_name, "low": best[1], "high": best[2]}


# ── OLS + CGM two-way plug-in SE (NEW; cross-check only, never headline) ─────


def ols_fit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS via lstsq; returns (coef, resid, pinv(X'X))."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    xtx_inv = np.linalg.pinv(design.T @ design)
    return coef, resid, xtx_inv


def _cluster_meat(design: np.ndarray, resid: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Sum over clusters of (X_g' u_g)(X_g' u_g)'."""
    k = design.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(codes):
        m = codes == g
        s = design[m].T @ resid[m]
        meat += np.outer(s, s)
    return meat


def cgm_twoway_se(
    design: np.ndarray,
    resid: np.ndarray,
    xtx_inv: np.ndarray,
    a_codes: np.ndarray,
    b_codes: np.ndarray,
) -> dict:
    """CGM (Cameron-Gelbach-Miller 2011) two-way plug-in covariance.

    V = V_a + V_b − V_(a∩b); on these panels the intersection clustering is
    the singleton cell, so V_(a∩b) is the HC0 sandwich. No small-sample
    correction (documented; cross-check only). Non-PSD outcomes (possible at
    16 clusters — plan concern 13.6) are reported + flagged, never NaN'd
    silently.
    """
    inter_codes = a_codes.astype(np.int64) * (int(b_codes.max()) + 1) + b_codes.astype(np.int64)
    v = (
        xtx_inv
        @ (
            _cluster_meat(design, resid, a_codes)
            + _cluster_meat(design, resid, b_codes)
            - _cluster_meat(design, resid, inter_codes)
        )
        @ xtx_inv
    )
    diag = np.diag(v).copy()
    eigmin = float(np.min(np.linalg.eigvalsh((v + v.T) / 2)))
    non_psd = bool(eigmin < -1e-12) or bool((diag < 0).any())
    se = np.where(diag >= 0, np.sqrt(np.clip(diag, 0, None)), np.nan)
    return {
        "se": [float(s) for s in se],
        "variance_diag": [float(d) for d in diag],
        "min_eigenvalue": eigmin,
        "non_psd_flag": non_psd,
        "n_clusters_a": int(a_codes.max()) + 1,
        "n_clusters_b": int(b_codes.max()) + 1,
        "small_sample_correction": "none (plug-in; cross-check only, never headline)",
    }


# ── Output plumbing ──────────────────────────────────────────────────────────


def result_metadata(args: argparse.Namespace, script: str) -> dict:
    """Reproducibility metadata block for every output JSON."""
    import scipy

    return {
        "task": 553,
        "script": script,
        "git_commit": i539._git_commit(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_cluster_boot": args.n_cluster_boot,
        "n_marginal_boot": args.n_marginal_boot,
        "n_perm": args.n_perm,
        "argv": sys.argv[1:],
    }


def ivr_entry(
    quantity: str, inline_value, reviewed_value, convention_changed: bool, note: str
) -> dict:
    """One ``inline_vs_reviewed`` row (plan section 8.4)."""
    return {
        "quantity": quantity,
        "inline_value": inline_value,
        "reviewed_value": reviewed_value,
        "convention_changed": convention_changed,
        "note": note,
    }


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_np_default) + "\n")
    print(f"[write] {path}")


def common_parser(short_desc: str) -> argparse.ArgumentParser:
    """Shared CLI (plan section 10): one code path for smoke and production."""
    parser = argparse.ArgumentParser(
        description=short_desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--i532-dir", type=Path, default=Path("eval_results/issue_532"))
    parser.add_argument(
        "--i478-parquet",
        type=Path,
        default=Path("eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_553"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/issue_553"))
    parser.add_argument("--n-boot", type=int, default=10_000, dest="n_boot")
    parser.add_argument("--n-cluster-boot", type=int, default=2_000, dest="n_cluster_boot")
    parser.add_argument("--n-marginal-boot", type=int, default=2_000, dest="n_marginal_boot")
    parser.add_argument("--n-perm", type=int, default=10_000, dest="n_perm")
    parser.add_argument("--seed", type=int, default=42)
    return parser
