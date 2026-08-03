"""Issue #1482 — side-specific SAE features: activity-stratified enrichment + browser.

Companion to `figures/issue_1482/side_specificity/side_specificity_fullwidth.png`.
Two halves, in this order on the page:

  (b) QUANTITATIVE — for CONTEXT-ONLY and ANSWER-ONLY features against the
      TWO-SIDED baseline, the enrichment of every judged class level
      (abstraction / content_type / speaker_property / functional_role /
      interpretable from #1773, plus the Gurnee promoting class),
      ACTIVITY-STRATIFIED with an exact conditional null.
  (a) BROWSABLE — one card per side-specific feature with its #1773 description,
      the five judged axis labels, occupancy, R^2 and the run-length token counts.

The enrichment table is what licenses a claim; the cards are what make it
inspectable. The page is ordered that way deliberately.

WHY STRATIFICATION IS NOT OPTIONAL HERE. Side-specific features are drastically
rarer than two-sided ones: median total row occupancy is 2 of 120,000 rows for
BOTH side-specific classes versus 1,283 for two-sided. A crude enrichment would
therefore mostly re-report activity composition, which has already bitten this
line twice (the `functional_role` unresolved effect dissolved into activity; the
unanimity dose-response needed activity matching). This script reports the crude
and the activity-standardized enrichment SIDE BY SIDE so any such dissolution is
visible rather than hidden.

THE STRUCTURAL EXCLUSION. A two-sided feature must fire on at least one context
row AND one answer row, so its occupancy is >= 2 and NO two-sided feature can
ever have occupancy 1. The occupancy-1 stratum (587 context-only, 725 answer-only
— about a third of each class) therefore has no baseline and is UNMATCHABLE by
construction, not by sampling. Those features are excluded from the standardized
estimate, counted, and reported; they are never silently pooled.

METHOD. Occupancy strata are fixed integer bins (1, 2, 3, 4-5, 6-9, ... 10000+).
Within each stratum the group is, under the null, a simple random draw from the
pooled {group + baseline} members of that stratum, so the group's per-stratum
level count is exactly Hypergeometric — no approximate permutation needed. The
reported effect is a directly standardized rate ratio: the group's own occupancy
distribution is used to reweight BOTH the group rate and the baseline rate, so
the comparison is between populations matched on occupancy. Two-sided p-values
come from `n_draws` exact hypergeometric draws (vectorised over draws x strata,
recomputing the baseline complement each draw so the null is fully consistent),
and are BH-FDR corrected across every (group, axis, level) test.

INTERPRETATION CONTRACT, carried into every artifact: an enrichment is an
ASSOCIATION between a judged label and a side class, measured on observational
data. It is not a causal or mechanistic claim about what the feature does. And
"never fires on side X" means zero across all 120,000 fit rows at ROW-OCCUPANCY
grain (active anywhere in the span) — a strong criterion: a feature firing at a
single token of a single answer is NOT answer-only.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402
import scipy  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482-sidespec")

DICT_SIZE = 131_072
N_FIT = 120_000
SCAN = PROJECT_ROOT / "data/issue_1482/fullwidth/fused_scan.npz"
DISCRETE = (
    PROJECT_ROOT / "eval_results/issue_1482/predictor_battery/fullwidth_discrete_covariates.npz"
)
RUNLEN = PROJECT_ROOT / "eval_results/issue_1482/run_length/run_length_perfeature.npz"
RUNLEN_META = PROJECT_ROOT / "eval_results/issue_1482/run_length/run_length_perfeature.meta.json"
R2_NPZ = Path("/mnt/eps-data/thomasjiralerspong/issue1482_sidespec/ridge__mean_perfeature.npz")
R2_HF_PATH = "issue1482_densesae_fullwidth/perfeature/ridge__mean_perfeature.npz"
LABELS_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1773_fulldict/labels_upload")
OUT_DIR = PROJECT_ROOT / "eval_results/issue_1482/side_specific"
DASH_PATH = PROJECT_ROOT / "tasks/awaiting_promotion/1482/artifacts/side_specific_dashboard.html"
# dashboard/public/ IS served; the task-artifact path is NOT (no artifacts route).
# Write BOTH from here — a hand-copy goes stale on the next regeneration.
PUBLIC_DASH_PATH = PROJECT_ROOT / "dashboard/public/side-specific-1482.html"

CENSUS_EXPECTED = {"context_only": 1654, "two_sided": 126348, "answer_only": 2164, "dead": 906}
SCORED_EXPECTED = 121_111
# integer occupancy strata; bin 1-1 is retained in the table but can never carry a
# two-sided baseline (a two-sided feature needs >=1 row on each side => occ >= 2)
OCC_EDGES = np.array([1, 2, 3, 4, 6, 10, 20, 50, 100, 300, 1000, 3000, 10000, 10**9])
MIN_BASELINE_PER_STRATUM = 20
N_DRAWS = 20_000
NULL_SEED = 14_820_001
AXES = ("abstraction", "content_type", "speaker_property", "functional_role", "interpretable")
GURNEE_NAMES = {0: "other", 1: "promoting", 2: "suppressing", 3: "partition"}
GROUPS = {"context_only": 0, "answer_only": 2}

CAVEATS = [
    "An enrichment here is an ASSOCIATION between a judged label and a side class, "
    "measured on observational data. It is not a causal or mechanistic claim about "
    "what a feature does.",
    "Descriptions and axis labels come from #1773, whose standing caveat is that they "
    "are SEARCH-INDEX-ONLY (neighbour discrimination 0.322 against a 0.50 bar). Any "
    "reading of these features as recognisable KINDS rests entirely on those labels "
    "and is the least evidenced thing in this round. The enrichment numbers are "
    "measured; the kind-names a reader forms from the card text are not.",
    "'Never fires on side X' means zero across all 120,000 fit rows at ROW-OCCUPANCY "
    "grain (the feature counts as active if it fires ANYWHERE in the span). This is a "
    "strong criterion: a feature firing at a single token of a single answer is NOT "
    "answer-only.",
    "Side-specific features are far rarer than two-sided ones (median total occupancy "
    "2 of 120,000 rows vs 1,283), so CRUDE enrichment largely re-reports activity "
    "composition. The activity-standardized column is the one to read; both are shown "
    "so any dissolution into activity is visible.",
    "Occupancy 1 carries NO two-sided baseline by construction (a two-sided feature "
    "needs at least one row on each side, so occupancy >= 2). Those features are "
    "excluded from the standardized estimate and counted separately — never pooled.",
    "Per-feature R^2 is the dense-context -> SAE-answer full-width ridge read. A "
    "feature is UNSCORED where its holdout answer variance is zero; unscored is shown "
    "as unscored, never as R^2 = 0.",
    "Token-level counts (ctx/ans tokens active) come from the run-length capture, "
    "which is a 2,000-ROW SUBSAMPLE — a different and much smaller denominator than "
    "the 120,000-row census that defines the side classes. They describe depth on the "
    "live side; they are NOT the evidence for one-sidedness.",
]


def _log(msg: str) -> None:
    logger.info("%s", msg)


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else "unavailable-no-git-checkout"


def _provenance() -> dict:
    return {
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)
    _log(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


# ── inputs ──────────────────────────────────────────────────────────────────────────


def load_inputs(args) -> dict:
    """Load every substrate array and FAIL LOUD on any census/shape drift."""
    with np.load(SCAN) as z:
        cnt = z["cnt_fit"].astype(np.int64)  # fit rows active in the ANSWER span
        psi = z["psi_cnt_fit"].astype(np.int64)  # fit rows active in the CONTEXT span
        cnt_ho = z["cnt_holdout"].astype(np.int64)
        n_fit = int(z["n_fit"])
    assert n_fit == N_FIT, n_fit
    d = np.load(DISCRETE)
    side = d["side_class"]
    gurnee = d["gurnee_class"]
    got = {
        "context_only": int((side == 0).sum()),
        "two_sided": int((side == 1).sum()),
        "answer_only": int((side == 2).sum()),
        "dead": int((side == -1).sum()),
    }
    assert got == CENSUS_EXPECTED, f"side_class census drift: {got} != {CENSUS_EXPECTED}"
    # the scan must independently reproduce the same partition
    scan_ctx_only = int(((psi > 0) & (cnt == 0)).sum())
    scan_ans_only = int(((cnt > 0) & (psi == 0)).sum())
    assert (scan_ctx_only, scan_ans_only) == (
        CENSUS_EXPECTED["context_only"],
        CENSUS_EXPECTED["answer_only"],
    ), (scan_ctx_only, scan_ans_only)
    _log(f"census verified against BOTH the covariates npz and the raw scan: {got}")

    if not R2_NPZ.exists():
        from explore_persona_space.orchestrate import hub

        R2_NPZ.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            path_in_repo=R2_HF_PATH,
            target=R2_NPZ,
        )
    rz = np.load(R2_NPZ)
    r2, scored = rz["r2"].astype(np.float64), rz["scored"].astype(bool)
    assert int(scored.sum()) == SCORED_EXPECTED, int(scored.sum())
    _log(f"R^2 (dense->SAE full width): {int(scored.sum()):,} scored of {scored.size:,}")

    runlen, runlen_rows = None, None
    if RUNLEN.exists():
        rl = np.load(RUNLEN)
        runlen = {k: rl[k] for k in ("ctx_tokens_active", "ans_tokens_active")}
        if RUNLEN_META.exists():
            meta = json.loads(RUNLEN_META.read_text())
            runlen_rows = int(meta["gates"]["gate1_row_occupancy_census"]["n_rows"])
        _log(f"run-length token counts present (subsample n_rows={runlen_rows})")
    else:
        _log("run-length capture ABSENT — token-count columns will be omitted")

    labels = load_labels(args.labels_dir)
    return {
        "cnt": cnt,
        "psi": psi,
        "cnt_holdout": cnt_ho,
        "occ": cnt + psi,
        "side": side,
        "gurnee": gurnee,
        "r2": r2,
        "scored": scored,
        "runlen": runlen,
        "runlen_rows": runlen_rows,
        "labels": labels,
    }


def load_labels(labels_dir: Path) -> dict:
    """#1773 autointerp descriptions + the five judged axis labels, full width."""
    desc: dict[int, str] = {}
    axes: dict[str, np.ndarray] = {a: np.full(DICT_SIZE, "", dtype=object) for a in AXES}
    for p in sorted(labels_dir.glob("descriptions.shard*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                fid = int(r.get("feat_id", -1))
                if 0 <= fid < DICT_SIZE:
                    desc[fid] = r.get("description") or ""
    for p in sorted(labels_dir.glob("axis_labels.shard*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                fid = int(r.get("feat_id", -1))
                ax = str(r.get("axis"))
                if 0 <= fid < DICT_SIZE and ax in axes:
                    axes[ax][fid] = r.get("label") or ""
    cov = {a: int((axes[a] != "").sum()) for a in AXES}
    _log(f"#1773 labels: {len(desc):,} descriptions; axis coverage {cov}")
    no_ev = labels_dir / "no_evidence_features.json"
    excluded = set(json.loads(no_ev.read_text())["feat_ids"]) if no_ev.exists() else set()
    return {"desc": desc, "axes": axes, "coverage": cov, "no_evidence": excluded}


def label_coverage_by_side(inp: dict) -> dict:
    """Per-side #1773 coverage — the structural gap this round is really about.

    #1773 draws its activating windows from the feature's own ANSWER-side
    activation (`ans_max` quantile bins) and `exclude_zero_evidence` drops any
    feature with no activating windows at zero API cost. A context-only feature
    has none BY THE VERY PROPERTY that makes it context-only, so it was never
    described or axis-labelled. This function measures that rather than assuming
    it, and checks the exclusion set is exactly {context-only} U {dead}."""
    side = inp["side"]
    axcol = inp["labels"]["axes"]["abstraction"]
    out = {}
    for nm, code in (("context_only", 0), ("two_sided", 1), ("answer_only", 2), ("dead", -1)):
        m = side == code
        out[nm] = {
            "n": int(m.sum()),
            "n_axis_labelled": int((axcol[m] != "").sum()),
            "n_described": int(
                sum(1 for f in np.nonzero(m)[0].tolist() if inp["labels"]["desc"].get(f))
            ),
        }
        out[nm]["axis_labelled_fraction"] = out[nm]["n_axis_labelled"] / max(out[nm]["n"], 1)
    ne = inp["labels"]["no_evidence"]
    one_sided_dead = set(np.nonzero((side == 0) | (side == -1))[0].tolist())
    out["no_evidence_exclusion"] = {
        "n_excluded": len(ne),
        "equals_context_only_union_dead": bool(ne == one_sided_dead) if ne else None,
        "mechanism": (
            "#1773 built its activating-example windows from ANSWER-side activation "
            "(ans_max quantile bins) and excluded features with zero activating windows "
            "before dispatch. Context-only features have zero by construction, so the "
            "labelling corpus has NO coverage of them — the one population that would "
            "most directly test whether context-specific KINDS exist is exactly the "
            "population the interpretability labels cannot speak to."
        ),
    }
    _log(f"label coverage by side: {json.dumps(out)}")
    return out


# ── (b) activity-stratified enrichment ──────────────────────────────────────────────


def _strata(occ: np.ndarray) -> tuple[np.ndarray, list[str]]:
    b = np.digitize(occ, OCC_EDGES) - 1
    lab = [
        f"{OCC_EDGES[i]}-{OCC_EDGES[i + 1] - 1}" if OCC_EDGES[i + 1] < 10**9 else f"{OCC_EDGES[i]}+"
        for i in range(len(OCC_EDGES) - 1)
    ]
    return b, lab


def enrichment(inp: dict, args) -> dict:
    """Crude + activity-standardized enrichment of every judged level, per group.

    Standardization is DIRECT to the group's own occupancy distribution: both the
    group rate and the baseline rate are reweighted by the group's per-stratum
    share, so the two populations are compared matched on occupancy. The null is
    the exact conditional (hypergeometric) draw of the group's per-stratum level
    count from the pooled stratum members."""
    side, occ = inp["side"], inp["occ"]
    bins, bin_labels = _strata(occ)
    base_mask = side == 1
    rng = np.random.default_rng(NULL_SEED)

    # per-stratum support table (reported verbatim so the reader sees the matching)
    support = []
    for i, lb in enumerate(bin_labels):
        row = {
            "occupancy": lb,
            "context_only": int(((side == 0) & (bins == i)).sum()),
            "answer_only": int(((side == 2) & (bins == i)).sum()),
            "two_sided": int((base_mask & (bins == i)).sum()),
        }
        if row["context_only"] or row["answer_only"] or row["two_sided"]:
            support.append(row)
    usable = np.array(
        [
            int((base_mask & (bins == i)).sum()) >= MIN_BASELINE_PER_STRATUM
            for i in range(len(bin_labels))
        ]
    )

    # every (axis, level) as a boolean indicator + a labelled mask
    variables: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for ax in AXES:
        col = inp["labels"]["axes"][ax]
        labelled = col != ""
        for lev in sorted({v for v in col[labelled].tolist()}):
            variables.append((ax, lev, col == lev, labelled))
    gl = inp["gurnee"]
    for code, nm in GURNEE_NAMES.items():
        variables.append(("gurnee_class", nm, gl == code, np.ones(DICT_SIZE, dtype=bool)))

    # A whole (group, axis) cell can be NOT COMPUTABLE — context-only features carry
    # no #1773 label at all. Record that explicitly; a silently absent row would read
    # as "nothing notable there" instead of "the substrate cannot answer".
    not_computable = []
    for gname, gcode in GROUPS.items():
        for ax in AXES:
            n_lab = int(((side == gcode) & (inp["labels"]["axes"][ax] != "")).sum())
            if n_lab == 0:
                not_computable.append(
                    {
                        "group": gname,
                        "axis": ax,
                        "n_group": int((side == gcode).sum()),
                        "n_group_labelled": 0,
                        "reason": (
                            "#1773 assigned no label to ANY feature in this group: its "
                            "activating-example windows are answer-side (ans_max), and a "
                            "context-only feature has zero answer-side windows by "
                            "construction, so it was excluded before dispatch."
                        ),
                    }
                )
    if not_computable:
        _log(
            f"NOT COMPUTABLE (no labels in group): {[(r['group'], r['axis']) for r in not_computable]}"
        )

    rows = []
    for gname, gcode in GROUPS.items():
        gmask_all = side == gcode
        for ax, lev, ind, labelled in variables:
            g = gmask_all & labelled
            b = base_mask & labelled
            crude_g = float(ind[g].mean()) if g.sum() else float("nan")
            crude_b = float(ind[b].mean()) if b.sum() else float("nan")

            gm = g & usable[bins]
            bm = b & usable[bins]
            sidx = np.unique(bins[gm])
            n_g = np.array([int((gm & (bins == s)).sum()) for s in sidx])
            k_g = np.array([int((gm & (bins == s) & ind).sum()) for s in sidx])
            n_b = np.array([int((bm & (bins == s)).sum()) for s in sidx])
            k_b = np.array([int((bm & (bins == s) & ind).sum()) for s in sidx])
            keep = (n_g > 0) & (n_b > 0)
            sidx, n_g, k_g, n_b, k_b = sidx[keep], n_g[keep], k_g[keep], n_b[keep], k_b[keep]
            if n_g.sum() == 0:
                continue
            w = n_g / n_g.sum()  # direct standardization to the GROUP's occupancy profile
            std_g = float((w * (k_g / n_g)).sum())
            std_b = float((w * (k_b / n_b)).sum())

            # exact conditional null: group's per-stratum count ~ Hypergeom(pooled)
            N, K, n = n_g + n_b, k_g + k_b, n_g
            draws = rng.hypergeometric(
                np.maximum(K, 0), np.maximum(N - K, 0), n, size=(args.n_draws, len(n))
            )
            gnull = (w * (draws / n)).sum(1)
            bnull = (w * ((K - draws) / n_b)).sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                lr_null = np.log((gnull + 1e-12) / (bnull + 1e-12))
                lr_obs = float(np.log((std_g + 1e-12) / (std_b + 1e-12)))
            p = float((np.abs(lr_null) >= abs(lr_obs) - 1e-15).sum() + 1) / (args.n_draws + 1)
            lo, hi = np.quantile(gnull, [0.025, 0.975])
            rows.append(
                {
                    "group": gname,
                    "axis": ax,
                    "level": lev,
                    "n_group_labelled": int(g.sum()),
                    "n_group_matched": int(n_g.sum()),
                    "crude_rate_group": crude_g,
                    "crude_rate_two_sided": crude_b,
                    "crude_enrichment": (crude_g / crude_b) if crude_b else float("nan"),
                    "std_rate_group": std_g,
                    "std_rate_two_sided_matched": std_b,
                    "std_enrichment": (std_g / std_b) if std_b else float("nan"),
                    "null_rate_ci95": [float(lo), float(hi)],
                    "p_perm": p,
                }
            )

    # BH-FDR across every test
    ps = np.array([r["p_perm"] for r in rows])
    order = np.argsort(ps)
    q = np.empty_like(ps)
    q[order] = np.minimum.accumulate((ps[order] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
    for r, qq in zip(rows, np.minimum(q, 1.0), strict=True):
        r["q_bh"] = float(qq)
        r["dissolves_into_activity"] = bool(
            np.isfinite(r["crude_enrichment"])
            and (r["crude_enrichment"] >= 1.5 or r["crude_enrichment"] <= 1 / 1.5)
            and qq >= 0.05
        )

    unmatched = {
        g: {
            "n_total": int((side == c).sum()),
            "n_matched": int(((side == c) & usable[bins]).sum()),
            "n_unmatchable_occupancy_1": int(((side == c) & (bins == 0)).sum()),
        }
        for g, c in GROUPS.items()
    }
    for g in unmatched:
        u = unmatched[g]
        u["matched_fraction"] = u["n_matched"] / u["n_total"]
    _log(f"enrichment: {len(rows)} tests; matched coverage {unmatched}")
    return {
        "method": (
            "Direct standardization to the GROUP's occupancy distribution over fixed "
            "integer occupancy strata; exact conditional (hypergeometric) null; "
            f"{args.n_draws:,} draws; BH-FDR across all {len(rows)} tests."
        ),
        "stratifier": (
            "total row occupancy = fit rows active in the CONTEXT span + fit rows active "
            "in the ANSWER span, out of 120,000. This is the only activity measure "
            "defined for BOTH classes: the substrate's `activity` column is answer-side "
            "only and is identically 0 for every context-only feature."
        ),
        "min_baseline_per_stratum": MIN_BASELINE_PER_STRATUM,
        "support_table": support,
        "matched_coverage": unmatched,
        "not_computable": not_computable,
        "tests": rows,
    }


# ── (a) cards ───────────────────────────────────────────────────────────────────────


def build_cards(inp: dict) -> list[dict]:
    side, occ = inp["side"], inp["occ"]
    out = []
    for gname, gcode in GROUPS.items():
        ids = np.nonzero(side == gcode)[0]
        ids = ids[np.argsort(-occ[ids])]
        for fid in ids.tolist():
            rec = {
                "feat_id": fid,
                "side_class": gname,
                "ctx_rows_active": int(inp["psi"][fid]),
                "ans_rows_active": int(inp["cnt"][fid]),
                "occupancy": int(occ[fid]),
                "r2": float(inp["r2"][fid]) if inp["scored"][fid] else None,
                "scored": bool(inp["scored"][fid]),
                "gurnee_class": GURNEE_NAMES.get(int(inp["gurnee"][fid])),
                "description": inp["labels"]["desc"].get(fid) or None,
                "label_status": (
                    "labelled"
                    if inp["labels"]["axes"]["abstraction"][fid]
                    else "no_evidence_zero_answer_side_windows"
                ),
                "axes": {a: (inp["labels"]["axes"][a][fid] or None) for a in AXES},
            }
            if inp["runlen"] is not None:
                ct = inp["runlen"]["ctx_tokens_active"][fid]
                at = inp["runlen"]["ans_tokens_active"][fid]
                rec["ctx_tokens_active_subsample"] = None if not np.isfinite(ct) else float(ct)
                rec["ans_tokens_active_subsample"] = None if not np.isfinite(at) else float(at)
            out.append(rec)
    return out


# ── enrichment forest figure (inlined into the page as base64) ──────────────────────


def forest_png(enr: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style()
    sig = [r for r in enr["tests"] if r["q_bh"] < 0.05 and np.isfinite(r["std_enrichment"])]
    sig.sort(key=lambda r: r["std_enrichment"])
    if not sig:
        sig = sorted(
            (r for r in enr["tests"] if np.isfinite(r["std_enrichment"])),
            key=lambda r: r["q_bh"],
        )[:20]
    fig, ax = plt.subplots(figsize=(8.2, max(3.0, 0.30 * len(sig) + 1.3)))
    colors = {
        "context_only": paper_palette_role("primary"),
        "answer_only": paper_palette_role("baseline"),
    }
    y = np.arange(len(sig))
    for i, r in enumerate(sig):
        c = colors[r["group"]]
        ax.plot([r["crude_enrichment"], r["std_enrichment"]], [i, i], color=c, alpha=0.3, lw=1.0)
        ax.scatter([r["crude_enrichment"]], [i], s=16, facecolors="none", edgecolors=c, lw=1.0)
        ax.scatter([r["std_enrichment"]], [i], s=34, color=c, zorder=3)
    ax.axvline(1.0, color=paper_palette_role("neutral"), ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['axis']}:{r['level']}  [{r['group'].replace('_', '-')}]" for r in sig], fontsize=7.5
    )
    ax.set_xscale("log")
    ax.set_xlabel("enrichment vs two-sided baseline (log scale)")
    # a legend, not a long xlabel: the previous one-line xlabel was clipped at the
    # figure edge exactly where it explained filled-vs-hollow
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=6,
            color=paper_palette_role("baseline"),
            label="activity-standardized (the read)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=6,
            mfc="none",
            mew=1.1,
            color=paper_palette_role("baseline"),
            label="crude (unmatched)",
        ),
    ]
    if any(r["group"] == "context_only" for r in sig):
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=6,
                color=paper_palette_role("primary"),
                label="context-only",
            )
        )
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    ax.set_title(
        "Side-specific feature enrichment, BH-FDR q < 0.05\n"
        "(association with a judged label, not a causal claim)",
        fontsize=9.5,
    )
    ax.set_ylim(-0.8, len(sig) - 0.2)
    fig.set_layout_engine("tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── dashboard ───────────────────────────────────────────────────────────────────────


def _esc(s: object) -> str:
    return html.escape(str(s)) if s not in (None, "") else "&mdash;"


CSS = """
:root { --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --bg:#fbfbfd; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 22px 60px; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:34px 0 6px; padding-top:14px; border-top:1px solid var(--line); }
p, li { color:var(--mut); font-size:13.5px; }
.head { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-bottom:10px; }
.head p { margin:6px 0; } .head b { color:var(--fg); }
.warn { background:#fff7ed; border:1px solid #f6d6bd; border-radius:9px; padding:12px 14px;
  margin:10px 0; } .warn p { color:#8a4a12; margin:5px 0; }
table { border-collapse:collapse; font-size:12.5px; width:100%; background:var(--card); }
th, td { padding:4px 8px; border-bottom:1px solid var(--line); text-align:right;
  font-variant-numeric:tabular-nums; }
th { text-align:right; color:var(--fg); font-weight:600; cursor:pointer;
  position:sticky; top:0; background:#f4f6fa; }
th:first-child, td:first-child, th.l, td.l { text-align:left; }
tr.sig td { background:#f2f8f4; }
tr.dis td { background:#fdf6ec; }
.card { background:var(--card); border:1px solid var(--line); border-radius:9px;
  padding:10px 13px; margin:8px 0; }
.card .top { display:flex; flex-wrap:wrap; align-items:baseline; gap:9px; }
.fid { font-weight:600; font-size:14px; }
.fid a { color:#1b4fd8; text-decoration:none; } .fid a:hover { text-decoration:underline; }
.badge { font-size:11px; padding:1px 7px; border-radius:9px; border:1px solid var(--line);
  color:var(--mut); }
.badge.ctx { background:#eef4ff; border-color:#c9d9fb; color:#1b3f9b; }
.badge.ans { background:#fff1e8; border-color:#f6d0b4; color:#8a4212; }
.nums { color:var(--mut); font-size:12px; font-variant-numeric:tabular-nums; }
.desc { margin:5px 0 0; font-size:12.5px; color:#2c313b; }
.ax { color:var(--mut); font-size:11.5px; margin-top:3px; }
.ctl { margin:10px 0; font-size:13px; color:var(--mut); }
.ctl input, .ctl select { font:inherit; padding:3px 6px; border:1px solid var(--line);
  border-radius:6px; }
img.fig { width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }
"""

JS = """
function sortTable(t, i, num) {
  const tb = t.tBodies[0], rows = Array.from(tb.rows);
  t._dir = t._dir === i ? -i - 1000 : i;
  const asc = t._dir === i;
  rows.sort((a, b) => {
    let x = a.cells[i].dataset.v ?? a.cells[i].textContent;
    let y = b.cells[i].dataset.v ?? b.cells[i].textContent;
    if (num) { x = parseFloat(x); y = parseFloat(y);
      if (isNaN(x)) x = -Infinity; if (isNaN(y)) y = -Infinity;
      return asc ? x - y : y - x; }
    return asc ? String(x).localeCompare(y) : String(y).localeCompare(x);
  });
  rows.forEach(r => tb.appendChild(r));
}
function applyFilter() {
  const cls = document.getElementById('fclass').value;
  const q = document.getElementById('fq').value.toLowerCase();
  const minA = parseFloat(document.getElementById('focc').value) || 0;
  let shown = 0;
  document.querySelectorAll('#cards .card').forEach(c => {
    const okc = cls === 'all' || c.dataset.cls === cls;
    const okq = !q || c.textContent.toLowerCase().includes(q);
    const oka = parseFloat(c.dataset.occ) >= minA;
    const ok = okc && okq && oka;
    c.style.display = ok ? '' : 'none';
    if (ok) shown++;
  });
  document.getElementById('shown').textContent = shown;
}
function sortCards(mode) {
  const box = document.getElementById('cards');
  const cs = Array.from(box.children);
  cs.sort((a, b) => {
    if (mode === 'occ') return parseFloat(b.dataset.occ) - parseFloat(a.dataset.occ);
    if (mode === 'r2') {
      const x = parseFloat(a.dataset.r2), y = parseFloat(b.dataset.r2);
      return (isNaN(y) ? -Infinity : y) - (isNaN(x) ? -Infinity : x);
    }
    return parseFloat(a.dataset.fid) - parseFloat(b.dataset.fid);
  });
  cs.forEach(c => box.appendChild(c));
}
"""


def _enrich_rows_html(enr: dict) -> str:
    rows = sorted(
        enr["tests"], key=lambda r: (r["q_bh"], -abs(np.log(max(r["std_enrichment"], 1e-9))))
    )
    out = []
    for r in rows:
        cls = "sig" if r["q_bh"] < 0.05 else ("dis" if r["dissolves_into_activity"] else "")
        note = ""
        if r["q_bh"] < 0.05:
            note = "enriched" if r["std_enrichment"] > 1 else "depleted"
        elif r["dissolves_into_activity"]:
            note = "dissolves into activity"
        out.append(
            f'<tr class="{cls}">'
            f'<td class="l">{_esc(r["group"].replace("_", "-"))}</td>'
            f'<td class="l">{_esc(r["axis"])}</td><td class="l">{_esc(r["level"])}</td>'
            f'<td data-v="{r["n_group_matched"]}">{r["n_group_matched"]:,}</td>'
            f'<td data-v="{r["std_rate_group"]:.6f}">{100 * r["std_rate_group"]:.1f}%</td>'
            f'<td data-v="{r["std_rate_two_sided_matched"]:.6f}">'
            f"{100 * r['std_rate_two_sided_matched']:.1f}%</td>"
            f'<td data-v="{r["std_enrichment"]:.6f}"><b>{r["std_enrichment"]:.2f}&times;</b></td>'
            f'<td data-v="{r["crude_enrichment"]:.6f}">{r["crude_enrichment"]:.2f}&times;</td>'
            f'<td data-v="{r["q_bh"]:.8f}">{r["q_bh"]:.1e}</td>'
            f'<td class="l">{note}</td></tr>'
        )
    return "".join(out)


def _support_html(enr: dict) -> str:
    return "".join(
        f'<tr><td class="l">{_esc(s["occupancy"])}</td>'
        f"<td>{s['context_only']:,}</td><td>{s['answer_only']:,}</td>"
        f"<td>{s['two_sided']:,}</td>"
        f'<td class="l">{"no baseline — unmatchable" if s["two_sided"] == 0 else ""}</td></tr>'
        for s in enr["support_table"]
    )


def _cards_html(cards: list[dict], runlen_rows: int | None) -> str:
    out = []
    for c in cards:
        badge = "ctx" if c["side_class"] == "context_only" else "ans"
        r2 = f"{c['r2']:.3f}" if c["scored"] else "<i>unscored</i>"
        tok = ""
        if runlen_rows is not None:
            ct, at = c.get("ctx_tokens_active_subsample"), c.get("ans_tokens_active_subsample")
            f = lambda v: "n/a" if v is None else f"{v:,.0f}"  # noqa: E731
            tok = f" &middot; tokens in {runlen_rows:,}-row subsample: ctx {f(ct)} / ans {f(at)}"
        ax = ", ".join(f"{k}: {v}" for k, v in c["axes"].items() if v)
        out.append(
            f'<div class="card" data-cls="{c["side_class"]}" data-occ="{c["occupancy"]}" '
            f'data-r2="{c["r2"] if c["scored"] else ""}" data-fid="{c["feat_id"]}">'
            f'<div class="top"><span class="fid">'
            f'<a href="https://www.neuronpedia.org/qwen2.5-7b-instruct/19-{c["feat_id"]}" '
            f'target="_blank" rel="noopener">{c["feat_id"]}</a></span>'
            f'<span class="badge {badge}">{c["side_class"].replace("_", "-")}</span>'
            f'<span class="badge">{_esc(c["gurnee_class"])}</span>'
            f'<span class="nums">rows active: ctx {c["ctx_rows_active"]:,} / ans '
            f"{c['ans_rows_active']:,} of 120,000 &middot; R&sup2; {r2}{tok}</span></div>"
            f'<p class="desc">{_esc(c["description"])}</p>'
            f'<p class="ax">{_esc(ax) if ax else "&mdash;"}</p></div>'
        )
    return "".join(out)


def _gap_html(lc: dict, enr: dict) -> str:
    """The coverage-gap block. Loud by design: the absence of context-only labels is
    the round's main structural result, not a footnote."""
    nc = enr["not_computable"]
    if not nc:
        return ""
    axes = sorted({r["axis"] for r in nc})
    groups = sorted({r["group"].replace("_", "-") for r in nc})
    ne = lc["no_evidence_exclusion"]
    return f"""<div class="warn" style="background:#fdecec;border-color:#f0c2c2">
<p style="color:#8a1212"><b>STRUCTURAL GAP &mdash; #1773 does not cover context-only
features at all.</b> Axis-label coverage is
{100 * lc["two_sided"]["axis_labelled_fraction"]:.0f}% of two-sided and
{100 * lc["answer_only"]["axis_labelled_fraction"]:.0f}% of answer-only features, but
<b>{lc["context_only"]["n_axis_labelled"]} of {lc["context_only"]["n"]:,}</b>
({100 * lc["context_only"]["axis_labelled_fraction"]:.0f}%) of context-only features.
None of them has a description either.</p>
<p style="color:#8a1212"><b>Why, and why it is not fixable by re-running anything.</b>
#1773 built its activating-example windows from each feature's own <i>answer-side</i>
activation (<code>ans_max</code> quantile bins) and dropped features with zero
activating windows before dispatch. A context-only feature has zero <i>by the very
property that makes it context-only</i>. Its exclusion set is exactly
{{context-only}} &cup; {{dead}} &mdash; {ne["n_excluded"]:,} features, set-equality
verified{"" if ne["equals_context_only_union_dead"] else " (MISMATCH — investigate)"}.</p>
<p style="color:#8a1212"><b>Consequence for this page.</b> The judged-label enrichment is
computable for <b>answer-only features only</b>. For {", ".join(groups)} the axes
{", ".join(axes)} are reported as NOT COMPUTABLE rather than omitted, so an empty row is
never mistaken for a null result. The Gurnee promoting class is derived from the decoder,
not from autointerp, so it IS available for both groups &mdash; it is the only judged
column that spans them. The cards below still carry every measured quantity for
context-only features; only the interpretive text is missing.</p>
<p style="color:#8a1212"><b>So the question "do context-specific kinds exist?" cannot be
answered from the current substrate</b>, and the reason is not sampling noise. Answering
it requires describing context-only features from CONTEXT-side activating windows &mdash;
a labelling run that does not yet exist.</p></div>"""


def render(cards: list[dict], enr: dict, inp: dict, fig_b64: str, lc: dict) -> str:
    mc = enr["matched_coverage"]
    cov = inp["labels"]["coverage"]
    rr = inp["runlen_rows"]
    n_ctx, n_ans = CENSUS_EXPECTED["context_only"], CENSUS_EXPECTED["answer_only"]
    live = n_ctx + n_ans + CENSUS_EXPECTED["two_sided"]
    tok_note = (
        f"Token counts come from the run-length capture, a <b>{rr:,}-row subsample</b> — a "
        f"different, far smaller denominator than the 120,000-row census that defines the "
        f"side classes. They describe depth on the live side and are NOT the evidence for "
        f"one-sidedness."
        if rr is not None
        else "The run-length capture had not landed when this page was built, so token-level "
        "side counts are omitted."
    )
    return f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Issue 1482 &mdash; context-only vs answer-only SAE features</title>
<style>{CSS}</style></head><body><div class='wrap'>
<h1>Context-only vs answer-only SAE features (issue #1482)</h1>

<div class="warn">
<p><b>Enrichments are associations, not mechanisms.</b> Every number below is an
association between a judged label and a side class, measured on observational data.
Nothing here says a feature <i>does</i> anything.</p>
<p><b>Descriptions are a reading aid, not evidence &mdash; and any KIND reading rests
entirely on them.</b> Descriptions and axis labels come from #1773, whose standing caveat
is that they are search-index-only (neighbour discrimination 0.322 against a 0.50 bar).
The enrichment table is measured; the impression that these features are recognisable
kinds, formed from the card text below, is the least evidenced thing on this page.</p>
<p><b>"Never fires on side X" is a strong criterion.</b> It means zero across all
<b>120,000 fit rows</b> at ROW-OCCUPANCY grain &mdash; the feature counts as active if it
fires anywhere in the span. A feature firing at a single token of a single answer is NOT
answer-only. {tok_note}</p>
</div>
{_gap_html(lc, enr)}
<div class="head">
<p><b>Populations.</b> Of {live:,} live features, <b>{n_ctx:,}</b> are context-only
(fire in the context, never in the answer) and <b>{n_ans:,}</b> are answer-only &mdash;
together 2.9% of live features. The remaining {CENSUS_EXPECTED["two_sided"]:,} two-sided
features are the baseline to contrast against, not a population to enumerate. Answer-only
features are the cleanest argument that the map must PRODUCE answer-space structure
rather than copy context-space structure, which is why the question is what they are, not
how many.</p>
<p><b>Why activity stratification is not optional.</b> Median total row occupancy is
<b>2 of 120,000</b> for both side-specific classes versus <b>1,283</b> for two-sided, so a
crude enrichment would largely re-report activity composition. Both columns are shown; a
level whose crude effect vanishes once matched is flagged
<i>dissolves into activity</i>.</p>
<p><b>Unmatchable by construction.</b> A two-sided feature needs at least one row on each
side, so no two-sided feature can have occupancy 1. Occupancy-1 features
({mc["context_only"]["n_unmatchable_occupancy_1"]:,} context-only,
{mc["answer_only"]["n_unmatchable_occupancy_1"]:,} answer-only) therefore have no baseline
and are excluded from the standardized estimate. Matched coverage:
context-only {100 * mc["context_only"]["matched_fraction"]:.1f}%
({mc["context_only"]["n_matched"]:,}/{mc["context_only"]["n_total"]:,}), answer-only
{100 * mc["answer_only"]["matched_fraction"]:.1f}%
({mc["answer_only"]["n_matched"]:,}/{mc["answer_only"]["n_total"]:,}).</p>
<p><b>Method.</b> {_esc(enr["method"])} Stratifier: {_esc(enr["stratifier"])}</p>
<p><b>Label coverage</b> (#1773, of {DICT_SIZE:,}): {_esc(cov)}. Unlabelled features are
excluded from that axis's denominator.</p>
</div>

<h2>(b) Activity-standardized enrichment &mdash; the licensing evidence</h2>
<p>Filled markers are activity-standardized, hollow are crude; the gap between them is the
activity artefact. Green rows are BH-FDR q &lt; 0.05.</p>
<img class="fig" alt="Enrichment forest plot" src="data:image/png;base64,{fig_b64}">
<table id="enr"><thead><tr>
<th class="l" onclick="sortTable(enr,0,0)">group</th>
<th class="l" onclick="sortTable(enr,1,0)">axis</th>
<th class="l" onclick="sortTable(enr,2,0)">level</th>
<th onclick="sortTable(enr,3,1)">n matched</th>
<th onclick="sortTable(enr,4,1)">rate (group)</th>
<th onclick="sortTable(enr,5,1)">rate (two-sided, matched)</th>
<th onclick="sortTable(enr,6,1)">enrichment (std)</th>
<th onclick="sortTable(enr,7,1)">enrichment (crude)</th>
<th onclick="sortTable(enr,8,1)">q (BH)</th>
<th class="l">note</th></tr></thead>
<tbody>{_enrich_rows_html(enr)}</tbody></table>

<h2>Not computable &mdash; reported, not omitted</h2>
<table><thead><tr><th class="l">group</th><th class="l">axis</th><th>n in group</th>
<th>n labelled</th><th class="l">reason</th></tr></thead><tbody>
{
        "".join(
            f'<tr><td class="l">{_esc(r["group"].replace("_", "-"))}</td>'
            f'<td class="l">{_esc(r["axis"])}</td><td>{r["n_group"]:,}</td>'
            f'<td>{r["n_group_labelled"]}</td><td class="l">{_esc(r["reason"])}</td></tr>'
            for r in enr["not_computable"]
        )
        or '<tr><td class="l" colspan="5">none &mdash; every group x axis cell was computable</td></tr>'
    }
</tbody></table>

<h2>Occupancy strata &mdash; the matching support</h2>
<table><thead><tr><th class="l">total rows active (of 120,000)</th><th>context-only</th>
<th>answer-only</th><th>two-sided</th><th class="l"></th></tr></thead>
<tbody>{_support_html(enr)}</tbody></table>

<h2>(a) Feature browser &mdash; {len(cards):,} side-specific features</h2>
<div class="ctl">
class <select id="fclass" onchange="applyFilter()">
<option value="all">all</option><option value="context_only">context-only</option>
<option value="answer_only">answer-only</option></select>
&nbsp; min rows active <input id="focc" type="number" value="0" style="width:80px"
oninput="applyFilter()">
&nbsp; search <input id="fq" type="text" placeholder="description / label"
oninput="applyFilter()" style="width:240px">
&nbsp; sort <select onchange="sortCards(this.value)">
<option value="occ">rows active</option><option value="r2">R&sup2;</option>
<option value="fid">feature id</option></select>
&nbsp; showing <b id="shown">{len(cards):,}</b>
</div>
<div id="cards">{_cards_html(cards, rr)}</div>
<script>{JS}
const enr = document.getElementById('enr');
applyFilter();</script>
</div></body></html>"""


# ── driver ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", type=Path, default=LABELS_DEFAULT)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        print("import-check OK")
        sys.stdout.flush()
        sys.exit(0)

    t0 = time.time()
    inp = load_inputs(args)
    lc = label_coverage_by_side(inp)
    enr = enrichment(inp, args)
    cards = build_cards(inp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        OUT_DIR / "side_specific_enrichment.json",
        {
            "goal": "activity-stratified judged-label enrichment of context-only and "
            "answer-only SAE features against the two-sided baseline (issue #1482)",
            "caveats": CAVEATS,
            "census": CENSUS_EXPECTED,
            "n_fit_rows": N_FIT,
            "r2_source": f"superkaiba1/explore-persona-space-data/{R2_HF_PATH}",
            "r2_scored": SCORED_EXPECTED,
            "runlen_subsample_rows": inp["runlen_rows"],
            "label_coverage": inp["labels"]["coverage"],
            "label_coverage_by_side": lc,
            **enr,
            "provenance": _provenance(),
        },
    )
    _write_json(
        OUT_DIR / "side_specific_features.json",
        {
            "caveats": CAVEATS,
            "n_features": len(cards),
            "runlen_subsample_rows": inp["runlen_rows"],
            "label_coverage_by_side": lc,
            "features": cards,
            "provenance": _provenance(),
        },
    )

    body = render(cards, enr, inp, forest_png(enr), lc)
    for p in (DASH_PATH, PUBLIC_DASH_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        _log(f"wrote {p} ({p.stat().st_size / 1024:.0f} KiB)")
    _log(f"ALL DONE in {time.time() - t0:.0f}s")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
