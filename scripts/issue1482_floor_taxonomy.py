"""Issue #1482 follow-up — floor-adjusted TOPIC taxonomy (0 GPU, CPU-only).

The parent judged taxonomy (``eval_results/issue_1482/taxonomy.json``, 15
contrasts, all BH-significant) was never corrected for the answer-sampling
floor by topic; the k-resample round adjusted only the binary
English/non-English language contrast.

The per-context floors DO exist: the k-resample round persisted them to the HF
data repo at ``issue1482_kresample/analysis_tensors/percontext_floor.npz``
(2000 subsample contexts x {floor_n, nerr_adj, nerr_stored, ...}), so NO
regeneration of the K=4 fresh draws is needed. This driver stages that array,
gates it against the k-resample's own published arm means + deltas (exact
re-derivation, atol 1e-9), joins the banked judge topic labels, and re-runs the
taxonomy contrast battery on the floor-adjusted DV.

Three DVs per contrast, all on the SAME 2000-context subsample so the
raw-vs-adjusted comparison is not confounded by n:

  raw            nerr_stored              (the parent's DV, subsample-restricted)
  adj_registered nerr_adj = m2/denom      (the k-resample PRIMARY estimator:
                                           m2 = ||vhat - mean(V_fresh)||^2 - trvar/K)
  adj_literal    nerr_stored - floor_n    (the brief's literal adj_i form;
                                           also unbiased for m2, noisier)

Contrast helpers (``_boot_group_delta`` joint per-draw resampling,
``_perm_pvals`` batched subset-sum GEMM, ``_bh_fdr``) are REUSED verbatim from
``issue1482_analysis`` -- not reimplemented.

DIGEST-ONLY: this driver touches labels, floors and scalars; no LMSYS/WildChat
prompt or completion text is read, logged, or written.
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

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# BEFORE numpy/torch-bearing imports: load_dotenv() setdefaults the shared-VM
# BLAS/OMP thread caps, and torch (pulled in transitively) freezes its intra-op
# pool from OMP_NUM_THREADS at import (#847).
load_dotenv()

import numpy as np  # noqa: E402

import issue1482_analysis as A  # noqa: E402  (contrast helpers, TOPICS -- reused verbatim)
import issue1482_error_analysis as D  # noqa: E402  (_write_json, SPLIT_SEED_1482)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482_floor_taxonomy")

HF_REPO = "superkaiba1/explore-persona-space-data"
FLOOR_REMOTE = "issue1482_kresample/analysis_tensors/percontext_floor.npz"
PARENT_PERCONTEXT_REMOTE = (
    "issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz"
)
PARENT_JOIN_MED_REL_MAX = 1e-3
BOOT_SEED = 1482  # the k-resample bootstrap seed (adjusted_contrast.json seeds.bootstrap)
N_DRAWS = 10000
FDR_Q = 0.05

# Exact re-derivation targets from the k-resample round's own published artifacts
# (eval_results/issue_1482/kresample/{floor_summary,adjusted_contrast}.json). These
# are means over the SAME stored array, so the gate is an identity check, not a
# tolerance-tuned comparison.
GATE_TARGETS = {
    "en/floor_n": 0.09353805440373048,
    "nonen/floor_n": 0.11016489387561568,
    "en/nerr_adj": 0.20058426818760722,
    "nonen/nerr_adj": 0.16860618437196853,
    "en/nerr_stored": 0.28504909383705607,
    "nonen/nerr_stored": 0.27288093597527757,
    "delta/nerr_adj": -0.031978083815638686,  # adjusted_contrast.s1_self_contained
    "delta/nerr_stored": -0.012168157861778495,  # adjusted_contrast.c1_coherence
    "delta/floor_n": 0.016626839471885202,  # adjusted_contrast.delta_floor.point
}
GATE_ATOL = 1e-9

DVS = ("raw", "adj_registered", "adj_literal")
DV_DESC = {
    "raw": "nerr_stored (parent DV, subsample-restricted)",
    "adj_registered": "nerr_adj = m2/denom (k-resample PRIMARY fresh-4 estimator)",
    "adj_literal": "nerr_stored - floor_n (brief's literal adj_i)",
}


def _stage_floor_npz(dest: Path) -> Path:
    """Stage the per-context floor array from HF (idempotent, fail-loud)."""
    if dest.exists():
        logger.info("[floor-tax] floors already staged: %s", dest)
        return dest
    from explore_persona_space.orchestrate import hub

    dest.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(HF_REPO, FLOOR_REMOTE, dest, repo_type="dataset")
    logger.info("[floor-tax] staged %s -> %s", FLOOR_REMOTE, dest)
    return dest


def _parent_join_gate(rows: np.ndarray, nerr_stored: np.ndarray, dest: Path) -> dict:
    """Row-wise join gate against the PARENT per-context ridge npz.

    The identity gate below re-derives the k-resample's own published aggregates;
    this one is orthogonal — it checks that the subsample's ``nerr_stored`` still
    matches the parent fit's ``holdout_nerr`` ROW BY ROW, i.e. that the join key
    itself is sound. Median relative deviation must sit under 1e-3.
    """
    if not dest.exists():
        from explore_persona_space.orchestrate import hub

        dest.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(HF_REPO, PARENT_PERCONTEXT_REMOTE, dest, repo_type="dataset")
    z = np.load(dest)
    pos = {int(r): i for i, r in enumerate(z["holdout_rows"])}
    missing = [int(r) for r in rows if int(r) not in pos]
    if missing:
        raise RuntimeError(f"[floor-tax] {len(missing)} subsample rows absent from the parent npz")
    idx = np.asarray([pos[int(r)] for r in rows], dtype=np.int64)
    parent = z["holdout_nerr"].astype(np.float64)[idx]
    rel = np.abs(nerr_stored - parent) / np.maximum(np.abs(parent), 1e-12)
    med, mx = float(np.median(rel)), float(rel.max())
    if med >= PARENT_JOIN_MED_REL_MAX:
        raise RuntimeError(
            f"[floor-tax] parent-join gate FAILED: median rel dev {med:.3e} "
            f">= {PARENT_JOIN_MED_REL_MAX}"
        )
    logger.info("[floor-tax] parent-join gate PASS: median rel dev %.3e (max %.3e)", med, mx)
    return {
        "source": f"HF {HF_REPO}/{PARENT_PERCONTEXT_REMOTE}",
        "n_joined": int(len(idx)),
        "median_rel_dev": med,
        "max_rel_dev": mx,
        "threshold_median_rel": PARENT_JOIN_MED_REL_MAX,
        "pass": True,
    }


def _identity_gate(dv: dict[str, np.ndarray], arm: np.ndarray) -> dict:
    """Reproduce the k-resample's published per-arm means + EN/non-EN deltas from
    the staged array. Any drift (wrong file, wrong estimator, re-generated inputs)
    fails LOUD here rather than silently shifting every downstream contrast."""
    en, ne = arm == "en", arm == "nonen"
    assert en.sum() == 1000 and ne.sum() == 1000, (int(en.sum()), int(ne.sum()))
    got: dict[str, float] = {}
    for name, vals in dv.items():
        got[f"en/{name}"] = float(vals[en].mean())
        got[f"nonen/{name}"] = float(vals[ne].mean())
        got[f"delta/{name}"] = float(vals[ne].mean() - vals[en].mean())
    bad = {
        k: (got[k], want, abs(got[k] - want))
        for k, want in GATE_TARGETS.items()
        if abs(got[k] - want) > GATE_ATOL
    }
    if bad:
        raise RuntimeError(f"[floor-tax] identity gate FAILED (atol {GATE_ATOL}): {bad}")
    logger.info(
        "[floor-tax] identity gate PASS: %d targets within %g", len(GATE_TARGETS), GATE_ATOL
    )
    return {
        "checked": {k: got[k] for k in GATE_TARGETS},
        "targets": GATE_TARGETS,
        "atol": GATE_ATOL,
        "pass": True,
    }


def _build_contrasts(labels: dict, ci: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Topic / refusal-adjacency / answer-refusal / format group-vs-rest masks over
    the subsample, mirroring issue1482_analysis's exploratory contrast set."""
    key = [str(int(c)) for c in ci]
    missing = [k for k in key if k not in labels]
    if missing:
        raise RuntimeError(f"[floor-tax] {len(missing)} subsample ci absent from judge labels")

    def field(name: str) -> np.ndarray:
        return np.array([labels[k].get(name, "") for k in key])

    topic, refadj = field("topic"), field("request_refusal_adjacent")
    ansref, fmt = field("answer_is_refusal"), field("format")
    out: list[tuple[str, np.ndarray]] = [(f"topic:{t}", topic == t) for t in A.TOPICS]
    out.append(("refusal_adjacent:yes+borderline", np.isin(refadj, ["yes", "borderline"])))
    out.append(("answer_is_refusal:yes+partial", np.isin(ansref, ["yes", "partial"])))
    out += [(f"format:{f}", fmt == f) for f in sorted(A.FIELDS["format"])]
    return out


def _one_dv(vals: np.ndarray, contrasts: list[tuple[str, np.ndarray]], n_draws: int) -> list[dict]:
    """Point delta + joint-resample bootstrap CI + permutation p (BH-FDR) per contrast."""
    masks = [m for _, m in contrasts]
    pvals = A._perm_pvals(vals, masks, n_draws, D.SPLIT_SEED_1482)
    sig = A._bh_fdr(pvals, q=FDR_Q)
    rows: list[dict] = []
    for (name, m), p, s in zip(contrasts, pvals, sig, strict=True):
        n_in = int(m.sum())
        if n_in == 0 or n_in == len(m):
            rows.append({"contrast": name, "n_in": n_in, "note": "degenerate group"})
            continue
        # contexts resampled JOINTLY per draw: one index draw applied to values AND
        # both masks (issue1482_analysis._boot_group_delta).
        draws = A._boot_group_delta(vals, m, ~m, n_draws, BOOT_SEED)
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        delta = float(vals[m].mean() - vals[~m].mean())
        rows.append(
            {
                "contrast": name,
                "n_in": n_in,
                "n_out": int((~m).sum()),
                "mean_in": float(vals[m].mean()),
                "mean_out": float(vals[~m].mean()),
                "delta": delta,
                "ci95": [lo, hi],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "perm_p": p,
                "bh_fdr_sig_q05": bool(s),
                "n_boot_finite": int(draws.size),
            }
        )
    return rows


def _figure(out_fig: Path, per_dv: dict[str, list[dict]], parent: dict[str, dict]) -> None:
    """Raw vs floor-adjusted delta per contrast, CI whiskers, ordered by raw delta."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style()
    raw = {r["contrast"]: r for r in per_dv["raw"] if "delta" in r}
    adj = {r["contrast"]: r for r in per_dv["adj_registered"] if "delta" in r}
    names = sorted(set(raw) & set(adj), key=lambda c: raw[c]["delta"])
    y = np.arange(len(names))
    pal = pp.paper_palette(2)
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(names) + 1.9), layout="constrained")
    for off, (src, lab, col) in enumerate(
        ((raw, "raw (nerr_stored)", pal[0]), (adj, "floor-adjusted (nerr_adj)", pal[1]))
    ):
        pts = np.array([src[c]["delta"] for c in names])
        lo = np.array([src[c]["ci95"][0] for c in names])
        hi = np.array([src[c]["ci95"][1] for c in names])
        err = np.vstack([np.maximum(0.0, pts - lo), np.maximum(0.0, hi - pts)])
        ax.errorbar(
            pts,
            y + (0.18 if off else -0.18),
            xerr=err,
            fmt="o",
            ms=4,
            lw=1.2,
            capsize=2,
            color=col,
            label=lab,
        )
    ax.axvline(0.0, color="0.35", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{c}  (n={raw[c]['n_in']}{'*' if parent.get(c, {}).get('bh_fdr_sig_q05') else ''})"
            for c in names
        ],
        fontsize=7,
    )
    ax.set_xlabel("group - rest mean normalized error (2000-context subsample)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title(
        "#1482 taxonomy contrasts: raw vs answer-sampling-floor-adjusted\n"
        "* = BH-significant in the parent full-n raw taxonomy",
        fontsize=9,
    )
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    logger.info("[floor-tax] figure -> %s", out_fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--floor-npz", type=Path, default=None)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    args = ap.parse_args()

    out_dir = args.out_dir or (
        PROJECT_ROOT / "eval_results" / "issue_1482" / "floor_adjusted_taxonomy"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    floor_npz = args.floor_npz or (
        PROJECT_ROOT / "data" / "issue_1482" / "kresample" / "percontext_floor.npz"
    )
    _stage_floor_npz(floor_npz)

    z = np.load(floor_npz)
    ci = z["ci"].astype(np.int64)
    arm = z["arm"].astype(str)
    dv = {
        "nerr_stored": z["nerr_stored"].astype(np.float64),
        "nerr_adj": z["nerr_adj"].astype(np.float64),
        "floor_n": z["floor_n"].astype(np.float64),
    }
    gate = _identity_gate(dv, arm)
    join_gate = _parent_join_gate(
        z["rows"].astype(np.int64),
        dv["nerr_stored"],
        PROJECT_ROOT / "data" / "issue_1482" / "percontext" / "refit_holdout__ridge__seed0.npz",
    )

    labels = json.loads(
        (PROJECT_ROOT / "eval_results" / "issue_1482" / "judge_labels" / "labels.json").read_text()
    )["labels"]
    lang_npz = z["language"].astype(str)
    lang_lab = np.array([labels[str(int(c))].get("language", "") for c in ci])
    n_lang_mismatch = int((lang_npz != lang_lab).sum())
    if n_lang_mismatch:
        raise RuntimeError(f"[floor-tax] language drift npz-vs-labels on {n_lang_mismatch} rows")

    values = {
        "raw": dv["nerr_stored"],
        "adj_registered": dv["nerr_adj"],
        "adj_literal": dv["nerr_stored"] - dv["floor_n"],
    }
    contrasts = _build_contrasts(labels, ci)
    logger.info("[floor-tax] %d contrasts x %d DVs, n=%d", len(contrasts), len(DVS), len(ci))

    per_dv: dict[str, list[dict]] = {}
    for name in DVS:
        logger.info("[floor-tax] DV %s ...", name)
        per_dv[name] = _one_dv(values[name], contrasts, args.n_draws)

    parent_tax = json.loads(
        (PROJECT_ROOT / "eval_results" / "issue_1482" / "taxonomy.json").read_text()
    )
    parent = {c["contrast"]: c for c in parent_tax["contrasts"]}

    # survival / flip bookkeeping: subsample-raw -> subsample-floor-adjusted
    raw_by = {r["contrast"]: r for r in per_dv["raw"]}
    adj_by = {r["contrast"]: r for r in per_dv["adj_registered"]}
    lit_by = {r["contrast"]: r for r in per_dv["adj_literal"]}
    survival = []
    for name, _m in contrasts:
        r, a, li = raw_by[name], adj_by[name], lit_by.get(name, {})
        if "delta" not in r or "delta" not in a:
            continue
        survival.append(
            {
                "contrast": name,
                "n_in": r["n_in"],
                "parent_full_n": parent.get(name, {}).get("n"),
                "parent_full_sig": parent.get(name, {}).get("bh_fdr_sig_q05"),
                "raw_delta": r["delta"],
                "raw_sig": r["bh_fdr_sig_q05"],
                "adj_delta": a["delta"],
                "adj_sig": a["bh_fdr_sig_q05"],
                "adj_literal_delta": li.get("delta"),
                "adj_literal_sig": li.get("bh_fdr_sig_q05"),
                "floor_delta": float(dv["floor_n"][_m].mean() - dv["floor_n"][~_m].mean()),
                "sign_flip_raw_to_adj": bool(
                    np.sign(r["delta"]) != np.sign(a["delta"]) and r["delta"] != 0.0
                ),
                "survives_adjustment": bool(r["bh_fdr_sig_q05"] and a["bh_fdr_sig_q05"]),
                "lost_to_adjustment": bool(r["bh_fdr_sig_q05"] and not a["bh_fdr_sig_q05"]),
                "gained_by_adjustment": bool((not r["bh_fdr_sig_q05"]) and a["bh_fdr_sig_q05"]),
            }
        )

    doc = {
        "what": "floor-adjusted taxonomy contrasts on the 2000-context k-resample subsample",
        "dv_definitions": DV_DESC,
        "n_contexts": int(len(ci)),
        "n_draws": int(args.n_draws),
        "bootstrap_seed": BOOT_SEED,
        "perm_seed": int(D.SPLIT_SEED_1482),
        "fdr_q": FDR_Q,
        "resampling": "contexts resampled JOINTLY per bootstrap draw "
        "(one index draw applied to values and both masks)",
        "identity_gate": gate,
        "parent_join_gate": join_gate,
        "language_join_mismatch": n_lang_mismatch,
        "floor_source": f"HF {HF_REPO}/{FLOOR_REMOTE} (k-resample round; NOT regenerated)",
        "survival": survival,
        "per_dv": per_dv,
        "parent_taxonomy_note": "parent contrasts ran on the FULL holdout (n_labeled "
        f"{parent_tax['contrasts'][0]['n'] if parent_tax.get('contrasts') else '?'} in group 0); "
        "the subsample here is 2000 contexts (1000 en + 1000 non-en by construction), so "
        "power is much lower and the en/non-en mix is NOT the corpus mix",
    }
    D._write_json(out_dir / "floor_adjusted_taxonomy.json", doc)
    logger.info("[floor-tax] wrote %s", out_dir / "floor_adjusted_taxonomy.json")

    _figure(PROJECT_ROOT / "figures" / "issue_1482" / "floor_adjusted_taxonomy.png", per_dv, parent)

    n_raw_sig = sum(1 for s in survival if s["raw_sig"])
    n_surv = sum(1 for s in survival if s["survives_adjustment"])
    n_flip = sum(1 for s in survival if s["sign_flip_raw_to_adj"])
    logger.info(
        "[floor-tax] subsample-raw significant %d/%d; survive adjustment %d; sign flips %d",
        n_raw_sig,
        len(survival),
        n_surv,
        n_flip,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
