"""CSLS hub-correction follow-up for task #2202 (proposal P3, screened not-redundant).

Quantifies how much of the 0.816 -> 0.9425 fresh-draw acc@1 gap the CSLS hub
correction (Conneau et al. 2018, arXiv 1710.04087) closes on the exact #1738
context-arm held-out tensors (n = n_pool = 9,941; layer-19 ridge predictions).

Reuse (no re-implementation):
- ``csls_scores`` + ``K_CSLS`` imported from ``scripts/issue1901_metric_battery.py``
  (cross-domain CSLS, K=10; both neighborhoods from the query x pool matrix).
- ``analysis/mapping_baselines.knn_retrieval`` for the euclidean + cosine
  baseline legs (reconciled against the banked r4 values in
  ``eval_results/issue_2202/repro_gate.json`` at the r4 row tolerances).
- Staging + ci/fingerprint identity asserts mirror
  ``scripts/issue2202_failchar.py`` (``stage_inputs`` / ``load_pred_y``).

CSLS convention corrected (follows #1901): CSLS is COSINE-native — the
similarity matrix is the cosine similarity S_cos = 1 - d_cos between the ridge
predictions and the held-out answer states; ``csls_scores(S_cos, k=10)``
penalizes hub pool columns by their mean top-k column similarity. Retrieval
under CSLS ranks by descending corrected score (distance = -score).

Outputs (all under the issue-2202 worktree):
- ``eval_results/issue_2202/csls_followup.json`` — baseline/CSLS acc@{1,5,10} +
  MRR, gap-closure fractions, FAIL-1 recovery counts, top-10 hub-capture change.
- ``eval_results/issue_2202/csls_percontext_ranks.npz`` — per-context mid-ranks
  (euclidean / cosine / CSLS) + per-pool-row top-10 in-degrees, keyed by ci.
- ``figures/issue_2202/fig_csls_gap.png`` — summary bars + FAIL-1 rank-change
  scatter.

Run (VM, thread-capped):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2202_csls_followup.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps BEFORE numpy (shared-VM rule, #847)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script imports
from issue1901_metric_battery import K_CSLS, csls_scores  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    _pairwise_dist,
    knn_retrieval,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue2202_csls_followup")

REPO_ROOT = Path(__file__).resolve().parents[1]
ISSUE = 2202
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PIN = "09788eef2f85330c6f9c6b7cd3d28cb47cfb8429"  # data-repo revision (#2202 plan §10 pin)
PARENT_PREFIX = "issue1738_multiturn"
LAYER = 19
EXPECTED_N = 9_941  # pinned holdout n (gate-asserted, matches issue2202_failchar)
KS = (1, 5, 10)
# Brief-pinned fresh-draw acc@1 reference (the #2202 0.816 -> 0.9425 gap target).
FRESH_DRAW_REF_ACC1 = 0.9425
# r4 banked-value tolerances (issue2202_failchar._gate_compare conventions).
ACC_TOL_ROWS = 2
MRR_TOL = 1e-4
# argpartition boundary ties are CPU-SIMD-kernel dependent (#1946) — the banked
# hubness.json was computed on a different machine, so reconcile with a small
# row tolerance, never a recompute-equality assert.
HUB_RECONCILE_TOL = 2

BANKED_REPRO = REPO_ROOT / "eval_results" / "issue_2202" / "repro_gate.json"
BANKED_HUBNESS = REPO_ROOT / "eval_results" / "issue_2202" / "hubness.json"
OUT_JSON = REPO_ROOT / "eval_results" / "issue_2202" / "csls_followup.json"
OUT_NPZ = REPO_ROOT / "eval_results" / "issue_2202" / "csls_percontext_ranks.npz"
OUT_FIG = REPO_ROOT / "figures" / "issue_2202" / "fig_csls_gap.png"


def stage_inputs(staged: Path) -> None:
    """Stage the two banked fp16 npz inputs at the pinned data-repo revision."""
    staged.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        HF_DATA_REPO,
        f"{PARENT_PREFIX}/analysis_tensors/pred16/context_L{LAYER}_ridge.npz",
        staged / "pred16.npz",
        revision=HF_PIN,
    )
    hub.stage_hub_file(
        HF_DATA_REPO,
        f"{PARENT_PREFIX}/analysis_tensors/y_holdout/L{LAYER}.npz",
        staged / "y_holdout_L19.npz",
        revision=HF_PIN,
    )


def load_pred_y(staged: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pred fp64, y fp64, ci int64) with the ci/fingerprint identity asserts.

    Mirrors ``issue2202_failchar.load_pred_y`` (same keys, same gates).
    """
    pd_ = np.load(staged / "pred16.npz")
    yd = np.load(staged / "y_holdout_L19.npz")
    pred, pci = pd_["pred16"].astype(np.float64), np.asarray(pd_["ci"], dtype=np.int64)
    y16, yci = yd["y16"].astype(np.float64), np.asarray(yd["ci"], dtype=np.int64)
    assert pred.shape == y16.shape, (pred.shape, y16.shape)
    assert (pci == yci).all(), "pred16/y_holdout ci misalign"
    assert np.array_equal(pd_["fingerprint"], yd["fingerprint"]), (
        "pred16/y_holdout assembly fingerprint mismatch — different capture generations"
    )
    assert len(pci) == EXPECTED_N, f"holdout n {len(pci)} != {EXPECTED_N}"
    return pred, y16, pci


def midranks_true(d: np.ndarray) -> np.ndarray:
    """Mid-rank of the true target (pool row i for query i) within row i of d.

    Verbatim the tolerance-based mid-rank convention of
    ``analysis/mapping_baselines.knn_retrieval`` (1 + #closer + 0.5*#tied-others),
    for the pool == true / true_pool_idx == arange(n) case.
    """
    n = d.shape[0]
    d_true = d[np.arange(n), np.arange(n)]
    tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
    closer = (d < d_true[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - d_true[:, None]) <= tol).sum(axis=1) - 1
    return 1.0 + closer + 0.5 * tied


def ranks_summary(ranks: np.ndarray, n_pool: int) -> dict:
    """acc@k / chance / median rank / MRR (the issue1901 ``_ranks_summary`` shape)."""
    return {
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in KS},
        "chance_at_k": {int(k): float(k / n_pool) for k in KS},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "n": int(ranks.shape[0]),
        "n_pool": int(n_pool),
    }


def n10_counts(d: np.ndarray) -> np.ndarray:
    """Per-pool-row top-10 in-degree N_10 (the issue2202_failchar retrieval-hubness
    convention: 10 smallest distances per query row via argpartition, self included)."""
    n = d.shape[0]
    kk = min(10, n - 1)
    counts = np.zeros(n, dtype=np.int64)
    top10 = np.argpartition(d, kk, axis=1)[:, :kk]
    np.add.at(counts, top10.ravel(), 1)
    return counts


def skewness(x: np.ndarray) -> float:
    """Population skewness (the issue1901 ``_skew`` formula, via issue2202_failchar)."""
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std()
    return float(((x - m) ** 3).mean() / (s**3 + 1e-30))


def reconcile_baseline(rec: dict, banked: dict, label: str) -> dict:
    """Compare a knn_retrieval record against the banked r4 cell; fail loud."""
    acc_tol = ACC_TOL_ROWS / rec["n"] + 1e-12
    deltas: dict = {"acc_at_k": {}, "mrr": None}
    ok = rec["n"] == banked["n"] and rec["n_pool"] == banked["n_pool"]
    for k in KS:
        d = abs(rec["acc_at_k"][int(k)] - banked["acc_at_k"][str(k)])
        deltas["acc_at_k"][str(k)] = d
        ok = ok and d <= acc_tol
    dm = abs(rec["mrr"] - banked["mrr"])
    deltas["mrr"] = dm
    ok = ok and dm <= MRR_TOL
    if not ok:
        raise RuntimeError(f"baseline reconciliation FAILED for {label}: deltas={deltas}")
    logger.info("[reconcile] %s baseline matches banked (deltas=%s)", label, deltas)
    return {"deltas": deltas, "ok": True}


def hub_stats(counts: np.ndarray, pci: np.ndarray) -> dict:
    """Compact hubness read for one metric (max, top-hub ci, skewness, zero-frac)."""
    j = int(np.lexsort((np.arange(len(counts)), -counts))[0])
    return {
        "n10_max": int(counts.max()),
        "top_hub_ci": int(pci[j]),
        "n10_skewness": skewness(counts),
        "n10_frac_zero": float((counts == 0).mean()),
    }


def render_figure(
    base_e: dict, base_c: dict, csls_sum: dict, ranks_e: np.ndarray, ranks_csls: np.ndarray
) -> None:
    """Summary bars (acc@1 + MRR, base eucl/cos vs CSLS, fresh-draw reference)
    plus the per-unit FAIL-1 rank-change scatter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    pal = paper_palette(4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = ["euclidean\n(base)", "cosine\n(base)", "CSLS\n(K=10)"]
    acc1 = [base_e["acc_at_k"][1], base_c["acc_at_k"][1], csls_sum["acc_at_k"][1]]
    mrr = [base_e["mrr"], base_c["mrr"], csls_sum["mrr"]]
    x = np.arange(3, dtype=float)
    w = 0.38
    ax1.bar(x - w / 2, acc1, width=w, color=pal[0], label="acc@1")
    ax1.bar(x + w / 2, mrr, width=w, color=pal[1], label="MRR")
    ax1.hlines(
        FRESH_DRAW_REF_ACC1,
        x[0] - w,
        x[-1] + w,
        color=pal[3],
        ls="--",
        lw=1.5,
        label=f"fresh-draw acc@1 reference ({FRESH_DRAW_REF_ACC1})",
    )
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0.78, 1.0)
    ax1.set_ylabel("retrieval score")
    ax1.set_title(f"Held-out retrieval, n = n_pool = {EXPECTED_N:,}")
    ax1.legend(loc="upper left", frameon=False)

    fail = ranks_e > 1.0
    re_f, rc_f = ranks_e[fail], ranks_csls[fail]
    rec_mask = rc_f <= 1.0
    ax2.scatter(
        re_f[~rec_mask],
        np.maximum(rc_f[~rec_mask], 1.0),
        s=6,
        alpha=0.35,
        color=pal[1],
        edgecolors="none",
        label=f"still failing under CSLS (n={int((~rec_mask).sum()):,})",
    )
    ax2.scatter(
        re_f[rec_mask],
        np.maximum(rc_f[rec_mask], 1.0),
        s=8,
        alpha=0.6,
        color=pal[2],
        edgecolors="none",
        label=f"recovered to rank 1 (n={int(rec_mask.sum()):,})",
    )
    lim = max(re_f.max(), rc_f.max()) * 1.3
    ax2.plot([1, lim], [1, lim], color="0.5", lw=1.0, ls=":")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(0.9, lim)
    ax2.set_ylim(0.9, lim)
    ax2.set_xlabel("euclidean rank of true target (FAIL-1 rows)")
    ax2.set_ylabel("CSLS rank of true target")
    ax2.set_title(f"Per-context rank change, {int(fail.sum()):,} euclidean rank-1 failures")
    ax2.legend(loc="lower right", frameon=False)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("[figs] wrote %s", OUT_FIG)


def main(argv: list[str] | None = None) -> int:
    """Run the full CSLS follow-up analysis end to end (single pass, ~minutes)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--staged-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_2202" / "csls_staged",
        help="local staging dir for the two pinned npz inputs",
    )
    args = ap.parse_args(argv)
    t0 = time.time()

    logger.info("[phase=stage] staging inputs at %s (pin %s)", args.staged_dir, HF_PIN[:12])
    stage_inputs(args.staged_dir)
    pred, y16, pci = load_pred_y(args.staged_dir)
    n = len(pci)

    # ── baseline legs (reuse knn_retrieval; reconcile vs banked r4 values) ──
    logger.info("[phase=baseline] knn_retrieval euclidean + cosine")
    base_e = knn_retrieval(pred, y16, ks=KS, metric="euclidean")
    base_c = knn_retrieval(pred, y16, ks=KS, metric="cosine")
    banked = json.loads(BANKED_REPRO.read_text())["metrics"]
    rec_e = reconcile_baseline(base_e, banked["euclidean"]["banked"], "euclidean")
    rec_c = reconcile_baseline(base_c, banked["cosine"]["banked"], "cosine")

    # per-context mid-ranks + hubness, euclidean (squared-euclid GEMM, rank-invariant)
    d_e = _pairwise_dist(pred, y16, "euclidean")
    ranks_e = midranks_true(d_e)
    counts_e = n10_counts(d_e)
    del d_e
    assert ranks_summary(ranks_e, n)["acc_at_k"] == base_e["acc_at_k"], (
        "euclidean rank vector diverges from knn_retrieval summary"
    )

    # cosine leg + the shared cosine-similarity matrix for CSLS
    d_c = _pairwise_dist(pred, y16, "cosine")
    ranks_c = midranks_true(d_c)
    counts_c = n10_counts(d_c)
    assert ranks_summary(ranks_c, n)["acc_at_k"] == base_c["acc_at_k"], (
        "cosine rank vector diverges from knn_retrieval summary"
    )
    s_cos = 1.0 - d_c
    del d_c

    # ── CSLS leg (reused issue1901 csls_scores, K=10, cosine-native) ──
    logger.info("[phase=csls] csls_scores K=%d on the %dx%d cosine-sim matrix", K_CSLS, n, n)
    d_csls = csls_scores(s_cos, K_CSLS)
    del s_cos
    np.negative(d_csls, out=d_csls)  # retrieval distance = -CSLS score
    ranks_csls = midranks_true(d_csls)
    counts_csls = n10_counts(d_csls)
    del d_csls
    csls_sum = ranks_summary(ranks_csls, n)

    # ── gap closure ──
    def closure(base_acc1: float) -> dict:
        csls_acc1 = csls_sum["acc_at_k"][1]
        return {
            "base_acc1": base_acc1,
            "csls_acc1": csls_acc1,
            "fresh_draw_ref_acc1": FRESH_DRAW_REF_ACC1,
            "fraction": (csls_acc1 - base_acc1) / (FRESH_DRAW_REF_ACC1 - base_acc1),
        }

    gap = {
        "formula": "(csls_acc1 - base_acc1) / (fresh_draw_ref_acc1 - base_acc1)",
        "euclidean_primary": closure(base_e["acc_at_k"][1]),
        "cosine": closure(base_c["acc_at_k"][1]),
    }

    # ── FAIL-1 recovery (hit@1 = mid-rank <= 1) ──
    def fail1_block(ranks_base: np.ndarray, label: str) -> dict:
        fail_b = ranks_base > 1.0
        fail_x = ranks_csls > 1.0
        blk = {
            "n_fail_base": int(fail_b.sum()),
            "recovered_rank1_under_csls": int((fail_b & ~fail_x).sum()),
            "new_failures_under_csls": int((~fail_b & fail_x).sum()),
            "n_fail_csls": int(fail_x.sum()),
        }
        logger.info("[fail1] vs %s: %s", label, blk)
        return blk

    fail1 = {
        "vs_euclidean_primary": fail1_block(ranks_e, "euclidean"),
        "vs_cosine": fail1_block(ranks_c, "cosine"),
    }
    expected_fail_e = round(n * (1.0 - banked["euclidean"]["banked"]["acc_at_k"]["1"]))
    assert fail1["vs_euclidean_primary"]["n_fail_base"] == expected_fail_e, (
        fail1["vs_euclidean_primary"]["n_fail_base"],
        expected_fail_e,
    )

    # ── hub capture (top-10 in-degree; reconcile the euclidean before vs banked) ──
    hub_banked = json.loads(BANKED_HUBNESS.read_text())["retrieval"]
    banked_top = hub_banked["top20"][0]  # {"ci": 22248, "count": 182}
    j_banked = int(np.nonzero(pci == banked_top["ci"])[0][0])
    delta = abs(int(counts_e[j_banked]) - int(banked_top["count"]))
    if delta > HUB_RECONCILE_TOL:
        raise RuntimeError(
            f"euclidean top-hub N_10 reconcile failed: recomputed {int(counts_e[j_banked])} "
            f"vs banked {banked_top['count']} for ci {banked_top['ci']} (tol {HUB_RECONCILE_TOL})"
        )
    hubness = {
        "euclidean": hub_stats(counts_e, pci)
        | {
            "banked_reconcile": {
                "banked_top_hub_ci": int(banked_top["ci"]),
                "banked_top_hub_count": int(banked_top["count"]),
                "recomputed_count": int(counts_e[j_banked]),
                "abs_delta": delta,
                "tol_rows": HUB_RECONCILE_TOL,
                "ok": True,
            }
        },
        "cosine": hub_stats(counts_c, pci),
        "csls": hub_stats(counts_csls, pci)
        | {"banked_euclidean_top_hub_count_after": int(counts_csls[j_banked])},
    }
    logger.info("[hubness] %s", json.dumps({k: v for k, v in hubness.items()}, default=str))

    # ── persist ──
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "issue": ISSUE,
            "generated_utc": datetime.now(UTC).isoformat(),
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "elapsed_s": round(time.time() - t0, 1),
            "inputs": {
                "repo": HF_DATA_REPO,
                "revision": HF_PIN,
                "paths": [
                    f"{PARENT_PREFIX}/analysis_tensors/pred16/context_L{LAYER}_ridge.npz",
                    f"{PARENT_PREFIX}/analysis_tensors/y_holdout/L{LAYER}.npz",
                ],
            },
            **as_metadata_dict(git_provenance(REPO_ROOT)),
        },
        "config": {
            "k_csls": int(K_CSLS),
            "ks": list(KS),
            "n": n,
            "n_pool": n,
            "fresh_draw_ref_acc1": FRESH_DRAW_REF_ACC1,
            "csls_convention": (
                "cosine-native: S = 1 - cosine_distance(pred, y); "
                "csls_scores from issue1901_metric_battery (arXiv 1710.04087); "
                "retrieval distance = -score; mid-rank ties per knn_retrieval"
            ),
        },
        "baseline": {"euclidean": base_e, "cosine": base_c},
        "baseline_reconciliation": {"euclidean": rec_e, "cosine": rec_c},
        "csls": csls_sum,
        "gap_closure": gap,
        "fail1": fail1,
        "hubness_top10": hubness,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("[out] wrote %s", OUT_JSON)

    np.savez_compressed(
        OUT_NPZ,
        ci=pci,
        rank_euclidean=ranks_e.astype(np.float32),
        rank_cosine=ranks_c.astype(np.float32),
        rank_csls=ranks_csls.astype(np.float32),
        n10_euclidean=counts_e.astype(np.int32),
        n10_cosine=counts_c.astype(np.int32),
        n10_csls=counts_csls.astype(np.int32),
    )
    logger.info("[out] wrote %s (%.0f KB)", OUT_NPZ, OUT_NPZ.stat().st_size / 1024)

    render_figure(base_e, base_c, csls_sum, ranks_e, ranks_csls)

    logger.info(
        "[done] base acc@1 eucl=%.4f cos=%.4f -> CSLS=%.4f (ref %.4f); "
        "gap closed eucl=%.1f%% cos=%.1f%%; elapsed %.1fs",
        base_e["acc_at_k"][1],
        base_c["acc_at_k"][1],
        csls_sum["acc_at_k"][1],
        FRESH_DRAW_REF_ACC1,
        100 * gap["euclidean_primary"]["fraction"],
        100 * gap["cosine"]["fraction"],
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
