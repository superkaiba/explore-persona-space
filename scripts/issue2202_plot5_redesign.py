#!/usr/bin/env python3
"""Issue #2202 — paper Plot 5 redesign: failure set under whitened cosine + CSLS.

The committed paper figure (figures/paper/c3_failure_attribution, produced by
``issue2202_regen_figs.fig_c3_failure_analysis_iclr``) defines its failure
population under RAW EUCLIDEAN retrieval (1,829/9,941 rank-1 failures). The
project's standing retrieval convention is whitened cosine + CSLS (k=10), under
which far fewer contexts fail. This script recomputes the per-context mid-ranks
under that convention from the staged banked tensors, re-runs the P5 contrast
battery on the new failure indicator, and overwrites the paper figure in place.

Phases (checkpoint-per-phase; each persists before the next runs):

- ``ranks``    recompute per-context mid-ranks under whitened-cos+CSLS from
               /mnt/eps-data/.../issue2202_freshwhiten/{pred16,y_holdout_L19,
               whiten_stats}.npz, REUSING issue2202_metric_zoo (load_staged,
               banked-L chol whitening at lam=0.1, csls_ranks, ranks_summary);
               reconciliation-gated against the banked metric-zoo
               csls_k10_whitencos summary (acc@1 = 0.97616).
- ``oppoint``  per-row ranks at the OPERATING POINT (draw-AVERAGED targets,
               whitened cosine + CSLS, the 1,988 kresample-covered rows; the
               avgtgt_completion convention), reconciliation-gated against its
               banked matrix; persists the failing-row records (ids / ranks /
               labels only — never corpus text).
- ``ladder``   regime ladder of rank-1 failure counts across the banked
               scoring/target regimes (regime_ladder.json).
- ``stats``    the parent P5 instrument on the single-draw whitened-cos+CSLS
               fail indicator: issue2202_stats_figs.run_battery (22 registered
               contrasts, 10k batched bootstrap + 10k batched permutations,
               BH q=0.05) + attribution-class counts over that set.
- ``drafts``   BOTH candidate figure variants rendered as DRAFTS under
               figures/issue_2202/plot5_redesign/ (ladder-only, and ladder +
               per-kind panel at the single-draw whitened-cos+CSLS regime).
               The paper stem figures/paper/c3_failure_attribution.* is NOT
               written by this phase — promoting a chosen draft is a manual
               copy plus commit.

Reads only committed eval_results/issue_2202 artifacts + the read-only staged
copies; 0 GPU-h; runs on the shared VM under the standard thread caps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2202_failchar as FC  # noqa: E402  (atomic_json / meta_block / out_eval_dir)
import issue2202_metric_zoo as MZ  # noqa: E402  (load_staged / csls_ranks / ranks_summary)
import numpy as np  # noqa: E402

OUT_SUBDIR = "plot5_redesign"
BANKED_ARM = "csls_k10_whitencos"
ACC1_GATE_TOL = 2e-4  # |recomputed − banked| acc@1; ~2 contexts of mid-rank tie drift
MIN_COVERED_FOR_PANEL_B = 30  # below this the attribution panel is dropped as too thin
CONVENTION = (
    "whitened cosine + CSLS (k=10): z = L^-1(x - mu_A) at the task-locked shrunk "
    "train-answer covariance (lam=0.1, Cholesky basis), cosine scores, CSLS rescore "
    "score = S - 0.5*r_j (query-bank k-NN mean penalty); mid-rank ties per "
    "knn_retrieval; fail = mid-rank > 1"
)

# plain-English labels for every registered contrast (the current figure's dict,
# extended to the 9 non-significant contrasts so any set change stays renderable)
CONTRAST_LABEL = {
    "language=en": "English",
    "topic=factual_qa": "factual QA topic",
    "topic=creative_writing": "creative-writing topic",
    "topic=coding": "coding topic",
    "topic=advice_howto": "advice / how-to topic",
    "topic=chitchat_social": "chit-chat topic",
    "topic=translation": "translation topic",
    "topic=math": "math topic",
    "topic=summarization_extraction": "summarization topic",
    "topic=harmful_or_unsafe_request": "harmful-request topic",
    "topic=roleplay_persona": "roleplay / persona topic",
    "topic=nsfw": "NSFW topic",
    "topic=other": "'other' topic",
    "refusal_adjacent=yes": "refusal-adjacent request",
    "answer_is_refusal=yes": "answer is a refusal",
    "format=code": "code-format answer",
    "format=list": "list-format answer",
    "format=prose": "prose-format answer",
    "depth=2-2": "2-turn conversation",
    "depth=3-4": "3-4-turn conversation",
    "depth=>=5": "deep conversation (5+ turns)",
    "corpus=wildchat": "WildChat corpus",
}


def out_dir(args) -> Path:
    d = FC.out_eval_dir(args) / OUT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_csv_rows(args) -> list[dict]:
    path = FC.out_eval_dir(args) / "percontext_ranks.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def banked_arm_summary(args) -> dict:
    """The banked metric-zoo csls_k10_whitencos summary (reconciliation target)."""
    path = FC.out_eval_dir(args) / "metric_zoo" / "results.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("name") == BANKED_ARM:
                return row["summary"]
    raise RuntimeError(f"{BANKED_ARM} not found in {path}")


def phase_ranks(args) -> None:
    """Recompute whitened-cos+CSLS mid-ranks; reconcile vs the banked arm."""
    from scipy.linalg import solve_triangular

    staged = MZ.load_staged(Path(args.staged))
    pred, y16, pci = staged["pred"], staged["y16"], staged["pci"]
    ell, mu_a = staged["stats"]["L"], staged["stats"]["mu_A"]
    n = pred.shape[0]
    true_idx = np.arange(n)

    csv_rows = load_csv_rows(args)
    ci_csv = np.asarray([int(r["ci"]) for r in csv_rows], dtype=np.int64)
    assert np.array_equal(ci_csv, pci), "percontext_ranks.csv / pred16 ci misalign"

    # banked-L Cholesky whitening (identical to MZ.Transforms.chol_whiten at lam=0.1,
    # taken directly so no eigendecomposition is needed)
    pred_w = solve_triangular(ell, (pred - mu_a).T, lower=True).T
    pool_w = solve_triangular(ell, (y16 - mu_a).T, lower=True).T
    pwn = pred_w / (np.linalg.norm(pred_w, axis=1, keepdims=True) + 1e-12)
    qwn = pool_w / (np.linalg.norm(pool_w, axis=1, keepdims=True) + 1e-12)
    s_wc = pwn @ qwn.T
    print("[ranks] whitened-cos S computed", flush=True)
    ranks = MZ.csls_ranks(s_wc, true_idx, 0.5)
    rec = MZ.ranks_summary(ranks, n)

    banked = banked_arm_summary(args)
    deltas = {
        str(k): rec["acc_at_k"][int(k)] - float(banked["acc_at_k"][str(k)]) for k in (1, 5, 10)
    }
    if abs(deltas["1"]) > ACC1_GATE_TOL:
        raise RuntimeError(
            f"reconciliation gate FAILED: recomputed acc@1 {rec['acc_at_k'][1]:.6f} vs "
            f"banked {banked['acc_at_k']['1']:.6f} (|delta| > {ACC1_GATE_TOL})"
        )

    fail_new = ranks > 1.0
    fail_raw = np.asarray([r["fail_raw_euclidean"] == "1" for r in csv_rows])
    z = np.load(FC.out_eval_dir(args) / "csls_percontext_ranks.npz")
    assert np.array_equal(np.asarray(z["ci"], dtype=np.int64), pci)
    fail_ccos = np.asarray(z["rank_csls"]) > 1.0

    od = out_dir(args)
    tmp = od / "whitencos_csls_ranks.tmp.npz"
    np.savez(tmp, ci=pci, rank_whitencos_csls=ranks)
    tmp.replace(od / "whitencos_csls_ranks.npz")
    FC.atomic_json(
        od / "ranks_summary.json",
        {
            "convention": CONVENTION,
            "recomputed": rec,
            "banked": banked,
            "reconciliation_deltas_acc_at_k": deltas,
            "gate": {"tol_acc1": ACC1_GATE_TOL, "ok": True},
            "failure_sets": {
                "raw_euclidean": int(fail_raw.sum()),
                "cosine_csls": int(fail_ccos.sum()),
                "whitencos_csls": int(fail_new.sum()),
                "whitencos_csls_and_raw": int((fail_new & fail_raw).sum()),
                "whitencos_csls_new_vs_raw": int((fail_new & ~fail_raw).sum()),
                "whitencos_csls_and_cosine_csls": int((fail_new & fail_ccos).sum()),
                "whitencos_csls_new_vs_cosine_csls": int((fail_new & ~fail_ccos).sum()),
            },
            "staged_inputs": str(args.staged),
            "meta": FC.meta_block(),
        },
    )
    print(
        f"[ranks] done: acc@1={rec['acc_at_k'][1]:.6f} (banked delta {deltas['1']:+.2e}); "
        f"failures {int(fail_new.sum())}/{n}",
        flush=True,
    )


def phase_oppoint(args) -> None:
    """Operating-point ranks: draw-AVERAGED targets, whitened cosine + CSLS
    (ridge, the 1,988 kresample-covered rows against the 9,941 pool) — the
    avgtgt_completion convention, reconciliation-gated against its banked
    matrix cells (single AND avg). Persists per-covered-row ranks plus the
    failing-row records (ids / ranks / labels only — never corpus text)."""
    import issue1738_characterize as CH
    import issue2202_labels as LB
    from scipy.linalg import solve_triangular

    staged = Path(args.staged)
    st = MZ.load_staged(staged)
    pred, y16, pci = st["pred"], st["y16"], st["pci"]
    ell, mu_a = st["stats"]["L"], st["stats"]["mu_A"]
    n_pool = y16.shape[0]
    full_idx = np.arange(n_pool)

    kns = argparse.Namespace(
        local_kresample_dir=str(staged / "kresample"),
        scratch=str(staged / "scratch"),
        hf_prefix="",
    )
    kci, vres = CH._load_kresample_v(kns, [FC.LAYER])
    n_cov, k_draws = vres.shape[0], vres.shape[1]
    assert vres.shape == (1988, 4, 1, y16.shape[1]), vres.shape
    draws = vres[:, :, 0, :].astype(np.float64)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)

    def _wh(x: np.ndarray) -> np.ndarray:
        return solve_triangular(ell, (np.asarray(x, np.float64) - mu_a).T, lower=True).T

    def _norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    # draw-averaged whitened pool (covered rows replaced), full query bank
    avg = (y16[pos] + draws.sum(axis=1)) / (1 + k_draws)
    pool_modw = _wh(y16)
    pool_modw[pos] = _wh(avg)
    pwn = _norm(_wh(pred))
    s_wc = pwn @ _norm(pool_modw).T
    print("[oppoint] whitened-cos S (avg pool) computed", flush=True)
    k = MZ.K_LOCAL
    r_p = np.partition(s_wc, n_pool - k, axis=0)[n_pool - k :, :].mean(axis=0)
    score = s_wc - 0.5 * r_p[None, :]
    ranks_avg_full = MZ.ranks_score_matrix(score, full_idx)
    ranks_avg = ranks_avg_full[pos]

    od = out_dir(args)
    zs = np.load(od / "whitencos_csls_ranks.npz")
    ranks_single = np.asarray(zs["rank_whitencos_csls"])[pos]

    # reconciliation vs the banked avgtgt matrix (ridge / csls_k10_whitencos)
    avgdoc = json.loads((FC.out_eval_dir(args) / "avgtgt_completion" / "summary.json").read_text())
    cell = avgdoc["matrix"]["ridge"]["csls_k10_whitencos"]
    rec = {
        "avg": {
            "recomputed_acc1": float((ranks_avg <= 1).mean()),
            "banked": cell["avg"]["acc_at_k"]["1"],
        },
        "single": {
            "recomputed_acc1": float((ranks_single <= 1).mean()),
            "banked": cell["single"]["acc_at_k"]["1"],
        },
    }
    for nm, r in rec.items():
        r["delta"] = r["recomputed_acc1"] - float(r["banked"])
        if abs(r["delta"]) > ACC1_GATE_TOL * (n_pool / n_cov):  # same +-2-row allowance
            raise RuntimeError(f"oppoint reconciliation gate FAILED ({nm}): {r}")

    # failing rows at the operating point (avg targets)
    labels = LB.load_labels_1738(args)
    fields = {int(kk): v for kk, v in LB.resolve_ci_fields(args).items()}
    kres_by_ci = {int(r_["ci"]): r_["kres_class"] for r_ in load_csv_rows(args)}
    pos_set = set(pos.tolist())
    fail_rows = []
    for i in np.flatnonzero(ranks_avg > 1.0):
        p = int(pos[i])
        ci = int(kci[i])
        row_score = score[p]
        top1 = int(np.argmax(row_score))
        lab = labels.get(str(ci), {})
        fail_rows.append(
            {
                "ci": ci,
                "rank_avg": float(ranks_avg[i]),
                "rank_single": float(ranks_single[i]),
                "top1_ci": int(pci[top1]),
                "top1_is_covered": bool(top1 in pos_set),
                "score_margin_true_minus_top1": float(row_score[p] - row_score[top1]),
                "labels_1738": {
                    kk: lab.get(kk)
                    for kk in (
                        "topic",
                        "language",
                        "format",
                        "request_refusal_adjacent",
                        "answer_is_refusal",
                    )
                },
                "kres_class": kres_by_ci.get(ci),
                "corpus": fields[ci]["corpus"],
                "depth": fields[ci]["depth"],
            }
        )
    fail_rows.sort(key=lambda r_: -r_["rank_avg"])

    tmp = od / "oppoint_ranks.tmp.npz"
    np.savez(tmp, ci=kci, pos=pos, rank_avg=ranks_avg, rank_single=ranks_single)
    tmp.replace(od / "oppoint_ranks.npz")
    FC.atomic_json(
        od / "oppoint_failures.json",
        {
            "convention": CONVENTION
            + "; targets DRAW-AVERAGED (covered pool rows replaced by mean(original + 4 "
            "fresh on-policy draws), the avgtgt_completion convention); eval on the 1,988 "
            "kresample-covered rows; pool stays 9,941",
            "reconciliation": rec,
            "n_covered": int(n_cov),
            "n_fail_avg": int((ranks_avg > 1.0).sum()),
            "n_fail_single_covered": int((ranks_single > 1.0).sum()),
            "failures": fail_rows,
            "meta": FC.meta_block(),
        },
    )
    print(
        f"[oppoint] done: avg acc@1={rec['avg']['recomputed_acc1']:.6f} "
        f"(banked delta {rec['avg']['delta']:+.2e}); failures avg={int((ranks_avg > 1).sum())} "
        f"single-covered={int((ranks_single > 1).sum())} of {n_cov}",
        flush=True,
    )


def phase_ladder(args) -> None:
    """Regime ladder of rank-1 failure counts (ridge map, banked + this round's
    recomputes): raw euclidean -> raw cosine -> centered cosine -> whitened
    cosine -> CSLS on raw cosine -> whitened+CSLS -> whitened+CSLS with
    draw-averaged targets. Two denominators, stated per rung: the full 9,941
    holdout pool, and the 1,988 kresample-covered rows (the only rows with
    fresh draws, hence the only rows the averaged-target regime can score)."""
    od = out_dir(args)
    csv_rows = load_csv_rows(args)
    n = len(csv_rows)
    z = np.load(FC.out_eval_dir(args) / "csls_percontext_ranks.npz")
    zs = np.load(od / "whitencos_csls_ranks.npz")
    op = json.loads((od / "oppoint_failures.json").read_text())

    def _csv_count(col: str) -> int:
        return sum(1 for r in csv_rows if r[col] == "1")

    rungs = [
        {
            "regime": "raw euclidean, single-draw targets",
            "n_fail": _csv_count("fail_raw_euclidean"),
            "n": n,
        },
        {"regime": "raw cosine, single-draw targets", "n_fail": _csv_count("fail_raw_cos"), "n": n},
        {
            "regime": "centered cosine, single-draw targets",
            "n_fail": _csv_count("fail_cent_cos"),
            "n": n,
        },
        {
            "regime": "whitened cosine, single-draw targets",
            "n_fail": _csv_count("fail_whiten_cos"),
            "n": n,
        },
        {
            "regime": "CSLS on raw cosine, single-draw targets",
            "n_fail": int((np.asarray(z["rank_csls"]) > 1.0).sum()),
            "n": n,
        },
        {
            "regime": "whitened cosine + CSLS, single-draw targets",
            "n_fail": int((np.asarray(zs["rank_whitencos_csls"]) > 1.0).sum()),
            "n": n,
        },
        {
            "regime": "whitened cosine + CSLS, single-draw targets (covered rows)",
            "n_fail": op["n_fail_single_covered"],
            "n": op["n_covered"],
        },
        {
            "regime": "whitened cosine + CSLS, draw-AVERAGED targets (covered rows)",
            "n_fail": op["n_fail_avg"],
            "n": op["n_covered"],
        },
    ]
    for r in rungs:
        r["fail_rate"] = r["n_fail"] / r["n"]
    FC.atomic_json(
        od / "regime_ladder.json",
        {
            "map": "ridge (pred16, layer 19)",
            "note": (
                "whitened euclidean is the known-degenerate outlier "
                f"({_csv_count('fail_whiten')}/{n} fail) and is excluded from the ladder; "
                "per-rung sources: percontext_ranks.csv fail_* columns (pod P1), "
                "csls_percontext_ranks.npz (cosine-native CSLS follow-up), this round's "
                "whitencos_csls_ranks.npz + oppoint_ranks.npz"
            ),
            "rungs": rungs,
            "meta": FC.meta_block(),
        },
    )
    for r in rungs:
        print(
            f"[ladder] {r['regime']:60s} {r['n_fail']:5d}/{r['n']} ({r['fail_rate']:.4f})",
            flush=True,
        )


def phase_stats(args) -> None:
    """P5 contrast battery + attribution-class counts on the new failure set."""
    import issue1738_characterize as CH
    import issue2202_labels as LB
    import issue2202_stats_figs as SF

    od = out_dir(args)
    z = np.load(od / "whitencos_csls_ranks.npz")
    ranks, pci = np.asarray(z["rank_whitencos_csls"]), np.asarray(z["ci"], dtype=np.int64)
    fail_new = (ranks > 1.0).astype(float)

    csv_rows = load_csv_rows(args)
    ci_rows = np.asarray([int(r["ci"]) for r in csv_rows], dtype=np.int64)
    assert np.array_equal(ci_rows, pci)
    labels = LB.load_labels_1738(args)
    fields = {int(k): v for k, v in LB.resolve_ci_fields(args).items()}
    masks = CH._contrast_masks(ci_rows, labels, fields)
    n_boot = 200 if args.smoke else SF.N_BOOT
    n_perm = 200 if args.smoke else SF.N_PERM
    battery = SF.run_battery(fail_new, masks, n_boot, n_perm, SF.STAT_SEED)

    # attribution classes (banked kres_class; covered = the 1,988 kresample contexts)
    fail_mask = fail_new.astype(bool)
    counts: dict[str, int] = {}
    for r, f in zip(csv_rows, fail_mask):
        if f:
            counts[r["kres_class"]] = counts.get(r["kres_class"], 0) + 1
    covered = sum(v for k, v in counts.items() if k != "UNKNOWN")

    comp = json.loads((FC.out_eval_dir(args) / "composition_stats.json").read_text())
    banked_sig = [r["contrast"] for r in comp["banked_battery"] if r["bh_significant"]]
    new_sig = [r["contrast"] for r in battery if r["bh_significant"]]
    FC.atomic_json(
        od / "plot5_stats.json",
        {
            "convention": CONVENTION,
            "n_fail": int(fail_mask.sum()),
            "n_boot": n_boot,
            "n_perm": n_perm,
            "bh_q": SF.BH_Q,
            "seed": SF.STAT_SEED,
            "battery": battery,
            "bh_significant_new": new_sig,
            "bh_significant_banked_raw_euclidean": banked_sig,
            "lost_significance_vs_raw": sorted(set(banked_sig) - set(new_sig)),
            "gained_significance_vs_raw": sorted(set(new_sig) - set(banked_sig)),
            "attribution_over_new_failures": {
                "counts": counts,
                "n_covered": covered,
                "n_fail": int(fail_mask.sum()),
                "coverage_note": (
                    "per-failure attribution classes exist only for the 1,988 "
                    "kresample-covered contexts (banked kres_class); shares in the "
                    "figure are over the covered subset of the new failures"
                ),
            },
            "meta": FC.meta_block({"smoke": bool(args.smoke)}),
        },
    )
    print(
        f"[stats] done: {len(new_sig)}/{len(battery)} BH-significant; "
        f"attribution coverage {covered}/{int(fail_mask.sum())}",
        flush=True,
    )


LADDER_SHORT = {
    "raw euclidean, single-draw targets": "raw euclidean",
    "raw cosine, single-draw targets": "raw cosine",
    "centered cosine, single-draw targets": "centered cosine",
    "whitened cosine, single-draw targets": "whitened cosine",
    "CSLS on raw cosine, single-draw targets": "CSLS on raw cosine",
    "whitened cosine + CSLS, single-draw targets": "whitened cosine + CSLS",
    "whitened cosine + CSLS, single-draw targets (covered rows)": (
        "whitened cos + CSLS, covered rows"
    ),
    "whitened cosine + CSLS, draw-AVERAGED targets (covered rows)": ("and draw-averaged targets"),
}


DRAFT_DIR_REL = "issue_2202/plot5_redesign"  # under figures/; NOT the paper stem


def phase_drafts(args) -> None:
    """Render BOTH candidate figure variants as DRAFTS under
    figures/issue_2202/plot5_redesign/ — never the paper stem
    (figures/paper/c3_failure_attribution.*), whose swap is a user decision;
    promotion is then a copy plus a commit.

    - plot5_draft_ladder_only: single panel, the regime ladder of rank-1
      failure rates (the 11 operating-point failures live as its last rung;
      the rows themselves are reported as text, oppoint_failures.json).
    - plot5_draft_ladder_perkind: the ladder plus the per-context-kind
      failure-rate contrasts at the richest paper-convention regime that
      still has enough failures (single-draw whitened cosine + CSLS,
      237 failures; per-kind tick labels carry exact group failure counts)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    od = out_dir(args)
    stats = json.loads((od / "plot5_stats.json").read_text())
    comp = json.loads((FC.out_eval_dir(args) / "composition_stats.json").read_text())
    ladder = json.loads((od / "regime_ladder.json").read_text())
    (PROJECT_ROOT / "figures" / DRAFT_DIR_REL).mkdir(parents=True, exist_ok=True)

    def draw_ladder(ax) -> None:
        """Rank-1 failure rate per scoring/target regime (log x, exact counts)."""
        rungs = sorted(ladder["rungs"], key=lambda r: -r["fail_rate"])
        ys = np.arange(len(rungs))[::-1]
        x_lo = 0.3
        for y, r in zip(ys, rungs):
            is_avg = "AVERAGED" in r["regime"]
            color = paper_color("instruct") if is_avg else paper_color("null")
            rate = r["fail_rate"] * 100
            ax.hlines(y, x_lo, rate, color="0.85", lw=0.8, zorder=1)
            ax.scatter([rate], [y], color=color, s=16, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(x_lo, 30)
        ax.set_yticks(
            ys,
            [f"{LADDER_SHORT[r['regime']]} ({r['n_fail']:,}/{r['n']:,})" for r in rungs],
            fontsize=6.5,
        )
        ax.set_xlabel("rank-1 failure rate (%, log)")
        ax.set_title("scoring/target regime ladder (ridge map)", fontsize=7)
        handles = [
            Patch(color=paper_color("null"), label="single-draw targets"),
            Patch(color=paper_color("instruct"), label="draw-averaged targets"),
        ]
        ax.legend(
            handles=handles,
            fontsize=6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncols=2,
            frameon=False,
        )

    def draw_perkind(ax) -> None:
        """Per-context-kind contrasts at single-draw whitened cos + CSLS: the
        union of the raw-euclidean-significant contrasts and those significant
        under the new failure set, all recomputed on the new set; tick labels
        carry the exact per-group failure counts."""
        keep = {r["contrast"] for r in comp["banked_battery"] if r["bh_significant"]}
        keep |= set(stats["bh_significant_new"])
        by_name = {r["contrast"]: r for r in stats["battery"]}
        rows = sorted((by_name[c] for c in keep), key=lambda r: r["delta"])
        ys = np.arange(len(rows))
        deltas = np.array([r["delta"] for r in rows]) * 100
        elo = np.maximum(0.0, np.array([r["delta"] - r["ci_lo"] for r in rows]) * 100)
        ehi = np.maximum(0.0, np.array([r["ci_hi"] - r["delta"] for r in rows]) * 100)
        sig = np.array([bool(r["bh_significant"]) for r in rows])
        colors = [paper_color("instruct") if s else "0.78" for s in sig]
        ax.barh(
            ys,
            deltas,
            xerr=(elo, ehi),
            color=colors,
            height=0.62,
            error_kw={"lw": 0.7, "capsize": 1.5},
        )
        ax.axvline(0, color=paper_color("reference"), lw=0.7)
        labels = []
        for r in rows:
            k_grp = int(round(r["fail_rate_group"] * r["n_group"]))
            labels.append(f"{CONTRAST_LABEL[r['contrast']]} ({k_grp:,}/{r['n_group']:,})")
        ax.set_yticks(ys, labels, fontsize=6.5)
        ax.set_xlabel("failure-rate difference (pp)")
        n_fail = stats["n_fail"]
        ax.set_title(
            f"failure kinds, single-draw whitened cos + CSLS ({n_fail}/9,941 fail)",
            fontsize=7,
        )
        handles = [
            Patch(color=paper_color("instruct"), label="significant (BH q=0.05)"),
            Patch(color="0.78", label="not significant"),
        ]
        ax.legend(
            handles=handles,
            fontsize=6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncols=2,
            frameon=False,
        )

    set_paper_style("iclr")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    draw_ladder(ax)
    savefig_paper(fig, f"{DRAFT_DIR_REL}/plot5_draft_ladder_only", dir="figures/")
    plt.close(fig)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.9, 2.6), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    draw_ladder(ax_a)
    draw_perkind(ax_b)
    savefig_paper(fig, f"{DRAFT_DIR_REL}/plot5_draft_ladder_perkind", dir="figures/")
    plt.close(fig)
    print(
        f"[drafts] wrote figures/{DRAFT_DIR_REL}/plot5_draft_ladder_only.* and "
        f"plot5_draft_ladder_perkind.* (paper stem NOT touched)",
        flush=True,
    )


PHASES = {
    "ranks": phase_ranks,
    "oppoint": phase_oppoint,
    "ladder": phase_ladder,
    "stats": phase_stats,
    "drafts": phase_drafts,
}
PHASE_ORDER = ["ranks", "oppoint", "ladder", "stats", "drafts"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2202 paper Plot 5 CSLS redesign")
    ap.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--import-check", action="store_true", dest="import_check")
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_2202"))
    ap.add_argument("--staged", default=MZ.STAGED_DEFAULT)
    ap.add_argument("--hf-prefix", default=FC.HF_PREFIX_2202)
    ap.add_argument(
        "--labels-1738",
        default="eval_results/issue_1738/judge_labels/labels.json",
        dest="labels_1738",
    )
    ap.add_argument(
        "--ci-fields",
        default=str(PROJECT_ROOT / "data" / "issue_2202" / "ci_fields.json"),
        dest="ci_fields",
    )
    ap.add_argument("--work-root", default="/workspace/data/issue_2202")
    return ap


def _import_check() -> None:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("import-check OK: issue2202_plot5_redesign")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check)")
    for ph in PHASE_ORDER if args.phase == "all" else [args.phase]:
        PHASES[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
